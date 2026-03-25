#!/usr/bin/env python3
"""
Create a merged GPAW QM9 HDF5 file with:
1. float32 storage for volumetric arrays
2. downsampled v_ext so its shape matches n_r

Expected input layout
---------------------
A directory of shard files such as:
- gpaw_qm9_shard_0.h5
- gpaw_qm9_shard_1.h5
- ...
- gpaw_qm9_shard_23.h5

Each shard is expected to contain top-level molecule groups, each with:
- dataset: n_r
- dataset: v_ext
- attributes copied from the GPAW run

What this script writes
-----------------------
A single merged file, by default:
- gpaw_qm9_all_fp32.h5

Per molecule group:
- n_r stored as float32
- v_ext stored as float32 after 2x2x2 mean downsampling
- original group attributes copied over
- extra attributes documenting conversion and shapes

File-level attributes document the conversion process as well.

Usage
-----
Run from the gpaw_qm9_shards directory, for example:

    python build_gpaw_qm9_all_fp32.py

Or specify explicit paths:

    python build_gpaw_qm9_all_fp32.py \
        --input-glob "gpaw_qm9_shard_*.h5" \
        --out-path "gpaw_qm9_all_fp32.h5"

Notes
-----
- v_ext is downsampled with block mean over 2x2x2 cells.
- This assumes v_ext has exactly 2x the size of n_r along each axis.
- If shapes do not match that expectation, the script can either:
  - fail (default), or
  - fall back to interpolation-like block averaging after center-cropping
    if you extend the script yourself.
- This script is conservative and fails loudly on unexpected shape mismatches.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge GPAW shard HDF5 files into a single HDF5 with float32 arrays "
            "and downsampled v_ext matching n_r shape."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="gpaw_qm9_shard_*.h5",
        help="Glob pattern for shard HDF5 files.",
    )
    parser.add_argument(
        "--out-path",
        default="gpaw_qm9_all_fp32.h5",
        help="Output merged HDF5 path.",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression algorithm for output datasets.",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=4,
        help="Compression level for gzip (ignored for lzf/none).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable HDF5 shuffle filter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="Optional limit for number of molecule groups to process (0 = all).",
    )
    return parser.parse_args()


def resolve_compression(
    compression: str, compression_opts: int
) -> Tuple[str | None, int | None]:
    if compression == "none":
        return None, None
    if compression == "lzf":
        return "lzf", None
    return "gzip", compression_opts


def find_shard_paths(input_glob: str) -> List[str]:
    shard_paths = sorted(glob.glob(input_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched input glob: {input_glob}")
    return shard_paths


def downsample_mean_2x2x2(
    v_ext: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    if v_ext.ndim != 3:
        raise ValueError(f"Expected 3D v_ext, got shape {v_ext.shape}")

    tx, ty, tz = target_shape
    expected_shape = (tx * 2, ty * 2, tz * 2)

    if tuple(v_ext.shape) != expected_shape:
        raise ValueError(
            "v_ext shape does not match expected 2x n_r shape: "
            f"v_ext={tuple(v_ext.shape)} expected={expected_shape} from n_r={target_shape}"
        )

    reshaped = v_ext.reshape(tx, 2, ty, 2, tz, 2)
    down = reshaped.mean(axis=(1, 3, 5), dtype=np.float64)
    return down


def create_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    compression: str | None,
    compression_opts: int | None,
    shuffle: bool,
) -> h5py.Dataset:
    kwargs: Dict[str, object] = {
        "data": data,
        "shuffle": shuffle,
    }

    if compression is not None:
        kwargs["compression"] = compression
    if compression == "gzip" and compression_opts is not None:
        kwargs["compression_opts"] = compression_opts

    return group.create_dataset(name, **kwargs)


def copy_attrs(
    src_attrs: h5py.AttributeManager, dst_attrs: h5py.AttributeManager
) -> None:
    for key, value in src_attrs.items():
        dst_attrs[key] = value


def molecule_iter(
    shard_paths: Iterable[str],
    limit_groups: int = 0,
) -> Iterable[Tuple[str, str, h5py.Group]]:
    emitted = 0
    seen_names = set()

    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as shard_db:
            for group_name in shard_db.keys():
                if group_name in seen_names:
                    continue
                obj = shard_db[group_name]
                if not isinstance(obj, h5py.Group):
                    continue

                seen_names.add(group_name)
                yield shard_path, group_name, obj

                emitted += 1
                if limit_groups > 0 and emitted >= limit_groups:
                    return


def main() -> int:
    args = parse_args()

    compression, compression_opts = resolve_compression(
        args.compression, args.compression_opts
    )
    shuffle = not args.no_shuffle

    shard_paths = find_shard_paths(args.input_glob)

    if os.path.exists(args.out_path):
        if args.overwrite:
            os.remove(args.out_path)
        else:
            raise FileExistsError(
                f"Output already exists: {args.out_path}. Use --overwrite to replace it."
            )

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log("Starting merged fp32 build")
    log(f"Input glob: {args.input_glob}")
    log(f"Shard files found: {len(shard_paths)}")
    log(f"Output path: {args.out_path}")
    log(f"Compression: {compression or 'none'}")
    log(f"Shuffle: {shuffle}")
    if args.limit_groups > 0:
        log(f"Group limit: {args.limit_groups}")

    start_time = time.time()

    total_groups = 0
    total_groups_written = 0
    total_groups_failed = 0

    total_n_r_elements = 0
    total_v_ext_original_elements = 0
    total_v_ext_downsampled_elements = 0

    with h5py.File(args.out_path, "w") as out_db:
        out_db.attrs["description"] = (
            "Merged GPAW shard outputs with float32 storage and downsampled v_ext"
        )
        out_db.attrs["source_glob"] = args.input_glob
        out_db.attrs["num_shards_found"] = len(shard_paths)
        out_db.attrs["array_dtype"] = "float32"
        out_db.attrs["v_ext_downsampling"] = "mean_2x2x2"
        out_db.attrs["v_ext_target_shape"] = "match_n_r"
        out_db.attrs["created_by"] = "build_gpaw_qm9_all_fp32.py"
        out_db.attrs["compression"] = compression or "none"
        out_db.attrs["compression_opts"] = (
            -1 if compression_opts is None else compression_opts
        )
        out_db.attrs["shuffle"] = int(shuffle)
        out_db.attrs["build_start_unix"] = start_time

        for shard_path, group_name, in_group in molecule_iter(
            shard_paths, limit_groups=args.limit_groups
        ):
            total_groups += 1

            try:
                if "n_r" not in in_group:
                    raise KeyError(f"Missing dataset 'n_r' in group {group_name}")
                if "v_ext" not in in_group:
                    raise KeyError(f"Missing dataset 'v_ext' in group {group_name}")

                n_r = in_group["n_r"][...]
                v_ext = in_group["v_ext"][...]

                if n_r.ndim != 3:
                    raise ValueError(
                        f"n_r is not 3D in group {group_name}: shape={n_r.shape}"
                    )
                if v_ext.ndim != 3:
                    raise ValueError(
                        f"v_ext is not 3D in group {group_name}: shape={v_ext.shape}"
                    )

                v_ext_down = downsample_mean_2x2x2(v_ext, tuple(n_r.shape))

                n_r_fp32 = np.asarray(n_r, dtype=np.float32)
                v_ext_fp32 = np.asarray(v_ext_down, dtype=np.float32)

                out_group = out_db.create_group(group_name)

                create_dataset(
                    out_group,
                    "n_r",
                    n_r_fp32,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                )
                create_dataset(
                    out_group,
                    "v_ext",
                    v_ext_fp32,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                )

                copy_attrs(in_group.attrs, out_group.attrs)
                out_group.attrs["stored_dtype"] = "float32"
                out_group.attrs["v_ext_original_shape"] = np.asarray(
                    v_ext.shape, dtype=np.int64
                )
                out_group.attrs["v_ext_downsampled_shape"] = np.asarray(
                    v_ext_down.shape, dtype=np.int64
                )
                out_group.attrs["n_r_shape"] = np.asarray(n_r.shape, dtype=np.int64)
                out_group.attrs["v_ext_downsampling"] = "mean_2x2x2"
                out_group.attrs["source_shard_path"] = shard_path

                total_n_r_elements += int(n_r.size)
                total_v_ext_original_elements += int(v_ext.size)
                total_v_ext_downsampled_elements += int(v_ext_down.size)
                total_groups_written += 1

                if total_groups_written % 100 == 0:
                    elapsed = time.time() - start_time
                    log(
                        f"Processed {total_groups_written} groups "
                        f"(failed={total_groups_failed}) in {elapsed:.1f}s"
                    )

            except Exception as exc:
                total_groups_failed += 1
                log(f"FAIL group={group_name} shard={shard_path} error={exc}")

        end_time = time.time()

        out_db.attrs["total_groups_seen"] = total_groups
        out_db.attrs["total_molecules"] = total_groups_written
        out_db.attrs["total_groups_failed"] = total_groups_failed
        out_db.attrs["total_n_r_elements"] = total_n_r_elements
        out_db.attrs["total_v_ext_original_elements"] = total_v_ext_original_elements
        out_db.attrs["total_v_ext_downsampled_elements"] = (
            total_v_ext_downsampled_elements
        )
        out_db.attrs["build_end_unix"] = end_time
        out_db.attrs["build_elapsed_seconds"] = end_time - start_time

        if total_v_ext_original_elements > 0:
            out_db.attrs["v_ext_element_reduction_ratio"] = (
                total_v_ext_downsampled_elements / total_v_ext_original_elements
            )
            out_db.attrs["v_ext_element_saving_fraction"] = 1.0 - (
                total_v_ext_downsampled_elements / total_v_ext_original_elements
            )
        else:
            out_db.attrs["v_ext_element_reduction_ratio"] = math.nan
            out_db.attrs["v_ext_element_saving_fraction"] = math.nan

    elapsed = time.time() - start_time
    log("")
    log("Build finished")
    log(f"Total groups seen: {total_groups}")
    log(f"Total groups written: {total_groups_written}")
    log(f"Total groups failed: {total_groups_failed}")
    log(f"Elapsed: {elapsed:.1f}s")
    log(f"Output file: {args.out_path}")

    if total_groups_failed > 0:
        log("Some groups failed. Inspect the logs above.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
