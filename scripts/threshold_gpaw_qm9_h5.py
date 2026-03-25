#!/usr/bin/env python3
"""Create a thresholded QM9 HDF5 dataset from `gpaw_qm9_all_fp32.h5`."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

_tqdm_mod = importlib.import_module("tqdm") if importlib.util.find_spec("tqdm") else None


def tqdm(iterable, **kwargs):
    if _tqdm_mod is None:
        return iterable
    return _tqdm_mod.tqdm(iterable, **kwargs)


DEFAULT_INPUT = "/clusterfs/students/achmadjae/gpaw-qm9/dataset/gpaw_qm9_all.h5"
DEFAULT_OUTPUT = (
    "/clusterfs/students/achmadjae/gpaw-qm9/dataset/"
    "gpaw_qm9_all_vext-ge-neg0.5_nr-le-0.05_fp16.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold v_ext and n_r in the QM9 fp32 HDF5 dataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source HDF5 file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HDF5 file.")
    parser.add_argument(
        "--vext-cutoff",
        type=float,
        default=-0.5,
        help="Set `v_ext` values >= this cutoff to 0.",
    )
    parser.add_argument(
        "--nr-cutoff",
        type=float,
        default=0.05,
        help="Set `n_r` values <= this cutoff to 0.",
    )
    parser.add_argument(
        "--fp",
        choices=["fp16", "fp32", "fp64"],
        default="fp32",
        help="Floating point precision for the output dataset.",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression used for the output datasets.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level when `--compression gzip` is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def threshold_vext(array: np.ndarray, cutoff: float) -> tuple[np.ndarray, int]:
    mask = array >= cutoff
    out = np.array(array, copy=True)
    out[mask] = 0.0
    return out, int(mask.sum())


def threshold_nr(array: np.ndarray, cutoff: float) -> tuple[np.ndarray, int]:
    mask = array <= cutoff
    out = np.array(array, copy=True)
    out[mask] = 0.0
    return out, int(mask.sum())


def create_dataset(group, name: str, data: np.ndarray, args: argparse.Namespace) -> None:
    kwargs = {}
    if args.compression != "none":
        kwargs["compression"] = args.compression
    if args.compression == "gzip":
        kwargs["compression_opts"] = args.compression_level
    group.create_dataset(name, data=data, dtype=data.dtype, **kwargs)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and args.force:
        output_path.unlink()

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "vext_cutoff": args.vext_cutoff,
        "nr_cutoff": args.nr_cutoff,
        "groups_processed": 0,
        "v_ext_zeroed_voxels": 0,
        "n_r_zeroed_voxels": 0,
        "total_v_ext_voxels": 0,
        "total_n_r_voxels": 0,
    }

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["thresholded_from"] = str(input_path)
        dst.attrs["v_ext_zero_if_greater_equal"] = args.vext_cutoff
        dst.attrs["n_r_zero_if_less_equal"] = args.nr_cutoff

        keys = sorted(src.keys())
        for key in tqdm(keys, desc="Thresholding groups", dynamic_ncols=True):
            src_group = src[key]
            dst_group = dst.create_group(key)
            copy_attrs(src_group, dst_group)

            v_ext = np.asarray(src_group["v_ext"][...])
            n_r = np.asarray(src_group["n_r"][...])

            v_ext_thr, v_ext_zeroed = threshold_vext(v_ext, args.vext_cutoff)
            n_r_thr, n_r_zeroed = threshold_nr(n_r, args.nr_cutoff)

            if args.fp == "fp16":
                v_ext_thr = v_ext_thr.astype(np.float16)
                n_r_thr = n_r_thr.astype(np.float16)
            elif args.fp == "fp32":
                v_ext_thr = v_ext_thr.astype(np.float32)
                n_r_thr = n_r_thr.astype(np.float32)
            elif args.fp == "fp64":
                v_ext_thr = v_ext_thr.astype(np.float64)
                n_r_thr = n_r_thr.astype(np.float64)

            create_dataset(dst_group, "v_ext", v_ext_thr, args)
            create_dataset(dst_group, "n_r", n_r_thr, args)

            dst_group.attrs["v_ext_zero_if_greater_equal"] = args.vext_cutoff
            dst_group.attrs["n_r_zero_if_less_equal"] = args.nr_cutoff
            dst_group.attrs["v_ext_zeroed_voxels"] = v_ext_zeroed
            dst_group.attrs["n_r_zeroed_voxels"] = n_r_zeroed

            summary["groups_processed"] += 1
            summary["v_ext_zeroed_voxels"] += v_ext_zeroed
            summary["n_r_zeroed_voxels"] += n_r_zeroed
            summary["total_v_ext_voxels"] += int(v_ext.size)
            summary["total_n_r_voxels"] += int(n_r.size)

    summary["v_ext_zeroed_fraction"] = (
        summary["v_ext_zeroed_voxels"] / summary["total_v_ext_voxels"]
        if summary["total_v_ext_voxels"]
        else 0.0
    )
    summary["n_r_zeroed_fraction"] = (
        summary["n_r_zeroed_voxels"] / summary["total_n_r_voxels"]
        if summary["total_n_r_voxels"]
        else 0.0
    )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved thresholded dataset to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
