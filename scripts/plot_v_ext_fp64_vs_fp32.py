#!/usr/bin/env python3
"""
Plot side-by-side 3D comparisons of v_ext from:
1. original fp64 GPAW output
2. fp32 + downsampled GPAW output

This script is intended to live in:
    achmadjae/gpaw-qm9/gpaw_qm9_shards/plot_v_ext_fp64_vs_fp32.py

Default behavior
----------------
- Reads the original merged file:
    ../gpaw_qm9_all.h5
- Reads the transformed file:
    gpaw_qm9_all_fp32.h5
- Selects one molecule group by key
- Downsamples the fp64 v_ext by 2x2x2 mean so it matches the fp32 shape
- Plots both side by side as rasterized 3D scatter plots
- Applies alpha blending
- Saves a PNG

Why downsample fp64 before plotting?
------------------------------------
The fp32 file stores v_ext after downsampling to match n_r shape. To compare apples
to apples, the script downsamples the fp64 v_ext using the same 2x2x2 mean pooling.

Example usage
-------------
Run from the gpaw_qm9_shards directory:

    python plot_v_ext_fp64_vs_fp32.py
    python plot_v_ext_fp64_vs_fp32.py --group dsgdb9nsd_000001
    python plot_v_ext_fp64_vs_fp32.py --group dsgdb9nsd_000049 --quantile 0.10
    python plot_v_ext_fp64_vs_fp32.py --group dsgdb9nsd_000001 --mode both

Plot modes
----------
- negative: plot only low-potential voxels (default)
- positive: plot only high-potential voxels
- both: plot both low and high extremes

Notes
-----
- Matplotlib 3D scatter is used for compatibility.
- Rasterization is enabled on the scatter artist to reduce output weight.
- The script samples points if too many voxels pass the threshold.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side 3D v_ext plots for fp64 and fp32 HDF5 data."
    )
    parser.add_argument(
        "--fp64-path",
        default=str(PROJECT_ROOT / "dataset" / "raw" / "gpaw_qm9_all_serial_backup.h5"),
        help="Path to original fp64 HDF5 file.",
    )
    parser.add_argument(
        "--fp64-shard-glob",
        default=str(PROJECT_ROOT / "dataset" / "shards" / "gpaw_qm9_shard_*.h5"),
        help=(
            "Glob pattern for original fp64 shard HDF5 files. "
            "Used as fallback when the merged fp64 file is unreadable."
        ),
    )
    parser.add_argument(
        "--fp32-path",
        default=str(PROJECT_ROOT / "dataset" / "gpaw_qm9_all_fp32.h5"),
        help="Path to merged fp32 HDF5 file.",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="Plot only the fp32 dataset and skip any fp64 loading/comparison.",
    )
    parser.add_argument(
        "--group",
        default="dsgdb9nsd_000001",
        help="Molecule group key to plot.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: v_ext_fp64_vs_fp32_<group>.png",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.10,
        help=(
            "Quantile for selecting extreme values. "
            "For mode=negative, lower quantile is used. "
            "For mode=positive, upper quantile is used."
        ),
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=None,
        help=(
            "Fixed value threshold. For mode=negative, select values <= threshold. "
            "For mode=positive, select values >= threshold. For mode=both, pass a "
            "non-negative value and select |value| >= threshold."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["negative", "positive", "both"],
        default="negative",
        help="Which voxels to plot based on value extremes.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.30,
        help="Alpha transparency for scatter points.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=12.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of plotted voxels per panel after thresholding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point subsampling.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--cmap-neg",
        default="RdYlBu_r",
        help="Colormap for negative/extreme-low values.",
    )
    parser.add_argument(
        "--cmap-pos",
        default="viridis",
        help="Colormap for positive/extreme-high values.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / path,
        script_dir / path,
        script_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((Path.cwd() / path).resolve())


def resolve_glob(glob_str: str) -> str:
    glob_path = Path(glob_str)
    if glob_path.is_absolute():
        return glob_str

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / glob_path,
        script_dir / glob_path,
        script_dir.parent / glob_path,
    ]
    for candidate in candidates:
        if glob.glob(str(candidate)):
            return str(candidate)

    return str((script_dir.parent / glob_path).resolve())


def downsample_mean_2x2x2(
    v_ext: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    tx, ty, tz = target_shape
    expected_shape = (tx * 2, ty * 2, tz * 2)
    if tuple(v_ext.shape) != expected_shape:
        raise ValueError(
            f"Expected fp64 v_ext shape {expected_shape} from target shape {target_shape}, "
            f"but got {tuple(v_ext.shape)}"
        )
    return v_ext.reshape(tx, 2, ty, 2, tz, 2).mean(axis=(1, 3, 5), dtype=np.float64)


def require_group(h5obj: h5py.File | h5py.Group, key: str) -> h5py.Group:
    obj = h5obj[key]
    if not isinstance(obj, h5py.Group):
        raise TypeError(f"Expected {key!r} to be an HDF5 group")
    return obj


def require_dataset(h5group: h5py.Group, key: str) -> h5py.Dataset:
    obj = h5group[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected {key!r} to be an HDF5 dataset")
    return obj


def dataset_to_array(
    h5group: h5py.Group, key: str, dtype: np.dtype | type[np.float64] = np.float64
) -> np.ndarray:
    dataset = require_dataset(h5group, key)
    return np.asarray(dataset[...], dtype=dtype)


def load_vext_from_fp64_shards(shard_glob: str, group_name: str) -> np.ndarray:
    resolved_glob = resolve_glob(shard_glob)
    shard_paths = sorted(glob.glob(resolved_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No fp64 shard files matched pattern: {resolved_glob}")

    for shard_path in shard_paths:
        try:
            with h5py.File(shard_path, "r") as shard_file:
                if group_name not in shard_file:
                    continue

                group_obj = require_group(shard_file, group_name)
                return dataset_to_array(group_obj, "v_ext")
        except Exception:
            continue

    raise KeyError(
        f"Group {group_name!r} not found in any fp64 shard matching {shard_glob}"
    )


def load_vext_pair(
    fp64_path: str, fp64_shard_glob: str, fp32_path: str, group_name: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    fp32_path = resolve_path(fp32_path)
    fp64_path = resolve_path(fp64_path)

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(
            f"fp32 file not found: {fp32_path}. "
            "Pass --fp32-path explicitly if your file is elsewhere."
        )

    formula = group_name

    with h5py.File(fp32_path, "r") as f32:
        if group_name not in f32:
            raise KeyError(f"Group {group_name!r} not found in fp32 file.")

        group32 = require_group(f32, group_name)
        v32 = dataset_to_array(group32, "v_ext")

        formula_attr = group32.attrs.get("formula")
        if formula_attr is not None:
            if isinstance(formula_attr, bytes):
                formula = formula_attr.decode("utf-8")
            else:
                formula = str(formula_attr)

    v64_full = None

    if os.path.exists(fp64_path):
        try:
            with h5py.File(fp64_path, "r") as f64:
                if group_name in f64:
                    group64 = require_group(f64, group_name)
                    v64_full = dataset_to_array(group64, "v_ext")
                else:
                    raise KeyError(f"Group {group_name!r} not found in fp64 file.")
        except Exception:
            v64_full = None

    if v64_full is None:
        v64_full = load_vext_from_fp64_shards(fp64_shard_glob, group_name)

    v64 = downsample_mean_2x2x2(v64_full, v32.shape)
    return v64, v32, formula


def load_vext_fp32(fp32_path: str, group_name: str) -> Tuple[np.ndarray, str]:
    fp32_path = resolve_path(fp32_path)
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(
            f"fp32 file not found: {fp32_path}. "
            "Pass --fp32-path explicitly if your file is elsewhere."
        )

    formula = group_name
    with h5py.File(fp32_path, "r") as f32:
        if group_name not in f32:
            raise KeyError(f"Group {group_name!r} not found in fp32 file.")

        group32 = require_group(f32, group_name)
        v32 = dataset_to_array(group32, "v_ext")

        formula_attr = group32.attrs.get("formula")
        if formula_attr is not None:
            if isinstance(formula_attr, bytes):
                formula = formula_attr.decode("utf-8")
            else:
                formula = str(formula_attr)

    return v32, formula


def select_points(
    volume: np.ndarray,
    mode: str,
    quantile: float,
    value_threshold: float | None,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = volume.reshape(-1)

    if value_threshold is not None:
        if mode == "negative":
            mask = values <= value_threshold
        elif mode == "positive":
            mask = values >= value_threshold
        else:
            if value_threshold < 0:
                raise ValueError(
                    "--value-threshold must be non-negative when --mode=both."
                )
            mask = np.abs(values) >= value_threshold
    else:
        if mode == "negative":
            threshold = np.quantile(values, quantile)
            mask = values <= threshold
        elif mode == "positive":
            threshold = np.quantile(values, 1.0 - quantile)
            mask = values >= threshold
        else:
            low = np.quantile(values, quantile)
            high = np.quantile(values, 1.0 - quantile)
            mask = (values <= low) | (values >= high)

    flat_idx = np.flatnonzero(mask)

    if flat_idx.size == 0:
        raise ValueError(
            "No points selected for plotting. Try changing --quantile or --mode."
        )

    if flat_idx.size > max_points:
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(flat_idx, size=max_points, replace=False)

    x, y, z = np.unravel_index(flat_idx, volume.shape)
    selected_values = volume[x, y, z]

    coords = np.column_stack([x, y, z])
    return coords, selected_values


def scatter_extremes(
    ax,
    volume: np.ndarray,
    mode: str,
    quantile: float,
    value_threshold: float | None,
    max_points: int,
    seed: int,
    alpha: float,
    size: float,
    cmap_neg: str,
    cmap_pos: str,
    title: str,
) -> None:
    coords, values = select_points(
        volume=volume,
        mode=mode,
        quantile=quantile,
        value_threshold=value_threshold,
        max_points=max_points,
        seed=seed,
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X grid")
    ax.set_ylabel("Y grid")
    ax.set_zlabel("Z grid")

    if mode == "negative":
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap=cmap_neg,
            alpha=alpha,
            s=size,
            linewidths=0,
            rasterized=True,
        )
        sc.set_rasterized(True)
        plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="v_ext")
        return

    if mode == "positive":
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap=cmap_pos,
            alpha=alpha,
            s=size,
            linewidths=0,
            rasterized=True,
        )
        sc.set_rasterized(True)
        plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="v_ext")
        return

    if value_threshold is not None:
        neg_mask = values <= -abs(value_threshold)
        pos_mask = values >= abs(value_threshold)
    else:
        low_q = np.quantile(volume, quantile)
        high_q = np.quantile(volume, 1.0 - quantile)
        neg_mask = values <= low_q
        pos_mask = values >= high_q

    if np.any(neg_mask):
        sc_neg = ax.scatter(
            coords[neg_mask, 0],
            coords[neg_mask, 1],
            coords[neg_mask, 2],
            c=values[neg_mask],
            cmap=cmap_neg,
            alpha=alpha,
            s=size,
            linewidths=0,
            rasterized=True,
        )
        sc_neg.set_rasterized(True)
        plt.colorbar(sc_neg, ax=ax, shrink=0.55, pad=0.02, label="v_ext low")

    if np.any(pos_mask):
        sc_pos = ax.scatter(
            coords[pos_mask, 0],
            coords[pos_mask, 1],
            coords[pos_mask, 2],
            c=values[pos_mask],
            cmap=cmap_pos,
            alpha=alpha,
            s=size,
            linewidths=0,
            rasterized=True,
        )
        sc_pos.set_rasterized(True)
        plt.colorbar(sc_pos, ax=ax, shrink=0.55, pad=0.12, label="v_ext high")


def main() -> None:
    args = parse_args()

    if args.value_threshold is None and not (0.0 < args.quantile < 0.5):
        raise ValueError("--quantile must be in the open interval (0, 0.5).")

    out_path = args.out or f"v_ext_fp64_vs_fp32_{args.group}.png"

    if args.fp32_only:
        v32, formula = load_vext_fp32(
            fp32_path=args.fp32_path,
            group_name=args.group,
        )

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        scatter_extremes(
            ax=ax,
            volume=v32,
            mode=args.mode,
            quantile=args.quantile,
            value_threshold=args.value_threshold,
            max_points=args.max_points,
            seed=args.seed,
            alpha=args.alpha,
            size=args.size,
            cmap_neg=args.cmap_neg,
            cmap_pos=args.cmap_pos,
            title="fp32",
        )
    else:
        v64, v32, formula = load_vext_pair(
            fp64_path=args.fp64_path,
            fp64_shard_glob=args.fp64_shard_glob,
            fp32_path=args.fp32_path,
            group_name=args.group,
        )

        fig = plt.figure(figsize=(16, 7))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        scatter_extremes(
            ax=ax1,
            volume=v64,
            mode=args.mode,
            quantile=args.quantile,
            value_threshold=args.value_threshold,
            max_points=args.max_points,
            seed=args.seed,
            alpha=args.alpha,
            size=args.size,
            cmap_neg=args.cmap_neg,
            cmap_pos=args.cmap_pos,
            title="fp64",
        )

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        scatter_extremes(
            ax=ax2,
            volume=v32,
            mode=args.mode,
            quantile=args.quantile,
            value_threshold=args.value_threshold,
            max_points=args.max_points,
            seed=args.seed,
            alpha=args.alpha,
            size=args.size,
            cmap_neg=args.cmap_neg,
            cmap_pos=args.cmap_pos,
            title="fp32",
        )

    fig.suptitle(
        (f"Electrostatic Potential (v_ext) for {formula}\n"),
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
