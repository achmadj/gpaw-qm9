#!/usr/bin/env python3
"""
Analyze whether HDF5 shard datasets are reasonably safe to store in float32.

This script is intended for GPAW shard files that contain per-molecule groups
with datasets such as:
- n_r
- v_ext

What it does
------------
For each requested dataset, it computes:
- min / max / mean / std
- counts of zeros, NaNs, infinities
- smallest nonzero absolute value
- largest absolute value
- float32 round-trip absolute and relative error statistics
- a heuristic recommendation: likely safe / caution / keep float64

Important note
--------------
This script does NOT prove scientific equivalence. It only measures numerical
loss from casting stored values to float32 and back. For ML training inputs,
float32 is often fine, but you should still validate downstream model quality.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class DatasetAggregate:
    name: str
    groups_seen: int = 0
    arrays_seen: int = 0
    total_elements: int = 0
    finite_elements: int = 0
    zero_count: int = 0
    nan_count: int = 0
    inf_count: int = 0

    min_value: float = math.inf
    max_value: float = -math.inf
    min_abs_nonzero: float = math.inf
    max_abs_value: float = 0.0

    sum_value: float = 0.0
    sum_sq_value: float = 0.0

    abs_err_max: float = 0.0
    abs_err_sum: float = 0.0
    abs_err_sq_sum: float = 0.0

    rel_err_max: float = 0.0
    rel_err_sum: float = 0.0
    rel_err_sq_sum: float = 0.0
    rel_err_count: int = 0

    underflow_to_zero_count: int = 0
    exact_equal_count: int = 0

    sample_extreme_small: List[Tuple[str, float]] = None
    sample_extreme_large: List[Tuple[str, float]] = None

    def __post_init__(self) -> None:
        if self.sample_extreme_small is None:
            self.sample_extreme_small = []
        if self.sample_extreme_large is None:
            self.sample_extreme_large = []


def human_int(value: int) -> str:
    return f"{value:,}"


def human_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8e}"


def safe_sqrt(value: float) -> float:
    return math.sqrt(value) if value > 0 else 0.0


def update_sample_small(
    samples: List[Tuple[str, float]], group_name: str, value: float, limit: int = 10
) -> None:
    if not math.isfinite(value):
        return
    samples.append((group_name, value))
    samples.sort(key=lambda item: item[1])
    del samples[limit:]


def update_sample_large(
    samples: List[Tuple[str, float]], group_name: str, value: float, limit: int = 10
) -> None:
    if not math.isfinite(value):
        return
    samples.append((group_name, value))
    samples.sort(key=lambda item: item[1], reverse=True)
    del samples[limit:]


def analyze_array(data64: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(data64, dtype=np.float64).reshape(-1)

    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]

    stats: Dict[str, float] = {
        "total_elements": int(flat.size),
        "nan_count": int(nan_mask.sum()),
        "inf_count": int(inf_mask.sum()),
        "finite_elements": int(finite.size),
    }

    if finite.size == 0:
        stats.update(
            {
                "zero_count": 0,
                "min_value": math.nan,
                "max_value": math.nan,
                "sum_value": 0.0,
                "sum_sq_value": 0.0,
                "min_abs_nonzero": math.inf,
                "max_abs_value": 0.0,
                "abs_err_max": math.nan,
                "abs_err_sum": 0.0,
                "abs_err_sq_sum": 0.0,
                "rel_err_max": math.nan,
                "rel_err_sum": 0.0,
                "rel_err_sq_sum": 0.0,
                "rel_err_count": 0,
                "underflow_to_zero_count": 0,
                "exact_equal_count": 0,
            }
        )
        return stats

    abs_finite = np.abs(finite)
    zero_mask = finite == 0.0
    nonzero_mask = ~zero_mask
    nonzero_abs = abs_finite[nonzero_mask]

    finite32 = finite.astype(np.float32)
    roundtrip64 = finite32.astype(np.float64)

    abs_err = np.abs(roundtrip64 - finite)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.zeros_like(abs_err)
        rel_mask = nonzero_mask
        rel_err[rel_mask] = abs_err[rel_mask] / nonzero_abs

    underflow_to_zero_count = int(np.logical_and(nonzero_mask, finite32 == 0.0).sum())
    exact_equal_count = int((roundtrip64 == finite).sum())

    stats.update(
        {
            "zero_count": int(zero_mask.sum()),
            "min_value": float(finite.min()),
            "max_value": float(finite.max()),
            "sum_value": float(finite.sum(dtype=np.float64)),
            "sum_sq_value": float(
                np.square(finite, dtype=np.float64).sum(dtype=np.float64)
            ),
            "min_abs_nonzero": float(nonzero_abs.min())
            if nonzero_abs.size
            else math.inf,
            "max_abs_value": float(abs_finite.max()) if abs_finite.size else 0.0,
            "abs_err_max": float(abs_err.max()) if abs_err.size else 0.0,
            "abs_err_sum": float(abs_err.sum(dtype=np.float64)),
            "abs_err_sq_sum": float(
                np.square(abs_err, dtype=np.float64).sum(dtype=np.float64)
            ),
            "rel_err_max": float(rel_err[rel_mask].max()) if nonzero_abs.size else 0.0,
            "rel_err_sum": float(rel_err[rel_mask].sum(dtype=np.float64))
            if nonzero_abs.size
            else 0.0,
            "rel_err_sq_sum": float(
                np.square(rel_err[rel_mask], dtype=np.float64).sum(dtype=np.float64)
            )
            if nonzero_abs.size
            else 0.0,
            "rel_err_count": int(nonzero_abs.size),
            "underflow_to_zero_count": underflow_to_zero_count,
            "exact_equal_count": exact_equal_count,
        }
    )
    return stats


def update_aggregate(
    agg: DatasetAggregate, group_name: str, array_stats: Dict[str, float]
) -> None:
    agg.groups_seen += 1
    agg.arrays_seen += 1
    agg.total_elements += int(array_stats["total_elements"])
    agg.finite_elements += int(array_stats["finite_elements"])
    agg.zero_count += int(array_stats["zero_count"])
    agg.nan_count += int(array_stats["nan_count"])
    agg.inf_count += int(array_stats["inf_count"])

    if math.isfinite(array_stats["min_value"]):
        agg.min_value = min(agg.min_value, float(array_stats["min_value"]))
    if math.isfinite(array_stats["max_value"]):
        agg.max_value = max(agg.max_value, float(array_stats["max_value"]))

    min_abs_nonzero = float(array_stats["min_abs_nonzero"])
    if math.isfinite(min_abs_nonzero):
        agg.min_abs_nonzero = min(agg.min_abs_nonzero, min_abs_nonzero)
        update_sample_small(agg.sample_extreme_small, group_name, min_abs_nonzero)

    max_abs_value = float(array_stats["max_abs_value"])
    agg.max_abs_value = max(agg.max_abs_value, max_abs_value)
    if math.isfinite(max_abs_value):
        update_sample_large(agg.sample_extreme_large, group_name, max_abs_value)

    agg.sum_value += float(array_stats["sum_value"])
    agg.sum_sq_value += float(array_stats["sum_sq_value"])

    agg.abs_err_max = max(agg.abs_err_max, float(array_stats["abs_err_max"]))
    agg.abs_err_sum += float(array_stats["abs_err_sum"])
    agg.abs_err_sq_sum += float(array_stats["abs_err_sq_sum"])

    rel_err_max = float(array_stats["rel_err_max"])
    if math.isfinite(rel_err_max):
        agg.rel_err_max = max(agg.rel_err_max, rel_err_max)
    agg.rel_err_sum += float(array_stats["rel_err_sum"])
    agg.rel_err_sq_sum += float(array_stats["rel_err_sq_sum"])
    agg.rel_err_count += int(array_stats["rel_err_count"])

    agg.underflow_to_zero_count += int(array_stats["underflow_to_zero_count"])
    agg.exact_equal_count += int(array_stats["exact_equal_count"])


def compute_summary(agg: DatasetAggregate) -> Dict[str, float]:
    if agg.finite_elements == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "abs_err_mean": math.nan,
            "abs_err_rms": math.nan,
            "rel_err_mean": math.nan,
            "rel_err_rms": math.nan,
        }

    mean = agg.sum_value / agg.finite_elements
    variance = max(0.0, agg.sum_sq_value / agg.finite_elements - mean * mean)
    std = safe_sqrt(variance)

    abs_err_mean = agg.abs_err_sum / agg.finite_elements
    abs_err_rms = safe_sqrt(agg.abs_err_sq_sum / agg.finite_elements)

    if agg.rel_err_count > 0:
        rel_err_mean = agg.rel_err_sum / agg.rel_err_count
        rel_err_rms = safe_sqrt(agg.rel_err_sq_sum / agg.rel_err_count)
    else:
        rel_err_mean = math.nan
        rel_err_rms = math.nan

    return {
        "mean": mean,
        "std": std,
        "abs_err_mean": abs_err_mean,
        "abs_err_rms": abs_err_rms,
        "rel_err_mean": rel_err_mean,
        "rel_err_rms": rel_err_rms,
    }


def classify_fp32_safety(
    agg: DatasetAggregate, summary: Dict[str, float]
) -> Tuple[str, str]:
    """
    Heuristic recommendation only.

    Float32 characteristics:
    - smallest normal ~1.175e-38
    - epsilon near 1.0 ~1.19e-7

    Rules below are practical, not absolute scientific truth.
    """
    if agg.finite_elements == 0:
        return "UNKNOWN", "No finite values found."

    reasons: List[str] = []

    if agg.nan_count > 0 or agg.inf_count > 0:
        reasons.append(
            "Contains NaN/Inf values; precision conversion is not the main issue."
        )

    if math.isfinite(agg.min_abs_nonzero) and agg.min_abs_nonzero < 1e-35:
        reasons.append(
            "Contains extremely tiny nonzero values close to float32 underflow range."
        )
        return "KEEP_FLOAT64", " ".join(reasons)

    if agg.underflow_to_zero_count > 0:
        frac = agg.underflow_to_zero_count / max(
            1, agg.finite_elements - agg.zero_count
        )
        reasons.append(
            f"{agg.underflow_to_zero_count} nonzero values would underflow to zero ({frac:.3e} fraction)."
        )
        return "KEEP_FLOAT64", " ".join(reasons)

    rel_err_mean = summary["rel_err_mean"]
    rel_err_rms = summary["rel_err_rms"]

    if math.isfinite(rel_err_rms) and rel_err_rms > 1e-5:
        reasons.append(
            f"Relative RMS error is {rel_err_rms:.3e}, which is somewhat high."
        )
        return "CAUTION", " ".join(reasons)

    if math.isfinite(rel_err_mean) and rel_err_mean > 1e-6:
        reasons.append(f"Mean relative error is {rel_err_mean:.3e}.")
        return "CAUTION", " ".join(reasons)

    if agg.abs_err_max > 1e-3 and agg.max_abs_value > 1e3:
        reasons.append(
            "Absolute max error exceeds 1e-3, but this may still be acceptable if values are large and training is ML-focused."
        )
        return "LIKELY_SAFE_FOR_ML", " ".join(reasons)

    reasons.append(
        "No strong signs of float32 underflow risk; round-trip errors are small."
    )
    return "LIKELY_SAFE_FOR_ML", " ".join(reasons)


def iter_target_groups(
    h5f: h5py.File, max_groups: Optional[int]
) -> Iterable[Tuple[str, h5py.Group]]:
    count = 0
    for name in h5f.keys():
        obj = h5f[name]
        if isinstance(obj, h5py.Group):
            yield name, obj
            count += 1
            if max_groups is not None and count >= max_groups:
                break


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_samples(title: str, samples: List[Tuple[str, float]]) -> None:
    print(title)
    if not samples:
        print("  (none)")
        return
    for group_name, value in samples:
        print(f"  - {group_name}: {human_float(value)}")


def print_dataset_report(agg: DatasetAggregate) -> None:
    summary = compute_summary(agg)
    verdict, reason = classify_fp32_safety(agg, summary)

    print_header(f"Dataset: {agg.name}")
    print(f"Groups analyzed           : {human_int(agg.groups_seen)}")
    print(f"Total elements            : {human_int(agg.total_elements)}")
    print(f"Finite elements           : {human_int(agg.finite_elements)}")
    print(f"Zero count                : {human_int(agg.zero_count)}")
    print(f"NaN count                 : {human_int(agg.nan_count)}")
    print(f"Inf count                 : {human_int(agg.inf_count)}")

    print(f"Min value                 : {human_float(agg.min_value)}")
    print(f"Max value                 : {human_float(agg.max_value)}")
    print(f"Mean                      : {human_float(summary['mean'])}")
    print(f"Std                       : {human_float(summary['std'])}")
    print(f"Min |nonzero|             : {human_float(agg.min_abs_nonzero)}")
    print(f"Max |value|               : {human_float(agg.max_abs_value)}")

    print(
        f"Exact float32 round-trip  : {human_int(agg.exact_equal_count)} / {human_int(agg.finite_elements)}"
    )
    print(f"Abs error max             : {human_float(agg.abs_err_max)}")
    print(f"Abs error mean            : {human_float(summary['abs_err_mean'])}")
    print(f"Abs error RMS             : {human_float(summary['abs_err_rms'])}")
    print(f"Rel error max             : {human_float(agg.rel_err_max)}")
    print(f"Rel error mean            : {human_float(summary['rel_err_mean'])}")
    print(f"Rel error RMS             : {human_float(summary['rel_err_rms'])}")
    print(f"Underflow to zero count   : {human_int(agg.underflow_to_zero_count)}")

    print(f"Recommendation            : {verdict}")
    print(f"Reason                    : {reason}")

    print()
    print_samples("Smallest |nonzero| samples:", agg.sample_extreme_small)
    print()
    print_samples("Largest |value| samples:", agg.sample_extreme_large)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether n_r and v_ext in an HDF5 shard are numerically safe to cast to float32."
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="gpaw_qm9_shard_0.h5",
        help="Path to HDF5 shard file (default: gpaw_qm9_shard_0.h5)",
    )
    parser.add_argument(
        "--datasets",
        default="n_r,v_ext",
        help="Comma-separated dataset names to analyze (default: n_r,v_ext)",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Limit number of top-level groups analyzed for quick testing",
    )
    parser.add_argument(
        "--verbose-missing",
        action="store_true",
        help="Print group names when a requested dataset is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.h5_path):
        print(f"ERROR: file not found: {args.h5_path}", file=sys.stderr)
        return 1

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not dataset_names:
        print("ERROR: no dataset names requested.", file=sys.stderr)
        return 1

    print_header("Float32 Safety Analysis")
    print(f"File                       : {args.h5_path}")
    print(f"Size (bytes)               : {os.path.getsize(args.h5_path)}")
    print(f"Datasets requested         : {', '.join(dataset_names)}")
    print(
        f"Max groups                 : {args.max_groups if args.max_groups is not None else 'ALL'}"
    )

    aggregates: Dict[str, DatasetAggregate] = {
        name: DatasetAggregate(name=name) for name in dataset_names
    }

    group_count = 0
    with h5py.File(args.h5_path, "r") as h5f:
        print()
        print("File attributes:")
        if len(h5f.attrs) == 0:
            print("  (none)")
        else:
            for key, value in h5f.attrs.items():
                print(f"  - {key} = {value!r}")

        for group_name, group in iter_target_groups(h5f, args.max_groups):
            group_count += 1
            for ds_name in dataset_names:
                if ds_name not in group:
                    if args.verbose_missing:
                        print(f"Missing dataset {ds_name!r} in group {group_name}")
                    continue

                obj = group[ds_name]
                if not isinstance(obj, h5py.Dataset):
                    continue

                data = obj[...]
                stats = analyze_array(data)
                update_aggregate(aggregates[ds_name], group_name, stats)

    print()
    print(f"Top-level groups analyzed  : {human_int(group_count)}")

    for ds_name in dataset_names:
        print_dataset_report(aggregates[ds_name])

    print_header("Interpretation Notes")
    print("- Float32 is usually adequate for ML training inputs.")
    print(
        "- Very tiny values matter only if they are scientifically important in your downstream task."
    )
    print(
        "- If min |nonzero| is far above ~1e-38, float32 underflow is generally not a concern."
    )
    print(
        "- Relative error is usually more informative than absolute error when values span multiple scales."
    )
    print("- If your plan is ML only, LIKELY_SAFE_FOR_ML is often sufficient.")
    print(
        "- If your plan includes high-precision scientific reuse, keep a float64 archival copy."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
