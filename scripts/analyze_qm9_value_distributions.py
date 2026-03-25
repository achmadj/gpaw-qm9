#!/usr/bin/env python3
"""Analyze global `v_ext` and `n_r` value distributions in `gpaw_qm9_all_fp32.h5`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_H5 = str(PROJECT_ROOT / "dataset" / "gpaw_qm9_all_fp32.h5")
DEFAULT_OUT = str(PROJECT_ROOT / "gpaw_analysis_outputs" / "distributions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized-frequency histograms for QM9 `v_ext` and `n_r`."
    )
    parser.add_argument("--h5-path", default=DEFAULT_H5)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--target",
        choices=["v_ext", "n_r", "both"],
        default="both",
        help="Choose whether to analyze `v_ext` only, `n_r` only, or both.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["v_ext", "n_r"],
        help="Backward-compatible explicit dataset list. Overrides --target when provided.",
    )
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Optional fixed values for reporting exceedance fractions.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999],
        help="Quantiles to approximate from the global histogram CDF.",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=None,
        help=(
            "Optional threshold used to zero weak values before analysis and plotting. "
            "With --threshold-mode dataset-default: n_r values < eps are set to 0, "
            "and v_ext values > -eps are set to 0."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["dataset-default", "abs-below", "greater-than", "less-than"],
        default="dataset-default",
        help=(
            "Threshold rule to apply before histogramming. "
            "dataset-default: n_r < eps -> 0, v_ext > -eps -> 0; "
            "abs-below: |x| < eps -> 0; greater-than: x > eps -> 0; "
            "less-than: x < eps -> 0."
        ),
    )
    parser.add_argument(
        "--nonzero-only",
        action="store_true",
        help="Exclude zero-valued entries after thresholding when building statistics and plots.",
    )
    parser.add_argument("--log-y", action="store_true", help="Plot y-axis on a log scale.")
    return parser.parse_args()


def resolve_dataset_names(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return list(dict.fromkeys(args.datasets))
    if args.target == "both":
        return ["v_ext", "n_r"]
    return [args.target]


def apply_threshold(
    array: np.ndarray,
    dataset_name: str,
    threshold_value: float | None,
    threshold_mode: str,
) -> np.ndarray:
    if threshold_value is None:
        return array

    out = np.array(array, copy=True)
    eps = abs(float(threshold_value))

    if threshold_mode == "dataset-default":
        if dataset_name == "v_ext":
            out[out > -eps] = 0.0
        else:
            out[out < eps] = 0.0
        return out

    if threshold_mode == "abs-below":
        out[np.abs(out) < eps] = 0.0
    elif threshold_mode == "greater-than":
        out[out > threshold_value] = 0.0
    elif threshold_mode == "less-than":
        out[out < threshold_value] = 0.0
    return out


def build_output_suffix(args: argparse.Namespace) -> str:
    parts = []
    if args.threshold_value is not None:
        value_text = str(args.threshold_value).replace("-", "neg").replace(".", "p")
        parts.append(f"thr-{args.threshold_mode}-{value_text}")
    if args.nonzero_only:
        parts.append("nonzero")
    return "" if not parts else "_" + "_".join(parts)


def prepare_values(
    array: np.ndarray,
    dataset_name: str,
    threshold_value: float | None,
    threshold_mode: str,
    nonzero_only: bool,
) -> np.ndarray:
    values = apply_threshold(array, dataset_name, threshold_value, threshold_mode).reshape(-1)
    if nonzero_only:
        values = values[values != 0.0]
    return values


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def first_pass(
    h5_path: str,
    dataset_names: list[str],
    threshold_value: float | None = None,
    threshold_mode: str = "dataset-default",
    nonzero_only: bool = False,
) -> dict[str, dict[str, float]]:
    stats = {
        name: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
        }
        for name in dataset_names
    }

    with h5py.File(h5_path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            for name in dataset_names:
                array = np.asarray(group[name][...], dtype=np.float64)
                flat = prepare_values(
                    array,
                    name,
                    threshold_value,
                    threshold_mode,
                    nonzero_only,
                )
                if flat.size == 0:
                    continue
                entry = stats[name]
                entry["count"] += int(flat.size)
                entry["sum"] += float(flat.sum())
                entry["sum_sq"] += float(np.square(flat).sum())
                entry["min"] = min(entry["min"], float(flat.min()))
                entry["max"] = max(entry["max"], float(flat.max()))

    for entry in stats.values():
        if entry["count"] == 0:
            entry["min"] = float("nan")
            entry["max"] = float("nan")
            entry["mean"] = float("nan")
            entry["std"] = float("nan")
        else:
            count = entry["count"]
            mean = entry["sum"] / count
            variance = max(entry["sum_sq"] / count - mean ** 2, 0.0)
            entry["mean"] = mean
            entry["std"] = variance ** 0.5
    return stats


def second_pass(
    h5_path: str,
    dataset_names: list[str],
    edges_by_dataset: dict[str, np.ndarray],
    threshold_value: float | None = None,
    threshold_mode: str = "dataset-default",
    nonzero_only: bool = False,
):
    counts = {name: np.zeros(len(edges) - 1, dtype=np.int64) for name, edges in edges_by_dataset.items()}
    threshold_stats = {
        name: {
            "below_or_equal": {},
            "above_or_equal": {},
            "abs_above_or_equal": {},
        }
        for name in dataset_names
    }

    with h5py.File(h5_path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            for name in dataset_names:
                flat = prepare_values(
                    np.asarray(group[name][...], dtype=np.float64),
                    name,
                    threshold_value,
                    threshold_mode,
                    nonzero_only,
                )
                if flat.size == 0:
                    continue
                counts[name] += np.histogram(flat, bins=edges_by_dataset[name])[0]
    return counts, threshold_stats


def compute_threshold_fractions(
    h5_path: str,
    dataset_names: list[str],
    thresholds: list[float],
    threshold_value: float | None = None,
    threshold_mode: str = "dataset-default",
    nonzero_only: bool = False,
):
    summary = {
        name: {
            "below_or_equal": {str(t): 0 for t in thresholds},
            "above_or_equal": {str(t): 0 for t in thresholds},
            "abs_above_or_equal": {str(t): 0 for t in thresholds},
            "count": 0,
        }
        for name in dataset_names
    }

    with h5py.File(h5_path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            for name in dataset_names:
                flat = prepare_values(
                    np.asarray(group[name][...], dtype=np.float64),
                    name,
                    threshold_value,
                    threshold_mode,
                    nonzero_only,
                )
                if flat.size == 0:
                    continue
                entry = summary[name]
                entry["count"] += int(flat.size)
                for threshold in thresholds:
                    label = str(threshold)
                    entry["below_or_equal"][label] += int(np.count_nonzero(flat <= threshold))
                    entry["above_or_equal"][label] += int(np.count_nonzero(flat >= threshold))
                    entry["abs_above_or_equal"][label] += int(np.count_nonzero(np.abs(flat) >= abs(threshold)))

    for entry in summary.values():
        count = max(entry["count"], 1)
        for key in ("below_or_equal", "above_or_equal", "abs_above_or_equal"):
            entry[key] = {label: value / count for label, value in entry[key].items()}
    return summary


def approximate_quantiles(edges: np.ndarray, counts: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    cdf = np.cumsum(counts, dtype=np.float64)
    total = cdf[-1] if cdf.size else 0.0
    if total <= 0:
        return {str(q): float("nan") for q in quantiles}

    results = {}
    for q in quantiles:
        q = float(q)
        target = q * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(max(idx, 0), len(counts) - 1)
        left_edge = edges[idx]
        right_edge = edges[idx + 1]
        prev_cdf = cdf[idx - 1] if idx > 0 else 0.0
        bin_count = counts[idx]
        if bin_count <= 0:
            value = left_edge
        else:
            frac = (target - prev_cdf) / bin_count
            value = left_edge + frac * (right_edge - left_edge)
        results[str(q)] = float(value)
    return results


def save_histogram_plot(output_dir: Path, histograms: dict[str, dict], log_y: bool) -> None:
    fig, axes = plt.subplots(1, len(histograms), figsize=(8 * len(histograms), 5))
    if len(histograms) == 1:
        axes = [axes]

    for axis, (name, info) in zip(axes, histograms.items()):
        centers = 0.5 * (info["edges"][:-1] + info["edges"][1:])
        widths = np.diff(info["edges"])
        axis.bar(centers, info["normalized_frequency"], width=widths, alpha=0.8, align="center")
        axis.set_title(name)
        axis.set_xlabel(f"{name} value")
        axis.set_ylabel("normalized frequency")
        if log_y:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)

    fig.suptitle("QM9 value distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "qm9_value_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_names = resolve_dataset_names(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = build_output_suffix(args)

    stats = first_pass(
        args.h5_path,
        dataset_names,
        threshold_value=args.threshold_value,
        threshold_mode=args.threshold_mode,
        nonzero_only=args.nonzero_only,
    )
    edges_by_dataset = {
        name: np.linspace(stats[name]["min"], stats[name]["max"], args.bins + 1, dtype=np.float64)
        for name in dataset_names
        if stats[name]["count"] > 0
    }
    counts_by_dataset, _unused = second_pass(
        args.h5_path,
        list(edges_by_dataset.keys()),
        edges_by_dataset,
        threshold_value=args.threshold_value,
        threshold_mode=args.threshold_mode,
        nonzero_only=args.nonzero_only,
    )
    threshold_summary = (
        compute_threshold_fractions(
            args.h5_path,
            dataset_names,
            args.thresholds,
            threshold_value=args.threshold_value,
            threshold_mode=args.threshold_mode,
            nonzero_only=args.nonzero_only,
        )
        if args.thresholds
        else {}
    )

    histograms = {}
    report = {}
    for name in dataset_names:
        counts = counts_by_dataset[name]
        total = max(int(counts.sum()), 1)
        normalized_frequency = counts / total
        approx_q = approximate_quantiles(edges_by_dataset[name], counts, list(args.quantiles))
        histograms[name] = {
            "edges": edges_by_dataset[name],
            "counts": counts,
            "normalized_frequency": normalized_frequency,
        }
        report[name] = {
            **stats[name],
            "bins": args.bins,
            "quantiles": approx_q,
            "applied_threshold": {
                "value": args.threshold_value,
                "mode": args.threshold_mode if args.threshold_value is not None else None,
            },
            "nonzero_only": args.nonzero_only,
            "threshold_fractions": threshold_summary.get(name, {}),
        }
        np.savez_compressed(
            output_dir / f"{name}_histogram{output_suffix}.npz",
            edges=edges_by_dataset[name],
            counts=counts,
            normalized_frequency=normalized_frequency,
        )

    plot_path = output_dir / f"qm9_value_distributions{output_suffix}.png"
    save_histogram_plot(output_dir, histograms, log_y=args.log_y)
    default_plot_path = output_dir / "qm9_value_distributions.png"
    if plot_path != default_plot_path and default_plot_path.exists():
        default_plot_path.replace(plot_path)

    (output_dir / f"distribution_summary{output_suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Saved plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
