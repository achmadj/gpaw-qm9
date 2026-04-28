#!/usr/bin/env python3
"""Evaluate one random validation sample and plot true/pred/error slices plus scatter.

Currently supports the dense backend. Sparse backend evaluation can be added later.

Usage:
    python models/evaluate.py \
        --data dataset/gpaw_qm9_merged.h5 \
        --checkpoint models/experiments/dense_baseline_v1/checkpoints/best.pt \
        --output-dir models/experiments/dense_baseline_v1/eval \
        --max-samples 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.config import (
    BASE_CHANNELS,
    DATA_PATH,
    DROPOUT,
    EXPERIMENTS_ROOT,
    INPUT_DATASET,
    INPUT_DATASET_MAP,
    NORM_GROUPS,
    RANDOM_SEED,
    TARGET_DATASET,
    VAL_SPLIT,
)
from models.dense.dataset import QM9DensityDataset, symlog_inv
from models.dense.model import SmallUNet3D
from models.equivariant.model import EquivariantUNet3D
from models.utils import ensure_dir, load_checkpoint, formula_to_electron_count, compute_voxel_volume

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one random QM9 validation sample"
    )

    # ── Shortcut utama ────────────────────────────────────────────
    parser.add_argument(
        "--experiment", default=None,
        help="Nama experiment di bawah EXPERIMENTS_ROOT. "
             "Otomatis set --checkpoint, --output-dir, dan --train-loss-path "
             "jika tidak dispesifikasikan secara eksplisit.",
    )
    parser.add_argument(
        "--backend", choices=["dense", "equivariant"], default="dense",
        help="Model backend: dense (SmallUNet3D) or equivariant (EquivariantUNet3D).",
    )
    parser.add_argument(
        "--max-freq", type=int, default=2,
        help="Maximum SO(3) frequency for equivariant backend (escnn L).",
    )

    # ── Path (semua pakai absolute default dari config) ───────────
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-loss-path", default=None)
    parser.add_argument("--train-loss-plot-outdir", default=None)

    # ── Sisanya sama seperti sebelumnya ───────────────────────────
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument(
        "--auto-base-channels",
        action="store_true",
        default=False,
        help=(
            "Infer base_channels from checkpoint weights "
            "(useful when config default differs from experiment architecture)."
        ),
    )
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--norm-groups", type=int, default=NORM_GROUPS)
    parser.add_argument("--final-activation", choices=["softplus", "relu"], default="relu")
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--target-dataset", default=TARGET_DATASET)
    parser.add_argument(
        "--auto-target-dataset",
        action="store_true",
        default=False,
        help=(
            "If --target-dataset is missing in the HDF5 group, auto-pick a compatible "
            "target dataset (prefers n_pseudo, then n_r, then first non-input dataset)."
        ),
    )
    parser.add_argument("--density-threshold", type=float, default=0.05)
    parser.add_argument("--delta-threshold", type=float, default=None)
    parser.add_argument("--delta-quantile", type=float, default=0.99)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--size", type=float, default=12.0)
    parser.add_argument("--scatter-scale", choices=["auto", "linear", "symlog"], default="auto")
    parser.add_argument("--scatter-linthresh", type=float, default=1e-2)
    try:
        parser.add_argument("--plot-train-loss", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--use-symlog", action=argparse.BooleanOptionalAction, default=False)
    except AttributeError:
        parser.add_argument("--plot-train-loss", action="store_true", default=True)
        parser.add_argument("--no-plot-train-loss", action="store_false", dest="plot_train_loss")
        parser.add_argument("--use-symlog", action="store_true", default=False)
        parser.add_argument("--no-use-symlog", action="store_false", dest="use_symlog")

    args = parser.parse_args()

    # ── Resolve semua path ke absolute ───────────────────────────
    # Tentukan experiment dir
    if args.experiment is not None:
        exp_dir = EXPERIMENTS_ROOT / args.experiment
    else:
        exp_dir = EXPERIMENTS_ROOT / "default"

    # data selalu dari config kecuali di-override
    if args.data is None:
        args.data = str(DATA_PATH)

    # checkpoint, output-dir, train-loss-path: derive dari experiment jika tidak dispesifikasikan
    if args.checkpoint is None:
        args.checkpoint = str(exp_dir / "checkpoints" / "best.pt")
    if args.output_dir is None:
        args.output_dir = str(exp_dir / "eval")
    if args.train_loss_path is None:
        args.train_loss_path = str(exp_dir / "logs" / "history.jsonl")

    # Selalu konversi ke absolute Path string
    args.data             = str(Path(args.data).resolve())
    args.checkpoint       = str(Path(args.checkpoint).resolve())
    args.output_dir       = str(Path(args.output_dir).resolve())
    args.train_loss_path  = str(Path(args.train_loss_path).resolve())

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_keys_and_formulas(h5_path: str, max_samples: int | None = None):
    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle.keys())
        if max_samples is not None:
            keys = keys[:max_samples]
        formulas = []
        for key in keys:
            formula = handle[key].attrs.get("formula", "unknown")
            if isinstance(formula, bytes):
                formula = formula.decode("utf-8")
            formulas.append(str(formula))
    return keys, formulas


def resolve_target_dataset(
    h5_path: str, requested_target: str, input_dataset: str, auto_pick: bool = False
) -> str:
    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle.keys())
        if not keys:
            raise ValueError(f"No groups found in dataset: {h5_path}")
        available = set(handle[keys[0]].keys())

    if requested_target in available:
        return requested_target

    if not auto_pick:
        raise KeyError(
            f"Target dataset '{requested_target}' not found in H5 groups. "
            f"Available datasets: {sorted(available)}. "
            "Pass --target-dataset with a valid name (e.g. n_pseudo) "
            "or use --auto-target-dataset."
        )

    preferred = ("n_pseudo", "n_r")
    for name in preferred:
        if name in available and name != input_dataset:
            print(
                f"[evaluate] target '{requested_target}' not found; "
                f"auto-selecting '{name}'."
            )
            return name

    fallback = next((name for name in sorted(available) if name != input_dataset), None)
    if fallback is None:
        raise KeyError(
            f"No suitable target dataset found in group datasets: {sorted(available)} "
            f"(input dataset is '{input_dataset}')."
        )

    print(
        f"[evaluate] target '{requested_target}' not found; "
        f"auto-selecting fallback dataset '{fallback}'."
    )
    return fallback


def infer_base_channels_from_checkpoint(checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    weight = state_dict.get("enc1.block.0.weight")
    if weight is None:
        raise KeyError(
            "Cannot infer base_channels: key 'enc1.block.0.weight' not found in checkpoint."
        )
    return int(weight.shape[0])


def stratified_split(formulas, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    by_formula = defaultdict(list)
    for index, formula in enumerate(formulas):
        by_formula[formula].append(index)

    train_indices = []
    val_indices = []
    for indices in by_formula.values():
        indices = np.asarray(indices)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_split))) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    return sorted(train_indices), sorted(val_indices)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    target_mean = float(np.mean(target))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((target - target_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pred_sum_raw": float(pred.sum()),
        "true_sum_raw": float(target.sum()),
        "delta_sum_raw": float(pred.sum() - target.sum()),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def plot_slice_triptych(
    target: np.ndarray, pred: np.ndarray, out_path: Path, title: str
) -> None:
    mid_z = target.shape[2] // 2
    true_slice = target[:, :, mid_z].T
    pred_slice = pred[:, :, mid_z].T
    diff_slice = (pred - target)[:, :, mid_z].T

    vmax = float(max(true_slice.max(), pred_slice.max(), 1e-8))
    emax = float(max(np.max(np.abs(diff_slice)), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(
        true_slice, cmap="hot", origin="lower", aspect="auto", vmin=0.0, vmax=vmax
    )
    axes[0].set_title("True n(r)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        pred_slice, cmap="hot", origin="lower", aspect="auto", vmin=0.0, vmax=vmax
    )
    axes[1].set_title("Predicted n(r)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        diff_slice, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-emax, vmax=emax
    )
    axes[2].set_title("Δn(r) = pred - true")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sample_points(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        return flat_idx
    if flat_idx.size > max_points:
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(flat_idx, size=max_points, replace=False)
    return flat_idx


def _scatter_volume(
    ax,
    volume: np.ndarray,
    mask: np.ndarray,
    cmap: str,
    title: str,
    alpha: float,
    size: float,
    vmin: float | None = None,
    vmax: float | None = None,
):
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        ax.set_title(f"{title}\n(no selected voxels)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return

    x, y, z = np.unravel_index(flat_idx, volume.shape)
    values = volume[x, y, z]
    sc = ax.scatter(
        x, y, z, c=values, cmap=cmap, alpha=alpha, s=size, linewidths=0, rasterized=True, vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)


def plot_3d_triptych(
    target: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
    density_threshold: float,
    delta_threshold: float | None,
    delta_quantile: float,
    max_points: int,
    alpha: float,
    size: float,
) -> dict[str, float]:
    diff = pred - target
    true_mask = target >= density_threshold
    pred_mask = pred >= density_threshold

    abs_diff = np.abs(diff)
    if delta_threshold is None:
        nonzero_abs = abs_diff[abs_diff > 0]
        chosen_delta_threshold = (
            float(np.quantile(nonzero_abs, delta_quantile))
            if nonzero_abs.size > 0
            else 0.0
        )
    else:
        chosen_delta_threshold = float(delta_threshold)
    diff_mask = abs_diff >= chosen_delta_threshold

    true_idx = _sample_points(true_mask, max_points=max_points, seed=42)
    pred_idx = _sample_points(pred_mask, max_points=max_points, seed=43)
    diff_idx = _sample_points(diff_mask, max_points=max_points, seed=44)

    true_mask_sampled = np.zeros_like(true_mask, dtype=bool)
    pred_mask_sampled = np.zeros_like(pred_mask, dtype=bool)
    diff_mask_sampled = np.zeros_like(diff_mask, dtype=bool)
    true_mask_sampled.reshape(-1)[true_idx] = True
    pred_mask_sampled.reshape(-1)[pred_idx] = True
    diff_mask_sampled.reshape(-1)[diff_idx] = True

    fig = plt.figure(figsize=(18, 6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    _scatter_volume(ax0, target, true_mask_sampled, "hot", "True n(r)", alpha, size)
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    _scatter_volume(ax1, pred, pred_mask_sampled, "hot", "Predicted n(r)", alpha, size)
    
    # Symmetric colorbar logic for delta
    v_limit = float(max(abs(diff.min()), abs(diff.max())))
    if v_limit == 0:
        v_limit = 1e-8 # Prevent zero range

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    _scatter_volume(
        ax2, diff, diff_mask_sampled, "bwr", "Δn(r) = pred - true", alpha, size, vmin=-v_limit, vmax=v_limit
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "density_threshold": float(density_threshold),
        "delta_threshold": float(chosen_delta_threshold),
        "true_selected_points": int(true_idx.size),
        "pred_selected_points": int(pred_idx.size),
        "delta_selected_points": int(diff_idx.size),
    }


def plot_scatter(
    target: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
    scale: str = "auto",
    linthresh: float = 1e-2,
) -> None:
    true_flat = target.reshape(-1)
    pred_flat = pred.reshape(-1)
    n_pts = min(200000, true_flat.size)
    rng = np.random.default_rng(42)
    idx = (
        rng.choice(true_flat.size, size=n_pts, replace=False)
        if n_pts < true_flat.size
        else np.arange(true_flat.size)
    )

    lim = float(max(true_flat.max(), pred_flat.max(), 1e-8) * 1.02)

    use_symlog = False
    if scale == "symlog":
        use_symlog = True
    elif scale == "auto":
        non_zero = true_flat[true_flat > linthresh]
        if non_zero.size > 0:
            median_val = np.median(non_zero)
            if median_val > 0 and (lim / median_val) > 100:
                use_symlog = True

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(true_flat[idx], pred_flat[idx], s=1.0, alpha=0.25, c="steelblue")
    ax.plot([0.0, lim], [0.0, lim], "r--", lw=1.2)

    if use_symlog:
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xlim(-linthresh / 2, lim)
        ax.set_ylim(-linthresh / 2, lim)
    else:
        ax.set_xlim(0.0, lim)
        ax.set_ylim(0.0, lim)

    ax.set_xlabel("True n(r)")
    ax.set_ylabel("Predicted n(r)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(history_path: Path, output_dir: Path) -> None:
    if not history_path.exists():
        print(f"Warning: History file {history_path} not found. Skipping loss plot.")
        return

    epochs = []
    train_losses = []
    val_losses = []

    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "epoch" in data and "train" in data and "val" in data:
                    epochs.append(data["epoch"])
                    train_losses.append(data["train"].get("loss", 0.0))
                    val_losses.append(data["val"].get("loss", 0.0))
            except json.JSONDecodeError:
                continue

    if not epochs:
        print(f"Warning: No valid data found in {history_path}. Skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.set_title("Training and Validation Loss vs Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    out_path = output_dir / "loss_vs_epoch.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.target_dataset = resolve_target_dataset(
        args.data,
        requested_target=args.target_dataset,
        input_dataset=args.input_dataset,
        auto_pick=args.auto_target_dataset,
    )
    if args.auto_base_channels:
        inferred = infer_base_channels_from_checkpoint(args.checkpoint)
        if inferred != args.base_channels:
            print(
                f"[evaluate] base_channels={args.base_channels} overridden by checkpoint-inferred "
                f"value {inferred}."
            )
            args.base_channels = inferred

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)

    keys, formulas = collect_keys_and_formulas(args.data, args.max_samples)
    dataset = QM9DensityDataset(
        args.data,
        keys=keys,
        input_dataset=args.input_dataset,
        target_dataset=args.target_dataset,
        use_symlog=args.use_symlog,
    )
    _, val_indices = stratified_split(formulas, args.val_split, args.seed)
    if not val_indices:
        raise ValueError("Validation set is empty.")

    if args.sample_index is None:
        sample_rng = np.random.default_rng(args.sample_seed)
        chosen_val_pos = int(sample_rng.integers(len(val_indices)))
    else:
        chosen_val_pos = int(args.sample_index)
        if chosen_val_pos < 0 or chosen_val_pos >= len(val_indices):
            raise IndexError(
                f"sample-index {chosen_val_pos} out of range for {len(val_indices)} validation samples"
            )

    ds_index = val_indices[chosen_val_pos]
    inp, target_tensor, meta = dataset[ds_index]
    group = dataset._require_file()[meta.key]
    cell_angstrom = np.asarray(group.attrs["cell_angstrom"], dtype=np.float64)
    dv = compute_voxel_volume(cell_angstrom, meta.original_shape)
    expected_electrons = formula_to_electron_count(meta.formula)

    # Auto-switch input dataset for equivariant backend
    if args.backend == "equivariant" and args.input_dataset == "v_ext":
        print(
            "[evaluate] Info: equivariant backend detected; switching input-dataset "
            "from v_ext to v_ion for compatibility."
        )
        args.input_dataset = "v_ion"

    # Model instantiation based on backend
    if args.backend == "equivariant":
        model = EquivariantUNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            max_freq=args.max_freq,
            last_activation=args.final_activation,
        ).to(device)
    else:
        model = SmallUNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            dropout=args.dropout,
            norm_groups=args.norm_groups,
            final_activation=args.final_activation,
        ).to(device)
    _, best_val = load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    model.eval()

    def round_up_to_multiple(value: int, multiple: int = 8) -> int:
        if value % multiple == 0:
            return value
        return value + (multiple - value % multiple)

    def pad_spatial(tensor: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        _, x, y, z = tensor.shape
        tx, ty, tz = target_shape
        padded = torch.zeros((tensor.shape[0], tx, ty, tz), dtype=tensor.dtype)
        padded[:, :x, :y, :z] = tensor
        return padded

    with torch.no_grad():
        _, orig_x, orig_y, orig_z = inp.shape
        padded_shape = (
            round_up_to_multiple(orig_x, 8),
            round_up_to_multiple(orig_y, 8),
            round_up_to_multiple(orig_z, 8),
        )
        inp_padded = pad_spatial(inp, padded_shape)

        pred_padded = model(inp_padded.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()

        # Crop back to original size
        pred = pred_padded[:orig_x, :orig_y, :orig_z]

    target = target_tensor.squeeze(0).numpy()

    # Optional: safeguard expm1 by casting to float64 and clipping to avoid overflow
    if args.use_symlog:
        pred = np.clip(pred, -200.0, 200.0)
        pred_physical = symlog_inv(pred.astype(np.float64)).astype(np.float32)
        target_physical = symlog_inv(target.astype(np.float64)).astype(np.float32)
    else:
        pred_physical, target_physical = pred, target

    metrics = compute_metrics(pred_physical, target_physical)
    pred_integral = float(pred_physical.sum() * dv)
    true_integral = float(target_physical.sum() * dv)
    metrics.update(
        {
            "dv_angstrom3": dv,
            "pred_integrated_electrons": pred_integral,
            "true_integrated_electrons": true_integral,
            "delta_integrated_electrons": pred_integral - true_integral,
            "expected_neutral_electrons": int(expected_electrons),
            "delta_pred_vs_expected_electrons": pred_integral - expected_electrons,
            "delta_true_vs_expected_electrons": true_integral - expected_electrons,
        }
    )

    title = (
        f"{meta.formula} | {meta.key} | val_pos={chosen_val_pos} | ds_idx={ds_index}"
    )
    plot_info = plot_3d_triptych(
        target_physical,
        pred_physical,
        output_dir / "nr_true_pred_delta_3d.png",
        title,
        density_threshold=args.density_threshold,
        delta_threshold=args.delta_threshold,
        delta_quantile=args.delta_quantile,
        max_points=args.max_points,
        alpha=args.alpha,
        size=args.size,
    )
    plot_scatter(
        target_physical,
        pred_physical,
        output_dir / "pred_vs_true_scatter.png",
        title,
        scale=args.scatter_scale,
        linthresh=args.scatter_linthresh,
    )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backend": args.backend,
        "best_val": best_val,
        "data": args.data,
        "max_samples": args.max_samples,
        "val_split": args.val_split,
        "split_seed": args.seed,
        "sample_seed": args.sample_seed,
        "selected_validation_position": chosen_val_pos,
        "selected_dataset_index": ds_index,
        "key": meta.key,
        "formula": meta.formula,
        "shape": list(meta.original_shape),
        "plot_info": plot_info,
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(payload, indent=2))
    print(f"Saved 3D plot to {output_dir / 'nr_true_pred_delta_3d.png'}")
    print(f"Saved scatter plot to {output_dir / 'pred_vs_true_scatter.png'}")

    if args.plot_train_loss:
        loss_out_dir = (
            Path(args.train_loss_plot_outdir)
            if args.train_loss_plot_outdir
            else output_dir
        )
        loss_out_dir = ensure_dir(loss_out_dir)
        history_path = Path(args.train_loss_path)
        plot_loss_history(history_path, loss_out_dir)
        print(f"Saved loss plot to {loss_out_dir / 'loss_vs_epoch.png'}")


if __name__ == "__main__":
    main()
