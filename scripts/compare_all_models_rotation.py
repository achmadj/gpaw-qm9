#!/usr/bin/env python3
"""Compare equivariant_v1, equivariant_v2, and dense_phase_a_v1 under 0°/90°/180° rotation.

Tests on validation-only and training-set molecules.
Outputs CSV + 3-column bar chart (0°/90°/180°, 3 bars each: v1/v2/dense).
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, "/home/achmadjae/gpaw-qm9")
from models.equivariant.model import EquivariantUNet3D
from models.dense.model import SmallUNet3D

H5_PATH = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
EQUI_V1_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v1/checkpoints/best.pt"
EQUI_V2_CKPT = "/home/achmadjae/models/experiments/equivariant_v2/checkpoints/best.pt"
DENSE_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/dense_phase_a_v1/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")

VAL_KEYS = [
    ("gdb_1006", "C6H4O", False),
    ("gdb_1011", "C5H4N2", False),
    ("gdb_130830", "C3HN3O3", False),
    ("gdb_8719", "C5HNO2", False),
    ("gdb_16", "C3H6", False),
]
TRAIN_KEYS = [
    ("gdb_110", "C4H4O", True),
    ("gdb_3820", "C3HN3O", True),
    ("gdb_447", "C3H5NO2", True),
    ("gdb_130", "C3H6O2", True),
]
ALL_KEYS = VAL_KEYS + TRAIN_KEYS
PAD_MULTIPLE = 8


def round_up(v, m=PAD_MULTIPLE):
    return v if v % m == 0 else v + (m - v % m)


def pad_spatial(tensor, orig_shape, padded_shape):
    x, y, z = orig_shape
    tx, ty, tz = padded_shape
    padded = torch.zeros((1, 1, tx, ty, tz), dtype=tensor.dtype, device=tensor.device)
    padded[0, 0, :x, :y, :z] = tensor
    return padded


def infer(model, v_ion_raw, device):
    mu = float(v_ion_raw.mean())
    sigma = float(v_ion_raw.std()) + 1e-8
    v_std = (v_ion_raw - mu) / sigma
    v_t = torch.from_numpy(v_std).float()
    x, y, z = v_t.shape
    ps = (round_up(x), round_up(y), round_up(z))
    inp = pad_spatial(v_t, (x, y, z), ps)
    with torch.no_grad():
        out = model(inp.to(device))
    return out[0, 0, :x, :y, :z].cpu().float().numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    equi_v1 = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=32,
                                max_freq=2, last_activation="softplus")
    ckpt = torch.load(EQUI_V1_CKPT, map_location="cpu", weights_only=False)
    equi_v1.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    equi_v1.to(device).eval()
    n1 = sum(p.numel() for p in equi_v1.parameters())
    print(f"equivariant_v1: {n1:,} params", flush=True)

    equi_v2 = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=96,
                                max_freq=2, last_activation="softplus")
    ckpt = torch.load(EQUI_V2_CKPT, map_location="cpu", weights_only=False)
    equi_v2.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    equi_v2.to(device).eval()
    n2 = sum(p.numel() for p in equi_v2.parameters())
    print(f"equivariant_v2: {n2:,} params", flush=True)

    dense = SmallUNet3D(in_channels=1, out_channels=1, base_channels=32,
                        final_activation="softplus")
    ckpt = torch.load(DENSE_CKPT, map_location="cpu", weights_only=False)
    dense.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    dense.to(device).eval()
    n3 = sum(p.numel() for p in dense.parameters())
    print(f"dense_phase_a_v1: {n3:,} params", flush=True)

    rotations = [
        ("0° (original)", 0, (0, 2)),
        ("90° y-axis", 90, (0, 2)),
        ("180° y-axis", 180, (0, 2)),
    ]

    rows = []
    with h5py.File(H5_PATH, "r") as f:
        for key, formula, in_train in ALL_KEYS:
            grp = f[key]
            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)
            sh = v_ion_raw.shape
            split = "TRAIN" if in_train else "VAL"
            print(f"\n{key} ({formula}) shape={sh} [{split}]", flush=True)

            for rot_label, angle, axes in rotations:
                if angle == 0:
                    vi = v_ion_raw
                    nr = n_r_raw
                else:
                    vi = rotate(v_ion_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)
                    nr = rotate(n_r_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)

                pred_v1 = infer(equi_v1, vi, device)
                r2_v1 = float(r2_score(nr.ravel(), pred_v1.ravel()))
                mae_v1 = float(mean_absolute_error(nr.ravel(), pred_v1.ravel()))

                pred_v2 = infer(equi_v2, vi, device)
                r2_v2 = float(r2_score(nr.ravel(), pred_v2.ravel()))
                mae_v2 = float(mean_absolute_error(nr.ravel(), pred_v2.ravel()))

                pred_d = infer(dense, vi, device)
                r2_d = float(r2_score(nr.ravel(), pred_d.ravel()))
                mae_d = float(mean_absolute_error(nr.ravel(), pred_d.ravel()))

                print(f"  {rot_label:18s}  v1 R²={r2_v1:.4f}  v2 R²={r2_v2:.4f}  dense R²={r2_d:.4f}", flush=True)

                rows.append({
                    "key": key, "formula": formula, "shape": f"{sh[0]}×{sh[1]}×{sh[2]}",
                    "in_train": in_train, "split": split,
                    "rotation": rot_label, "angle": angle,
                    "equi_v1_r2": r2_v1, "equi_v1_mae": mae_v1,
                    "equi_v2_r2": r2_v2, "equi_v2_mae": mae_v2,
                    "dense_r2": r2_d, "dense_mae": mae_d,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "all_models_rotation_comparison.csv"
    df.to_csv(csv_path, index=False)

    colors = {"v1": "#2166ac", "v2": "#4dac26", "dense": "#b2182b"}
    labels = {"v1": "Equi v1 (542K)", "v2": "Equi v2 (4.87M)", "dense": "Dense (5.70M)"}

    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    for i, (rot_label, angle, _) in enumerate(rotations):
        ax = axes[i]
        sub = df[df["rotation"] == rot_label].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.25

        bars_v1 = ax.bar(x - w, sub["equi_v1_r2"].values, w, label=labels["v1"],
                         color=colors["v1"], alpha=0.85)
        bars_v2 = ax.bar(x, sub["equi_v2_r2"].values, w, label=labels["v2"],
                         color=colors["v2"], alpha=0.85)
        bars_d = ax.bar(x + w, sub["dense_r2"].values, w, label=labels["dense"],
                        color=colors["dense"], alpha=0.85)

        for j in range(len(sub)):
            ax.text(x[j] - w, sub["equi_v1_r2"].values[j] + 0.008,
                    f"{sub['equi_v1_r2'].values[j]:.3f}", ha="center", va="bottom", fontsize=5.5)
            ax.text(x[j], sub["equi_v2_r2"].values[j] + 0.008,
                    f"{sub['equi_v2_r2'].values[j]:.3f}", ha="center", va="bottom", fontsize=5.5)
            ax.text(x[j] + w, sub["dense_r2"].values[j] + 0.008,
                    f"{sub['dense_r2'].values[j]:.3f}", ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['formula']}\n{r['shape']}\n[{r['split']}]"
                            for _, r in sub.iterrows()], fontsize=6.5)
        ax.set_ylim(0.5, 1.01)
        ax.set_ylabel("R²")
        ax.set_title(rot_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        if i == 0:
            ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Equivariant v1 / v2 vs Dense: R² under Grid Rotations\n"
                 "Validation (unseen) + Training molecules, order=0 no interpolation",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    svg_path = OUT_DIR / "all_models_rotation_comparison.svg"
    png_path = OUT_DIR / "all_models_rotation_comparison.png"
    fig.savefig(svg_path, dpi=150, bbox_inches="tight", format="svg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {svg_path}", flush=True)
    print(f"Saved: {png_path}", flush=True)
    print(f"Saved: {csv_path}", flush=True)

    print("\n=== R² Summary ===")
    for rot_label, _, _ in rotations:
        sub = df[df["rotation"] == rot_label]
        print(f"\n{rot_label}:")
        print(f"  Equi v1: avg R²={sub['equi_v1_r2'].mean():.4f}, avg MAE={sub['equi_v1_mae'].mean():.5f}")
        print(f"  Equi v2: avg R²={sub['equi_v2_r2'].mean():.4f}, avg MAE={sub['equi_v2_mae'].mean():.5f}")
        print(f"  Dense:   avg R²={sub['dense_r2'].mean():.4f}, avg MAE={sub['dense_mae'].mean():.5f}")

    print("\n=== 90° R² DROP ===")
    sub_90 = df[df["rotation"] == "90° y-axis"]
    for _, r in sub_90.iterrows():
        best = max(r["equi_v1_r2"], r["equi_v2_r2"], r["dense_r2"])
        delta_v1 = r["dense_r2"] - r["equi_v1_r2"]
        delta_v2 = r["dense_r2"] - r["equi_v2_r2"]
        print(f"  {r['key']} ({r['formula']}): v1={r['equi_v1_r2']:.4f} v2={r['equi_v2_r2']:.4f} "
              f"dense={r['dense_r2']:.4f}  best={best:.4f}")


if __name__ == "__main__":
    main()
