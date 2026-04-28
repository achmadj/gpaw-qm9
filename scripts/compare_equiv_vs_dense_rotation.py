#!/usr/bin/env python3
"""Compare equivariant vs dense model under 0°/90°/180° rotation on multiple molecules.

Tests both models on: gdb_110 (C4H4O, near-isotropic), gdb_3820 (C3HN3O, anisotropic),
gdb_447 (C3H5NO2, moderate), gdb_130 (C3H6O2, moderate).

Outputs:
  - CSV with per-molecule/per-model R²
  - Bar chart: R² comparison for each rotation (equivariant vs dense)
  - Scatter grid: 4 molecules × 3 rotations × 2 models (optional, heavy)
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
EQUI_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v1/checkpoints/best.pt"
DENSE_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/dense_phase_a_v1/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
TEST_KEYS = ["gdb_110", "gdb_3820", "gdb_447", "gdb_130"]
PAD_MULTIPLE = 8
SEED = 42


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

    # Load equivariant model
    equi_model = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=32,
                                   max_freq=2, last_activation="softplus")
    ckpt_e = torch.load(EQUI_CKPT, map_location="cpu", weights_only=False)
    equi_model.load_state_dict(ckpt_e["model_state_dict"] if "model_state_dict" in ckpt_e else ckpt_e)
    equi_model.to(device).eval()
    print(f"Equivariant model: {sum(p.numel() for p in equi_model.parameters()):,} params", flush=True)

    # Load dense model
    dense_model = SmallUNet3D(in_channels=1, out_channels=1, base_channels=32,
                              final_activation="softplus")
    ckpt_d = torch.load(DENSE_CKPT, map_location="cpu", weights_only=False)
    dense_model.load_state_dict(ckpt_d["model_state_dict"] if "model_state_dict" in ckpt_d else ckpt_d)
    dense_model.to(device).eval()
    print(f"Dense model: {sum(p.numel() for p in dense_model.parameters()):,} params", flush=True)

    rotations = [
        ("0° (original)", 0, (0, 2)),
        ("90° y-axis", 90, (0, 2)),
        ("180° y-axis", 180, (0, 2)),
    ]

    rows = []
    with h5py.File(H5_PATH, "r") as f:
        for key in TEST_KEYS:
            grp = f[key]
            formula = grp.attrs["formula"]
            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)
            sh = v_ion_raw.shape
            print(f"\n{key} ({formula}) shape={sh}", flush=True)

            for rot_label, angle, axes in rotations:
                if angle == 0:
                    vi = v_ion_raw
                    nr = n_r_raw
                else:
                    vi = rotate(v_ion_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)
                    nr = rotate(n_r_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)

                # Equivariant
                pred_e = infer(equi_model, vi, device)
                r2_e = float(r2_score(nr.ravel(), pred_e.ravel()))
                mae_e = float(mean_absolute_error(nr.ravel(), pred_e.ravel()))

                # Dense
                pred_d = infer(dense_model, vi, device)
                r2_d = float(r2_score(nr.ravel(), pred_d.ravel()))
                mae_d = float(mean_absolute_error(nr.ravel(), pred_d.ravel()))

                print(f"  {rot_label:18s}  equi R²={r2_e:.4f} MAE={mae_e:.5f}  |  dense R²={r2_d:.4f} MAE={mae_d:.5f}", flush=True)

                rows.append({
                    "key": key, "formula": formula, "shape": f"{sh[0]}×{sh[1]}×{sh[2]}",
                    "rotation": rot_label, "angle": angle,
                    "equi_r2": r2_e, "equi_mae": mae_e,
                    "dense_r2": r2_d, "dense_mae": mae_d,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "equiv_vs_dense_rotation_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}", flush=True)

    # --- Bar chart comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (rot_label, angle, _) in enumerate(rotations):
        ax = axes[i]
        sub = df[df["rotation"] == rot_label]
        x = np.arange(len(sub))
        w = 0.35
        bars_e = ax.bar(x - w/2, sub["equi_r2"].values, w, label="Equivariant (escnn)",
                         color="#2166ac", alpha=0.85)
        bars_d = ax.bar(x + w/2, sub["dense_r2"].values, w, label="Dense (standard)",
                         color="#b2182b", alpha=0.85)

        for bar, v in zip(bars_e, sub["equi_r2"].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars_d, sub["dense_r2"].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['formula']}\n({r['shape']})" for _, r in sub.iterrows()], fontsize=7)
        ax.set_ylim(0.6, 1.01)
        ax.set_ylabel("R²")
        ax.set_title(rot_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Equivariant vs Dense 3D U-Net: R² under Grid Rotations (order=0, no interpolation)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    bar_path = OUT_DIR / "equiv_vs_dense_rotation_bar.svg"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", format="svg")
    png_path = OUT_DIR / "equiv_vs_dense_rotation_bar.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart: {bar_path}", flush=True)
    print(f"Saved bar chart: {png_path}", flush=True)

    # --- Summary ---
    print("\n=== Summary ===")
    for rot_label, _, _ in rotations:
        sub = df[df["rotation"] == rot_label]
        print(f"\n{rot_label}:")
        print(f"  Equivariant: avg R²={sub['equi_r2'].mean():.4f}, avg MAE={sub['equi_mae'].mean():.5f}")
        print(f"  Dense:       avg R²={sub['dense_r2'].mean():.4f}, avg MAE={sub['dense_mae'].mean():.5f}")


if __name__ == "__main__":
    main()
