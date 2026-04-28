#!/usr/bin/env python3
"""Re-run Equivariant vs Dense rotation comparison on VALIDATION-ONLY molecules.

Key difference from compare_equiv_vs_dense_rotation.py:
- Uses ONLY molecules from the validation split (unseen during training)
- Picks anisotropic molecules (aspect ratio > 2x) for worst-case 90° test
- Also includes an isotropic control (cubic grid)
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
# Validation-only molecules (from seed=42, val_split=0.1, NOT in training set)
VAL_KEYS = [
    ("gdb_1006", "C6H4O", "40×56×20", 2.8),   # highly anisotropic
    ("gdb_1011", "C5H4N2", "44×48×20", 2.4),  # anisotropic
    ("gdb_130830", "C3HN3O3", "36×48×20", 2.4), # anisotropic
    ("gdb_8719", "C5HNO2", "56×36×20", 2.8),   # highly anisotropic
    ("gdb_16", "C3H6", "32×32×32", 1.0),       # cubic control
]
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

    equi_model = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=32,
                                   max_freq=2, last_activation="softplus")
    ckpt_e = torch.load(EQUI_CKPT, map_location="cpu", weights_only=False)
    equi_model.load_state_dict(ckpt_e["model_state_dict"] if "model_state_dict" in ckpt_e else ckpt_e)
    equi_model.to(device).eval()

    dense_model = SmallUNet3D(in_channels=1, out_channels=1, base_channels=32,
                              final_activation="softplus")
    ckpt_d = torch.load(DENSE_CKPT, map_location="cpu", weights_only=False)
    dense_model.load_state_dict(ckpt_d["model_state_dict"] if "model_state_dict" in ckpt_d else ckpt_d)
    dense_model.to(device).eval()
    print(f"Models loaded.", flush=True)

    rotations = [
        ("0° (original)", 0, (0, 2)),
        ("90° y-axis", 90, (0, 2)),
        ("180° y-axis", 180, (0, 2)),
    ]

    rows = []
    with h5py.File(H5_PATH, "r") as f:
        for key, formula, shape_str, aspect in VAL_KEYS:
            grp = f[key]
            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)
            sh = v_ion_raw.shape
            print(f"\n{key} ({formula}) shape={sh} aspect={aspect:.1f}x [VALIDATION]", flush=True)

            for rot_label, angle, axes in rotations:
                if angle == 0:
                    vi = v_ion_raw
                    nr = n_r_raw
                else:
                    vi = rotate(v_ion_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)
                    nr = rotate(n_r_raw, angle, axes=axes, reshape=False, order=0,
                                prefilter=False, mode="constant", cval=0.0)

                pred_e = infer(equi_model, vi, device)
                r2_e = float(r2_score(nr.ravel(), pred_e.ravel()))
                mae_e = float(mean_absolute_error(nr.ravel(), pred_e.ravel()))

                pred_d = infer(dense_model, vi, device)
                r2_d = float(r2_score(nr.ravel(), pred_d.ravel()))
                mae_d = float(mean_absolute_error(nr.ravel(), pred_d.ravel()))

                winner = "EQUI" if r2_e > r2_d else "DENSE"
                print(f"  {rot_label:18s}  equi R²={r2_e:.4f}  dense R²={r2_d:.4f}  {winner}", flush=True)

                rows.append({
                    "key": key, "formula": formula, "shape": f"{sh[0]}×{sh[1]}×{sh[2]}",
                    "aspect": aspect, "in_train": False,
                    "rotation": rot_label, "angle": angle,
                    "equi_r2": r2_e, "equi_mae": mae_e,
                    "dense_r2": r2_d, "dense_mae": mae_d,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "equiv_vs_dense_rotation_val_only.csv"
    df.to_csv(csv_path, index=False)

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for i, (rot_label, angle, _) in enumerate(rotations):
        ax = axes[i]
        sub = df[df["rotation"] == rot_label]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub["equi_r2"].values, w, label="Equivariant (escnn, 542K)",
               color="#2166ac", alpha=0.85)
        ax.bar(x + w/2, sub["dense_r2"].values, w, label="Dense (standard, 5.7M)",
               color="#b2182b", alpha=0.85)
        # Add R² values on bars
        for j in range(len(sub)):
            ax.text(x[j] - w/2, sub["equi_r2"].values[j] + 0.008,
                    f"{sub['equi_r2'].values[j]:.3f}", ha="center", va="bottom", fontsize=6)
            ax.text(x[j] + w/2, sub["dense_r2"].values[j] + 0.008,
                    f"{sub['dense_r2'].values[j]:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['formula']}\n({r['shape']})\nasp={r['aspect']:.1f}x"
                            for _, r in sub.iterrows()], fontsize=7)
        ax.set_ylim(0.5, 1.01)
        ax.set_ylabel("R²")
        ax.set_title(f"{rot_label}  (VALIDATION ONLY — unseen molecules)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Equivariant vs Dense 3D U-Net: R² under Grid Rotations\n"
                 "VALIDATION-ONLY molecules (never seen during training, order=0 no interpolation)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()

    fig.savefig(OUT_DIR / "equiv_vs_dense_rotation_VAL_ONLY.svg", dpi=150, bbox_inches="tight", format="svg")
    fig.savefig(OUT_DIR / "equiv_vs_dense_rotation_VAL_ONLY.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUT_DIR / 'equiv_vs_dense_rotation_VAL_ONLY.svg'}", flush=True)
    print(f"Saved: {OUT_DIR / 'equiv_vs_dense_rotation_VAL_ONLY.png'}", flush=True)
    print(f"Saved: {csv_path}", flush=True)

    # Summary
    print("\n=== VALIDATION-ONLY Summary ===")
    for rot_label, _, _ in rotations:
        sub = df[df["rotation"] == rot_label]
        print(f"\n{rot_label}:")
        print(f"  Equivariant: avg R²={sub['equi_r2'].mean():.4f}, avg MAE={sub['equi_mae'].mean():.5f}")
        print(f"  Dense:       avg R²={sub['dense_r2'].mean():.4f}, avg MAE={sub['dense_mae'].mean():.5f}")

    # Per-molecule 90° delta
    print("\n=== 90° R² DROP (critical test) ===")
    sub_90 = df[df["rotation"] == "90° y-axis"]
    for _, r in sub_90.iterrows():
        delta = r["dense_r2"] - r["equi_r2"]
        print(f"  {r['key']} ({r['formula']}, asp={r['aspect']:.1f}x): "
              f"equi={r['equi_r2']:.4f} dense={r['dense_r2']:.4f}  Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
