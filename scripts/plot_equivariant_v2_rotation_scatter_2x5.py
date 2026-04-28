#!/usr/bin/env python3
"""Generate 2x5 scatter grid for equivariant_v2 under rotation (same molecules as v1 plot).

This reproduces the exact same 10 molecules and rotation assignments
as the v1 scatter grid, but with the v2 model (4.87M params) and
corrected reshape=True rotation (no zero-padding artifact).
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

H5_PATH = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
CKPT_PATH = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
SEED = 42
PAD_MULTIPLE = 8


def round_up(v: int, m: int = PAD_MULTIPLE) -> int:
    return v if v % m == 0 else v + (m - v % m)


def pad_spatial(tensor: torch.Tensor, orig_shape: tuple, padded_shape: tuple) -> torch.Tensor:
    x, y, z = orig_shape
    tx, ty, tz = padded_shape
    padded = torch.zeros((1, 1, tx, ty, tz), dtype=tensor.dtype, device=tensor.device)
    padded[0, 0, :x, :y, :z] = tensor
    return padded


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = EquivariantUNet3D(
        in_channels=1, out_channels=1, base_channels=96,
        max_freq=2, last_activation="softplus",
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    axis_names = ["x (yz-plane)", "y (xz-plane)", "z (xy-plane)"]

    with h5py.File(H5_PATH, "r") as f:
        keys = sorted(f.keys())
        chosen = sorted(rng.choice(keys, 10, replace=False))
        print(f"Chosen molecules: {chosen}", flush=True)

        # Pre-determine rotations (same as v1 script)
        rotations = []
        for i in range(10):
            axis = int(rng.integers(0, 3))
            angle = int(rng.choice([90, 180]))
            rotations.append((axis, angle))

        # Collect data for plots
        plot_data = []
        for i, key in enumerate(chosen):
            grp = f[key]
            formula = grp.attrs.get("formula", "?")
            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)

            axis, angle = rotations[i]
            if axis == 0:
                rot_axes = (1, 2)
            elif axis == 1:
                rot_axes = (0, 2)
            else:
                rot_axes = (0, 1)

            v_ion_rot = rotate(v_ion_raw, angle, axes=rot_axes, reshape=True, order=0,
                               prefilter=False, mode="constant", cval=0.0)
            n_r_rot = rotate(n_r_raw, angle, axes=rot_axes, reshape=True, order=0,
                             prefilter=False, mode="constant", cval=0.0)

            mu = float(v_ion_rot.mean())
            sigma = float(v_ion_rot.std()) + 1e-8
            v_ion_std = (v_ion_rot - mu) / sigma

            v_ion_t = torch.from_numpy(v_ion_std).float()
            x, y, z = v_ion_t.shape
            orig_shape = (x, y, z)
            padded_shape = (round_up(x), round_up(y), round_up(z))
            inp_padded = pad_spatial(v_ion_t, orig_shape, padded_shape)

            with torch.no_grad():
                pred_padded = model(inp_padded.to(device))

            pred = pred_padded[0, 0, :x, :y, :z].cpu().float().numpy()

            r2 = float(r2_score(n_r_rot.ravel(), pred.ravel()))
            mae = float(mean_absolute_error(n_r_rot.ravel(), pred.ravel()))

            print(f"[{i+1}/10] {key} ({formula}) | axis={axis_names[axis]}, angle={angle}° | R²={r2:.4f}, MAE={mae:.4f}", flush=True)

            plot_data.append({
                "formula": formula,
                "axis": axis_names[axis],
                "angle": angle,
                "r2": r2,
                "mae": mae,
                "true": n_r_rot.ravel(),
                "pred": pred.ravel(),
            })

    # Create 2x5 scatter grid
    fig, axes = plt.subplots(2, 5, figsize=(28, 10))
    fig.suptitle("Equivariant 3D U-Net (equivariant_v2): R² under 90°/180° Axis Rotation",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, ax in enumerate(axes.flat):
        d = plot_data[idx]
        ax.scatter(d["true"], d["pred"], s=1.5, alpha=0.25, edgecolors="none", color="#1f77b4")

        # Diagonal line
        lo = min(d["true"].min(), d["pred"].min())
        hi = max(d["true"].max(), d["pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)

        ax.set_title(f"{d['formula']}  |  {d['axis']}, {d['angle']}°\n"
                     f"R² = {d['r2']:.4f}, MAE = {d['mae']:.4f}",
                     fontsize=10)
        ax.set_xlabel("True n_r", fontsize=9)
        ax.set_ylabel("Pred n_r", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = OUT_DIR / "equivariant_v2_rotation_scatter_2x5.png"
    out_pdf = OUT_DIR / "equivariant_v2_rotation_scatter_2x5.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
