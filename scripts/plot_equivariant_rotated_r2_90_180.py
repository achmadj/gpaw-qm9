#!/usr/bin/env python3
"""Plot R² scatter for equivariant_v1 model with 90°/180° axis rotations on 10 molecules.

For each molecule:
  1. Load v_ion (input) and n_r (target) from Phase A dataset
  2. Apply rotation: random axis (x/y/z) + random choice of 90° or 180°
     Using order=0 (nearest-neighbor) — no interpolation, pure voxel permute/flip
  3. Apply per-sample standardization to v_ion (matching training)
  4. Pad to multiple of 8, run inference on GPU
  5. Compute R² and MAE between predicted n_r and rotated ground-truth n_r

Output: SVG figure with 10 rasterized scatter subplots.
"""

import sys
import os
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
CKPT_PATH = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v1/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
N_MOLS = 10
SEED = 42
PAD_MULTIPLE = 8
ROTATION_ANGLES = [90, 180]


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
        in_channels=1,
        out_channels=1,
        base_channels=32,
        max_freq=2,
        last_activation="softplus",
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
        chosen = sorted(rng.choice(keys, N_MOLS, replace=False))
        print(f"Chosen molecules: {chosen}", flush=True)

        results = []
        for i, key in enumerate(chosen):
            grp = f[key]
            formula = grp.attrs.get("formula", "?")

            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)

            axis = int(rng.integers(0, 3))
            angle = int(rng.choice(ROTATION_ANGLES))

            if axis == 0:
                rot_axes = (1, 2)
            elif axis == 1:
                rot_axes = (0, 2)
            else:
                rot_axes = (0, 1)

            print(f"[{i+1}/{N_MOLS}] {key} ({formula}) | axis={axis_names[axis]}, angle={angle}°", flush=True)

            # order=0 = nearest-neighbour, no interpolation — perfect for 90°/180°
            v_ion_rot = rotate(v_ion_raw, angle, axes=rot_axes, reshape=False, order=0,
                               prefilter=False, mode="constant", cval=0.0)
            n_r_rot = rotate(n_r_raw, angle, axes=rot_axes, reshape=False, order=0,
                             prefilter=False, mode="constant", cval=0.0)

            # Per-sample standardization (matching PhaseADataset)
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

            print(f"  R²={r2:.6f}, MAE={mae:.6f}", flush=True)

            results.append({
                "key": key,
                "formula": formula,
                "axis": axis_names[axis],
                "angle": angle,
                "r2": r2,
                "mae": mae,
                "pred": pred,
                "target": n_r_rot,
                "shape": orig_shape,
            })

    # --- Plot: SVG with rasterized scatter ---
    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    axes = axes.flatten()

    for i, (ax, res) in enumerate(zip(axes, results)):
        true = res["target"].ravel()
        pred = res["pred"].ravel()

        ax.scatter(true, pred, s=2, alpha=0.25, c="#2166ac", rasterized=True)

        vmin = min(float(true.min()), float(pred.min()))
        vmax = max(float(true.max()), float(pred.max()))
        pad_val = (vmax - vmin) * 0.05
        ax.plot([vmin - pad_val, vmax + pad_val],
                [vmin - pad_val, vmax + pad_val],
                "k--", lw=1.0, alpha=0.6)

        ax.set_xlabel("True n_r", fontsize=9)
        ax.set_ylabel("Pred n_r", fontsize=9)
        ax.set_title(
            f"{res['formula']}  |  {res['axis']}, {res['angle']}°\n"
            f"R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}",
            fontsize=9,
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Equivariant 3D U-Net (equivariant_v1): R² under 90°/180° Axis Rotation (Nearest-Neighbor, No Interpolation)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    svg_path = OUT_DIR / "equivariant_rotated_r2_90_180.svg"
    fig.savefig(svg_path, dpi=150, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"\nSaved: {svg_path}", flush=True)

    # --- Also save PNG for quick preview ---
    fig2, axes2 = plt.subplots(2, 5, figsize=(28, 11))
    axes2 = axes2.flatten()
    for i, (ax, res) in enumerate(zip(axes2, results)):
        true = res["target"].ravel()
        pred = res["pred"].ravel()
        ax.scatter(true, pred, s=2, alpha=0.25, c="#2166ac", rasterized=True)
        vmin = min(float(true.min()), float(pred.min()))
        vmax = max(float(true.max()), float(pred.max()))
        pad_val = (vmax - vmin) * 0.05
        ax.plot([vmin - pad_val, vmax + pad_val],
                [vmin - pad_val, vmax + pad_val], "k--", lw=1.0, alpha=0.6)
        ax.set_xlabel("True n_r", fontsize=9)
        ax.set_ylabel("Pred n_r", fontsize=9)
        ax.set_title(
            f"{res['formula']}  |  {res['axis']}, {res['angle']}°\n"
            f"R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}",
            fontsize=9,
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=7)
    fig2.suptitle(
        "Equivariant 3D U-Net (equivariant_v1): R² under 90°/180° Axis Rotation",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig2.tight_layout()
    png_path = OUT_DIR / "equivariant_rotated_r2_90_180.png"
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {png_path}", flush=True)

    # --- Summary ---
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['key']}: {r['formula']} rot={r['axis']}({r['angle']}°) R²={r['r2']:.6f} MAE={r['mae']:.6f}")


if __name__ == "__main__":
    main()
