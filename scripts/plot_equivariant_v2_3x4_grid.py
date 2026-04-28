#!/usr/bin/env python3
"""3x4 grid for equivariant_v2: V_ion / n_pred / n_true / delta_nr under 0°/90°/180°.

Molecule: gdb_110 (C4H4O) from QM9 Phase A, equivariant_v2 model.
Output: SVG + PNG to figures/ dir.
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

sys.path.insert(0, "/home/achmadjae/gpaw-qm9")
from models.equivariant.model import EquivariantUNet3D


H5_PATH = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
CKPT_PATH = "/home/achmadjae/models/experiments/equivariant_v2/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
KEY = "gdb_110"
ROT_AXIS = 1
ROT_AXIS_NAME = "y (xz-plane)"
PAD_MULTIPLE = 8
SEED = 42
DENSITY_THRESHOLD = 0.01
MAX_POINTS = 8000
ALPHA = 0.55
SIZE = 2.5
DPI = 120


def round_up(v: int, m: int = PAD_MULTIPLE) -> int:
    return v if v % m == 0 else v + (m - v % m)


def pad_spatial(tensor: torch.Tensor, orig_shape: tuple, padded_shape: tuple) -> torch.Tensor:
    x, y, z = orig_shape
    tx, ty, tz = padded_shape
    padded = torch.zeros((1, 1, tx, ty, tz), dtype=tensor.dtype, device=tensor.device)
    padded[0, 0, :x, :y, :z] = tensor
    return padded


def sample_points(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        return flat_idx
    if flat_idx.size > max_points:
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(flat_idx, size=max_points, replace=False)
    return flat_idx


def scatter_volume(ax, volume: np.ndarray, mask: np.ndarray, cmap: str,
                   title: str, alpha: float, size: float,
                   vmin: float | None = None, vmax: float | None = None):
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        ax.set_title(f"{title}\n(no voxels)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        return
    x, y, z = np.unravel_index(flat_idx, volume.shape)
    values = volume[x, y, z]
    sc = ax.scatter(x, y, z, c=values, cmap=cmap, alpha=alpha, s=size,
                     linewidths=0, rasterized=True, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=5)
    plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.04)


def infer(model: EquivariantUNet3D, v_ion_raw: np.ndarray, device: torch.device) -> np.ndarray:
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

    model = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=96,
                               max_freq=2, last_activation="softplus")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    with h5py.File(H5_PATH, "r") as f:
        grp = f[KEY]
        formula = grp.attrs["formula"]
        v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
        n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)
        print(f"Molecule: {KEY} ({formula}), v_ion shape={v_ion_raw.shape}", flush=True)

    pred0 = infer(model, v_ion_raw, device)

    rot_axes = (0, 2)
    v_ion_90 = rotate(v_ion_raw, 90, axes=rot_axes, reshape=False, order=0,
                      prefilter=False, mode="constant", cval=0.0)
    n_r_90 = rotate(n_r_raw, 90, axes=rot_axes, reshape=False, order=0,
                    prefilter=False, mode="constant", cval=0.0)
    pred90 = infer(model, v_ion_90, device)

    v_ion_180 = rotate(v_ion_raw, 180, axes=rot_axes, reshape=False, order=0,
                       prefilter=False, mode="constant", cval=0.0)
    n_r_180 = rotate(n_r_raw, 180, axes=rot_axes, reshape=False, order=0,
                     prefilter=False, mode="constant", cval=0.0)
    pred180 = infer(model, v_ion_180, device)

    scenarios = [
        ("Original (no rotation)", v_ion_raw, n_r_raw, pred0),
        (f"Rotated 90° ({ROT_AXIS_NAME})", v_ion_90, n_r_90, pred90),
        (f"Rotated 180° ({ROT_AXIS_NAME})", v_ion_180, n_r_180, pred180),
    ]

    fig = plt.figure(figsize=(22, 16))

    for row_idx, (label, vi, nr, pr) in enumerate(scenarios):
        diff = pr - nr
        nr_mask = nr >= DENSITY_THRESHOLD
        pr_mask = pr >= DENSITY_THRESHOLD

        abs_diff = np.abs(diff)
        nonzero_abs = abs_diff[abs_diff > 0]
        if nonzero_abs.size > 0:
            dthresh = float(np.quantile(nonzero_abs, 0.75))
        else:
            dthresh = 0.0
        diff_mask = abs_diff >= dthresh

        for col_idx, (data, mask, cmap, ctitle, vmin, vmax) in enumerate([
            (vi, np.abs(vi) >= np.quantile(np.abs(vi[vi != 0]) if (vi != 0).any() else [0], 0.3),
             "viridis", "V_ion (input)", None, None),
            (pr, pr_mask, "hot", "n_pred", 0, None),
            (nr, nr_mask, "hot", "n_true", 0, None),
            (diff, diff_mask, "bwr", "Δn = pred - true",
             -float(max(abs(diff.min()), abs(diff.max()))),
             float(max(abs(diff.min()), abs(diff.max())))),
        ]):
            ax = fig.add_subplot(3, 4, row_idx * 4 + col_idx + 1, projection="3d")

            f_idx = sample_points(mask, max_points=MAX_POINTS, seed=SEED + row_idx * 10 + col_idx)
            if f_idx.size > 0:
                sampled_mask = np.zeros_like(mask, dtype=bool)
                sampled_mask.reshape(-1)[f_idx] = True
            else:
                sampled_mask = mask

            scatter_volume(ax, data, sampled_mask, cmap, ctitle, ALPHA, SIZE, vmin, vmax)

            if col_idx == 0:
                ax.set_title(f"[{row_idx+1}] {label}\n\n{ctitle}", fontsize=9,
                             fontweight="bold", loc="left")

    fig.suptitle(
        f"Equivariant 3D U-Net (equivariant_v2, 4.87M): Density Prediction under Grid Rotations\n"
        f"Molecule: {formula} ({KEY}), grid {v_ion_raw.shape[0]}×{v_ion_raw.shape[1]}×{v_ion_raw.shape[2]}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    svg_path = OUT_DIR / "equivariant_v2_3x4_scatter.svg"
    fig.savefig(svg_path, dpi=DPI, bbox_inches="tight", format="svg")
    print(f"\nSaved SVG: {svg_path} ({svg_path.stat().st_size / 1024:.0f} KB)", flush=True)

    png_path = OUT_DIR / "equivariant_v2_3x4_scatter.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved PNG: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)", flush=True)

    plt.close(fig)

    from sklearn.metrics import r2_score, mean_absolute_error
    print("\n=== R² / MAE Summary ===")
    for label, _, nr, pr in scenarios:
        r2 = r2_score(nr.ravel(), pr.ravel())
        mae = mean_absolute_error(nr.ravel(), pr.ravel())
        print(f"  {label}: R²={r2:.6f}, MAE={mae:.6f}")

    v_ion_90_r2 = r2_score(n_r_90.ravel(), pred90.ravel())
    v_ion_180_r2 = r2_score(n_r_180.ravel(), pred180.ravel())
    print(f"\n 90° R²={v_ion_90_r2:.6f}, 180° R²={v_ion_180_r2:.6f}")


if __name__ == "__main__":
    main()
