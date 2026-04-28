#!/usr/bin/env python3
"""Run equivariant_v2 rotation R2 on 10 random molecules."""
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

            print(f"  R²={r2:.6f}, MAE={mae:.6f}", flush=True)
            results.append({"key": key, "formula": formula, "axis": axis_names[axis],
                            "angle": angle, "r2": r2, "mae": mae})

    # Summary
    print("\n=== Summary ===")
    by_angle = {90: [], 180: []}
    for r in results:
        by_angle[r["angle"]].append(r)
        print(f"  {r['key']}: {r['formula']} rot={r['axis']}({r['angle']}°) R²={r['r2']:.6f} MAE={r['mae']:.6f}")

    for angle in [90, 180]:
        sub = by_angle[angle]
        if sub:
            r2s = [r["r2"] for r in sub]
            maes = [r["mae"] for r in sub]
            best = max(sub, key=lambda x: x["r2"])
            worst = min(sub, key=lambda x: x["r2"])
            print(f"\n{angle}°: mean R²={np.mean(r2s):.4f}, mean MAE={np.mean(maes):.4f}")
            print(f"  Best:  {best['r2']:.4f} ({best['formula']})")
            print(f"  Worst: {worst['r2']:.4f} ({worst['formula']})")


if __name__ == "__main__":
    main()
