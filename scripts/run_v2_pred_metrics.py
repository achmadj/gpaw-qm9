#!/usr/bin/env python3
"""Prediction metrics for equivariant_v2 on 5 specific validation molecules."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, "/home/achmadjae/gpaw-qm9")
from models.equivariant.model import EquivariantUNet3D

H5_PATH = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
CKPT_PATH = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/checkpoints/best.pt"
PAD_MULTIPLE = 8

# Target formulas from original table
TARGET_FORMULAS = ["C3H4O2", "C5HNO2", "C3H4N2O", "CHN3O2", "C3HN3O3"]


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=96,
                               max_freq=2, last_activation="softplus")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    rows = []
    with h5py.File(H5_PATH, "r") as f:
        # Build formula -> key mapping
        formula_to_keys = {}
        for key in f.keys():
            formula = f[key].attrs.get("formula", "?")
            if isinstance(formula, bytes):
                formula = formula.decode("utf-8")
            formula_to_keys.setdefault(formula, []).append(key)

        for target_formula in TARGET_FORMULAS:
            if target_formula not in formula_to_keys:
                print(f"WARNING: formula {target_formula} not found in dataset")
                continue
            # Pick first occurrence
            key = formula_to_keys[target_formula][0]
            grp = f[key]
            formula = grp.attrs.get("formula", "?")
            if isinstance(formula, bytes):
                formula = formula.decode("utf-8")
            v_ion_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            n_r_raw = np.asarray(grp["n_r"], dtype=np.float32)
            sh = v_ion_raw.shape

            pred = infer(model, v_ion_raw, device)
            r2 = float(r2_score(n_r_raw.ravel(), pred.ravel()))
            mae = float(mean_absolute_error(n_r_raw.ravel(), pred.ravel()))
            pred_peak = float(pred.max())
            true_peak = float(n_r_raw.max())

            rows.append({
                "key": key, "formula": formula, "atoms": int(grp.attrs.get("num_atoms", 0)),
                "shape": f"{sh[0]}×{sh[1]}×{sh[2]}",
                "mae": mae, "r2": r2,
                "peak_pred": pred_peak, "peak_true": true_peak,
            })
            print(f"{key} ({formula}): MAE={mae:.5f}, R²={r2:.4f}, Peak={pred_peak:.2f}/{true_peak:.2f}")

    print("\n=== Summary Table ===")
    print(f"{'Molecule':<15} {'Atoms':>6} {'Grid':>18} {'MAE':>10} {'R²':>8} {'Peak (pred/true)':>20}")
    print("-" * 85)
    for r in rows:
        print(f"{r['formula']:<15} {r['atoms']:>6} {r['shape']:>18} {r['mae']:>10.5f} {r['r2']:>8.4f} {r['peak_pred']:.2f} / {r['peak_true']:.2f}")
    if rows:
        avg_mae = np.mean([r['mae'] for r in rows])
        avg_r2 = np.mean([r['r2'] for r in rows])
        print("-" * 85)
        print(f"{'Average':<15} {'':>6} {'':>18} {avg_mae:>10.5f} {avg_r2:>8.4f}")


if __name__ == "__main__":
    main()
