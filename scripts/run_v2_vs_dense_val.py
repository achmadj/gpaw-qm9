#!/usr/bin/env python3
"""Compare equivariant_v2 vs dense on validation-only molecules under rotation."""
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
EQUI_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/checkpoints/best.pt"
DENSE_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/dense_phase_a_v1/checkpoints/best.pt"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
VAL_KEYS = [
    ("gdb_1006", "C6H4O", "40×56×20", 2.8),
    ("gdb_1011", "C5H4N2", "44×48×20", 2.4),
    ("gdb_130830", "C3HN3O3", "36×48×20", 2.4),
    ("gdb_8719", "C5HNO2", "56×36×20", 2.8),
    ("gdb_16", "C3H6", "32×32×32", 1.0),
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

    equi_model = EquivariantUNet3D(in_channels=1, out_channels=1, base_channels=96,
                                    max_freq=2, last_activation="softplus")
    ckpt_e = torch.load(EQUI_CKPT, map_location="cpu", weights_only=False)
    equi_model.load_state_dict(ckpt_e["model_state_dict"] if "model_state_dict" in ckpt_e else ckpt_e)
    equi_model.to(device).eval()
    n_e = sum(p.numel() for p in equi_model.parameters())
    print(f"Equivariant: {n_e:,} params", flush=True)

    dense_model = SmallUNet3D(in_channels=1, out_channels=1, base_channels=32,
                               final_activation="softplus")
    ckpt_d = torch.load(DENSE_CKPT, map_location="cpu", weights_only=False)
    dense_model.load_state_dict(ckpt_d["model_state_dict"] if "model_state_dict" in ckpt_d else ckpt_d)
    dense_model.to(device).eval()
    n_d = sum(p.numel() for p in dense_model.parameters())
    print(f"Dense:       {n_d:,} params", flush=True)

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
    csv_path = OUT_DIR / "equiv_v2_vs_dense_rotation_val_only.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== VALIDATION-ONLY Summary ===")
    for rot_label, _, _ in rotations:
        sub = df[df["rotation"] == rot_label]
        print(f"\n{rot_label}:")
        print(f"  Equivariant: avg R²={sub['equi_r2'].mean():.4f}, avg MAE={sub['equi_mae'].mean():.5f}")
        print(f"  Dense:       avg R²={sub['dense_r2'].mean():.4f}, avg MAE={sub['dense_mae'].mean():.5f}")

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
