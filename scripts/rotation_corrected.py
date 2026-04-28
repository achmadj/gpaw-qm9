#!/usr/bin/env python3
"""Corrected rotation comparison: reshape=True (no zero-padding artifact).

Fixes the critical bug where reshape=False on 90° rotation introduced 40% zero
padding in anisotropic grids, corrupting the v_ion input to the model.

With reshape=True, grid dimensions properly swap: (32,48,20) → (20,48,32).
Both input and target are rotated with reshape=True, ensuring one-to-one voxel
correspondence between pred and target.

Models: equivariant_v2 (4.87M) and dense_phase_a_v1 (5.7M)
Molecules: 5 validation-only + 1 cubic control
"""

import sys, os
from pathlib import Path

import h5py, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, "/home/achmadjae/gpaw-qm9")
from models.equivariant.model import EquivariantUNet3D
from models.dense.model import SmallUNet3D

H5_PATH = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")
EQUI_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/checkpoints/best.pt"
DENSE_CKPT = "/home/achmadjae/gpaw-qm9/models/experiments/dense_phase_a_v1/checkpoints/best.pt"
PAD_MULTIPLE = 8
MAX_POINTS = 8000
ALPHA = 0.55
SIZE = 2.5
DPI = 120

# Validation-only molecules
VAL_MOLS = [
    ("gdb_1006", "C6H4O"),
    ("gdb_1011", "C5H4N2"),
    ("gdb_130830", "C3HN3O3"),
    ("gdb_8719", "C5HNO2"),
    ("gdb_16", "C3H6"),
]


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


def scatter_volume(ax, volume, mask, cmap, title, alpha, size, vmin=None, vmax=None):
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        ax.set_title(f"{title}\n(no voxels)")
        return
    x, y, z = np.unravel_index(flat_idx, volume.shape)
    values = volume[x, y, z]
    sc = ax.scatter(x, y, z, c=values, cmap=cmap, alpha=alpha, s=size,
                     linewidths=0, rasterized=True, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=5)
    plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.04)


def sample_points(mask, max_points, seed):
    f_idx = np.flatnonzero(mask.reshape(-1))
    if f_idx.size == 0:
        return f_idx
    if f_idx.size > max_points:
        f_idx = np.random.default_rng(seed).choice(f_idx, size=max_points, replace=False)
    return f_idx


def rotate_field(field, angle, axes):
    """Rotate with reshape=True — proper dimension swap, no zero-padding artifact."""
    return rotate(field, angle, axes=axes, reshape=True, order=0,
                  prefilter=False, mode="constant", cval=0.0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load models
    equi = EquivariantUNet3D(1, 1, base_channels=96, max_freq=2, last_activation="softplus")
    ckpt = torch.load(EQUI_CKPT, map_location="cpu", weights_only=False)
    equi.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    equi.to(device).eval()

    dense = SmallUNet3D(1, 1, base_channels=32, final_activation="softplus")
    ckpt = torch.load(DENSE_CKPT, map_location="cpu", weights_only=False)
    dense.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    dense.to(device).eval()
    print(f"Models loaded.", flush=True)

    rotations = [
        ("0° (original)", 0),
        ("90° y-axis", 90),
        ("180° y-axis", 180),
    ]
    rot_axes = (0, 2)  # y-axis: xz-plane

    rows = []
    with h5py.File(H5_PATH, "r") as f:
        for key, formula in VAL_MOLS:
            grp = f[key]
            vi_raw = np.asarray(grp["v_ion"], dtype=np.float32)
            nr_raw = np.asarray(grp["n_r"], dtype=np.float32)
            sh = vi_raw.shape
            print(f"\n{key} ({formula}) shape={sh} [VALIDATION]", flush=True)

            for rot_label, angle in rotations:
                if angle == 0:
                    vi, nr = vi_raw, nr_raw
                else:
                    vi = rotate_field(vi_raw, angle, rot_axes)
                    nr = rotate_field(nr_raw, angle, rot_axes)

                re = infer(equi, vi, device)
                rd = infer(dense, vi, device)

                r2e = float(r2_score(nr.ravel(), re.ravel()))
                r2d = float(r2_score(nr.ravel(), rd.ravel()))
                mae_e = float(mean_absolute_error(nr.ravel(), re.ravel()))
                mae_d = float(mean_absolute_error(nr.ravel(), rd.ravel()))

                winner = "EQUI" if r2e > r2d else "DENSE"
                print(f"  {rot_label:18s} [{vi.shape[0]}×{vi.shape[1]}×{vi.shape[2]}] "
                      f"equi R²={r2e:.4f} dense R²={r2d:.4f} {winner}", flush=True)

                rows.append({
                    "key": key, "formula": formula, "orig_shape": f"{sh[0]}×{sh[1]}×{sh[2]}",
                    "rot_shape": f"{vi.shape[0]}×{vi.shape[1]}×{vi.shape[2]}",
                    "rotation": rot_label, "angle": angle,
                    "equi_r2": r2e, "equi_mae": mae_e,
                    "dense_r2": r2d, "dense_mae": mae_d,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rotation_corrected_reshape_true.csv", index=False)

    # --- SUMMARY ---
    print("\n=== SUMMARY (reshape=True, no zero-padding) ===")
    for rot_label, _ in rotations:
        s = df[df["rotation"] == rot_label]
        print(f"  {rot_label}: equi avg R²={s['equi_r2'].mean():.4f} dense avg R²={s['dense_r2'].mean():.4f}")

    print("\n=== 90° DETAIL ===")
    s90 = df[df["rotation"] == "90° y-axis"]
    for _, r in s90.iterrows():
        print(f"  {r['key']} ({r['formula']}) {r['rot_shape']}: equi={r['equi_r2']:.4f} dense={r['dense_r2']:.4f}")

    # --- BAR CHART ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (rot_label, _) in enumerate(rotations):
        ax = axes[i]
        sub = df[df["rotation"] == rot_label]
        x = np.arange(len(sub)); w = 0.35
        ax.bar(x - w/2, sub["equi_r2"], w, label="Equivariant v2 (4.87M)", color="#2166ac", alpha=0.85)
        ax.bar(x + w/2, sub["dense_r2"], w, label="Dense (5.7M)", color="#b2182b", alpha=0.85)
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.text(j - w/2, r["equi_r2"] + 0.005, f"{r['equi_r2']:.3f}", ha="center", va="bottom", fontsize=6)
            ax.text(j + w/2, r["dense_r2"] + 0.005, f"{r['dense_r2']:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['formula']}\n{r['rot_shape']}" for _, r in sub.iterrows()], fontsize=6)
        ax.set_ylim(0.80, 1.01); ax.set_ylabel("R²")
        ax.set_title(f"{rot_label}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle("Equivariant v2 vs Dense: R² under Grid Rotation — CORRECTED (reshape=True, no zero-padding artifact)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rotation_corrected_bar.svg", dpi=150, bbox_inches="tight", format="svg")
    fig.savefig(OUT_DIR / "rotation_corrected_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 3×4 SCATTER for gdb_1006 (C6H4O) with equivariant_v2 ---
    key = "gdb_1006"
    with h5py.File(H5_PATH, "r") as f:
        grp = f[key]
        formula = grp.attrs["formula"]
        vi_raw = np.asarray(grp["v_ion"], dtype=np.float32)
        nr_raw = np.asarray(grp["n_r"], dtype=np.float32)

    scenarios = [
        ("Original", vi_raw, nr_raw, infer(equi, vi_raw, device)),
        ("90° y-axis", rotate_field(vi_raw, 90, rot_axes),
         rotate_field(nr_raw, 90, rot_axes),
         infer(equi, rotate_field(vi_raw, 90, rot_axes), device)),
        ("180° y-axis", rotate_field(vi_raw, 180, rot_axes),
         rotate_field(nr_raw, 180, rot_axes),
         infer(equi, rotate_field(vi_raw, 180, rot_axes), device)),
    ]

    fig2 = plt.figure(figsize=(22, 16))
    for row_idx, (label, vi, nr, pr) in enumerate(scenarios):
        diff = pr - nr
        dt = float(np.quantile(np.abs(diff[diff != 0]), 0.75)) if (diff != 0).any() else 0.0
        for col_idx, (data, mask, cmap, ctitle, vmin, vmax) in enumerate([
            (vi, np.abs(vi) >= np.quantile(np.abs(vi[vi != 0]), 0.3) if (vi != 0).any() else np.ones_like(vi, dtype=bool),
             "viridis", "V_ion (input)", None, None),
            (pr, pr >= 0.01, "hot", "n_pred", 0, None),
            (nr, nr >= 0.01, "hot", "n_true", 0, None),
            (diff, np.abs(diff) >= dt, "bwr", "Δn = pred - true",
             -float(max(abs(diff.min()), abs(diff.max()))), float(max(abs(diff.min()), abs(diff.max())))),
        ]):
            ax = fig2.add_subplot(3, 4, row_idx * 4 + col_idx + 1, projection="3d")
            f_idx = sample_points(mask, MAX_POINTS, seed=42 + row_idx * 10 + col_idx)
            sm = np.zeros_like(mask, dtype=bool)
            if f_idx.size > 0: sm.reshape(-1)[f_idx] = True
            scatter_volume(ax, data, sm, cmap, ctitle, ALPHA, SIZE, vmin, vmax)
        r2 = r2_score(nr.ravel(), pr.ravel())
        print(f"3x4 [{label}] R²={r2:.4f}, shape={vi.shape}", flush=True)

    fig2.suptitle(f"Equivariant v2: Density Prediction under Grid Rotation — CORRECTED\n"
                  f"{formula} ({key})",
                  fontsize=11, fontweight="bold", y=1.01)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "equivariant_v2_3x4_corrected.svg", dpi=DPI, bbox_inches="tight", format="svg")
    fig2.savefig(OUT_DIR / "equivariant_v2_3x4_corrected.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nOutputs in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
