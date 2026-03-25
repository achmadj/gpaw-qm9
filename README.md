# GPAW QM9 — Electron Density Prediction

Predict electron density fields (`n_r`) from electrostatic potentials (`v_ext`) for ~20,000 QM9 molecules using 3D U-Net architectures. DFT ground truth computed with GPAW.

---

## Directory Structure

```
gpaw-qm9/
├── README.md               ← you are here
├── models/                  ← ML code (unified)
│   ├── train.py             ← single entry point: --backend dense|sparse
│   ├── evaluate.py          ← single-sample evaluation + plots
│   ├── config.py            ← all hyperparameters & paths
│   ├── utils.py             ← shared helpers (checkpoint I/O, etc.)
│   ├── dense/               ← dense backend (PyTorch Conv3d)
│   │   ├── model.py         ← SmallUNet3D
│   │   ├── dataset.py       ← QM9DensityDataset + pad collate
│   │   └── loss.py          ← MaskedDensityLoss
│   ├── sparse/              ← sparse backend (MinkowskiEngine)
│   │   ├── model.py         ← SparseUNet3D
│   │   ├── dataset.py       ← QM9SparseDataset + sparse collate
│   │   └── loss.py          ← SparseDensityLoss
│   └── experiments/         ← training run outputs (gitignored)
│       └── <experiment>/
│           ├── checkpoints/  ← best.pt, last.pt, epoch_N.pt
│           ├── logs/         ← history.jsonl, run_config.json
│           └── eval/         ← evaluation plots & metrics
├── dataset/                 ← ML-ready HDF5 files (symlinked, gitignored)
├── data/raw/                ← raw QM9 source data
├── scripts/                 ← data analysis & visualization
│   ├── analyze_qm9_value_distributions.py
│   ├── threshold_gpaw_qm9_h5.py
│   ├── plot_v_ext_fp64_vs_fp32.py
│   ├── plot_n_r_fp64_vs_fp32.py
│   └── ...
├── src/                     ← GPAW pipeline scripts
│   ├── run_gpaw_from_h5.py  ← run GPAW DFT on QM9 molecules
│   ├── merge_h5_shards.py   ← merge shard outputs
│   └── ...
├── jobs/                    ← Slurm job scripts
│   ├── submit_dense.sh      ← train dense backend
│   ├── submit_sparse.sh     ← train sparse backend
│   ├── resume_sparse.sh     ← resume sparse training
│   └── submit_gpaw_test.sh  ← run GPAW test
├── gpaw_logs/               ← GPAW computation logs (gitignored)
└── gpaw_analysis_outputs/   ← analysis script outputs (gitignored)
```

---

## Quick Start

### Prerequisites

Two conda environments are required:

| Environment | Purpose | Key packages |
|-------------|---------|-------------|
| `research`  | ML training (dense backend) | PyTorch, h5py, tqdm, matplotlib |
| `cuda118`   | ML training (sparse backend) | PyTorch, MinkowskiEngine |
| `gpaw`      | DFT calculations | GPAW, ASE, h5py |

### Train (Dense Backend)

```bash
cd gpaw-qm9
conda activate research

python models/train.py \
    --backend dense \
    --experiment-name dense_v1 \
    --data dataset/gpaw_qm9_20k.h5 \
    --epochs 80 \
    --batch-size 32
```

Or submit via Slurm:
```bash
sbatch jobs/submit_dense.sh
```

### Train (Sparse Backend)

> **⚠️ Blocked**: MinkowskiEngine needs to be recompiled for the current CUDA version.

```bash
conda activate cuda118

python models/train.py \
    --backend sparse \
    --experiment-name sparse_v1 \
    --data dataset/gpaw_qm9_20k.h5 \
    --epochs 300 \
    --batch-size 64 \
    --no-amp
```

### Resume Training

```bash
python models/train.py \
    --backend dense \
    --experiment-name dense_v1 \
    --resume models/experiments/dense_v1/checkpoints/last.pt \
    --data dataset/gpaw_qm9_20k.h5
```

### Evaluate

```bash
python models/evaluate.py \
    --data dataset/gpaw_qm9_20k.h5 \
    --checkpoint models/experiments/dense_v1/checkpoints/best.pt \
    --output-dir models/experiments/dense_v1/eval \
    --max-samples 1000 --seed 42
```

### All CLI Options

```bash
python models/train.py --help
python models/evaluate.py --help
```

---

## Dataset

| File | Size | Description |
|------|------|-------------|
| `gpaw_qm9_20k.h5` | ~25 GB | Base fp64 dataset (19,951 molecules) |
| `gpaw_qm9_20k.h5` | ~1 GB | **Thresholded** — used for training |

Each HDF5 group contains:
- **`v_ext`** — electrostatic potential (input, 1 channel)
- **`n_r`** — pseudo electron density (target, 1 channel)
- **Attributes**: `formula`, `smiles`, `num_atoms`, `cell_angstrom`, `grid_spacing_angstrom`

Shapes are **variable** across molecules (e.g. `28×28×32`, `24×28×20`). The dense backend pads to multiples of 8; the sparse backend uses only active (non-zero) voxels.

### Thresholding

| Field | Rule | Effect |
|-------|------|--------|
| `v_ext` | if `v_ext >= -0.5` → 0 | Zeros ~80.6% of voxels |
| `n_r` | if `n_r <= 0.05` → 0 | Zeros ~86.1% of voxels |

---

## Physics Notes

- **`n_r`** is pseudo/valence density from `calc.get_pseudo_density()` — does NOT sum to total electrons
- **`v_ext`** is a potential — do NOT add it to `n_r`
- After thresholding, `∫n_r dV` is lower than valence electron count (accepted trade-off for sparsity)
- Dense baseline achieved **R² = 0.9971** on single-sample eval after ~80 epochs

---

## GPAW Pipeline

To re-run DFT calculations (requires `gpaw` conda env):

```bash
conda activate gpaw
sbatch jobs/submit_gpaw_test.sh
```

Pipeline scripts in `src/`:
- `run_gpaw_from_h5.py` — parallel GPAW computation
- `merge_h5_shards.py` — merge output shards
- `validate_h5.py` — validate HDF5 structure

Analysis scripts in `scripts/`:
- `analyze_qm9_value_distributions.py` — global histograms
- `threshold_gpaw_qm9_h5.py` — apply physics-based thresholds
- `plot_v_ext_fp64_vs_fp32.py` / `plot_n_r_fp64_vs_fp32.py` — 3D scatter visualization

---

## Known Issues

1. **MinkowskiEngine install**: The sparse backend requires a compiled MinkowskiEngine compatible with the cluster's CUDA version. Currently blocked.
2. **Electron count conservation**: The model does not enforce exact charge conservation. Predicted ∫n_r dV deviates from expected valence count, especially after thresholding.
