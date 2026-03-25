"""Unified configuration for QM9 3D U-Net (dense & sparse backends).

All training hyperparameters and paths live here. Backend-specific
defaults are clearly marked.
"""

from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # gpaw-qm9/
DATA_PATH = Path("/home/achmadjae/gpaw-qm9/dataset/gpaw_qm9_merged.h5")
EXPERIMENTS_ROOT = PROJECT_ROOT / "models" / "experiments"

# MinkowskiEngine paths (sparse backend only)
MINKOWSKI_REPO = Path("/home/achmadjae/MinkowskiEngine")
MINKOWSKI_BUILD_LIB = MINKOWSKI_REPO / "build" / "lib.linux-x86_64-cpython-311"

# ── Data ─────────────────────────────────────────────────────────────
INPUT_DATASET = "v_ext"
TARGET_DATASET = "n_r"
IN_CHANNELS = 1
OUT_CHANNELS = 1
VAL_SPLIT = 0.10
RANDOM_SEED = 42
MAX_SAMPLES = None
NUM_WORKERS = 4
PIN_MEMORY = True

# ── Model ────────────────────────────────────────────────────────────
BASE_CHANNELS = 16
DROPOUT = 0.05

# Dense-specific
PAD_MULTIPLE = 8        # three 2× downsamplings require multiples of 8
NUM_LEVELS = 3
NORM_GROUPS = 8

# Sparse-specific
DIMENSION = 3           # MinkowskiEngine spatial dimension

# ── Optimization ─────────────────────────────────────────────────────
BATCH_SIZE = 4          # conservative default; override via CLI
EPOCHS = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
AMP = True              # automatic mixed precision

GRADIENT_CHECKPOINTING = False   # Recompute activations saat backward untuk hemat VRAM
GRADIENT_ACCUMULATION_STEPS = 1  # 1 = nonaktif; >1 = akumulasi N step sebelum optimizer.step()

# ── Loss ─────────────────────────────────────────────────────────────
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.0
GRADIENT_WEIGHT = 0.05   # dense only: Sobolev-style gradient penalty

# ── Logging / Checkpoints ───────────────────────────────────────────
CHECKPOINT_EVERY = 5
BEST_CHECKPOINT_NAME = "best.pt"
LAST_CHECKPOINT_NAME = "last.pt"
