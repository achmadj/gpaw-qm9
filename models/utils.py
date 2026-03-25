"""Shared utility helpers for QM9 3D U-Net training & evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, best_val: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val", float("inf")))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error masked to physical (non-padded) voxels. Used by dense backend."""
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def formula_to_electron_count(formula: str) -> int:
    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    total = 0
    for symbol, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol not in atomic_numbers:
            raise ValueError(f"Unsupported element in formula: {symbol}")
        count = int(count_text) if count_text else 1
        total += atomic_numbers[symbol] * count
    return total

def compute_voxel_volume(cell_angstrom: np.ndarray, shape: tuple[int, int, int]) -> float:
    cell = np.asarray(cell_angstrom, dtype=np.float64)
    return float(abs(np.linalg.det(cell)) / np.prod(np.asarray(shape, dtype=np.float64)))
