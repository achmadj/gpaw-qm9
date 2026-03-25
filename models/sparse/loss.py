"""Sparse regression losses for QM9 density prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseDensityLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, mse_weight: float = 0.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, pred_features: torch.Tensor, target_features: torch.Tensor):
        diff = pred_features - target_features
        l1 = diff.abs().mean()
        mse = (diff ** 2).mean()
        total = self.l1_weight * l1 + self.mse_weight * mse
        stats = {
            "l1": l1.detach(),
            "mse": mse.detach(),
            "total": total.detach(),
        }
        return total, stats
