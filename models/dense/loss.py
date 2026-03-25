"""Masked losses for QM9 density regression (dense backend)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedDensityLoss(nn.Module):
    """Masked regression loss for the dense 3D U-Net."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.0,
        gradient_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_masked = pred * mask
        target_masked = target * mask
        denom = mask.sum().clamp(min=1.0)

        l1 = torch.abs(pred_masked - target_masked).sum() / denom
        mse = ((pred_masked - target_masked) ** 2).sum() / denom
        grad = self._gradient_loss(pred_masked, target_masked, mask)

        total = (
            self.l1_weight * l1
            + self.mse_weight * mse
            + self.gradient_weight * grad
        )
        stats = {
            "l1": l1.detach(),
            "mse": mse.detach(),
            "grad": grad.detach(),
            "total": total.detach(),
        }
        return total, stats

    def _gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_weight <= 0:
            return pred.new_tensor(0.0)

        loss = pred.new_tensor(0.0)
        for axis in (2, 3, 4):
            left = [slice(None)] * 5
            right = [slice(None)] * 5
            left[axis] = slice(1, None)
            right[axis] = slice(None, -1)
            pred_grad = pred[left] - pred[right]
            target_grad = target[left] - target[right]
            valid = mask[left] * mask[right]
            denom = valid.sum().clamp(min=1.0)
            loss = loss + (torch.abs(pred_grad - target_grad) * valid).sum() / denom
        return loss / 3.0


class PhysicsAwareLoss(nn.Module):
    """Physics-aware loss function for raw density training without symlog."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        conservation_weight: float = 0.1,
        gradient_weight: float = 0.05,
        use_density_weighting: bool = True,
        use_pnll: bool = False,
        eps_low: float = 1e-3,
        eps_high: float = 10.0,
        pnll_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.conservation_weight = conservation_weight
        self.gradient_weight = gradient_weight
        self.use_density_weighting = use_density_weighting
        self.use_pnll = use_pnll
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.pnll_eps = pnll_eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        dv: float = 1.0,
        n_electrons: torch.Tensor | None = None,
        pred_physical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        # Main Regression Component: PNLL or L1
        if self.use_pnll:
            # Poisson NLL: pred - target * log(pred + eps)
            # We assume pred is physically consistent (non-negative)
            pnll = pred - target * torch.log(pred + self.pnll_eps)
            main_loss = (pnll * mask).sum() / mask.sum().clamp(min=1.0)
            l1 = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1.0) # Still track L1
        else:
            # Component 1: Density-Weighted L1 (Invers-Proporsional) or Plain L1
            if self.use_density_weighting:
                weights = 1.0 / target.clamp(min=self.eps_low, max=self.eps_high)
                weights = weights * mask
                w_sum = weights.sum().clamp(min=1.0)
                l1 = (weights * torch.abs(pred - target)).sum() / w_sum
            else:
                denom = mask.sum().clamp(min=1.0)
                l1 = (torch.abs(pred - target) * mask).sum() / denom
            main_loss = l1

        # Component 2: Electron Conservation Penalty
        pred_for_cons = pred_physical if pred_physical is not None else pred
        if n_electrons is not None and self.conservation_weight > 0:
            pred_ne = (pred_for_cons * mask).sum(dim=(1, 2, 3, 4)) * dv
            n_e = n_electrons.float().to(pred.device)
            cons = ((pred_ne - n_e) / n_e.clamp(min=1.0)).pow(2).mean()
        else:
            cons = pred.new_tensor(0.0)

        # Component 3: Sobolev / Gradient Loss
        grad_loss = self._gradient_loss(pred, target, mask)

        total = (
            self.l1_weight * main_loss
            + self.conservation_weight * cons
            + self.gradient_weight * grad_loss
        )
        stats = {
            "l1": l1.detach() if self.use_pnll else main_loss.detach(),
            "cons": cons.detach(),
            "grad": grad_loss.detach(),
            "total": total.detach(),
        }
        if self.use_pnll:
            stats["pnll"] = main_loss.detach()
            
        return total, stats

    def _gradient_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_weight <= 0:
            return pred.new_tensor(0.0)
            
        loss = pred.new_tensor(0.0)
        # Compute finite differences along X, Y, Z (dims 2, 3, 4)
        for axis in (2, 3, 4):
            sl_l = [slice(None)] * 5
            sl_r = [slice(None)] * 5
            sl_l[axis] = slice(1, None)
            sl_r[axis] = slice(None, -1)
            
            dpred = pred[sl_l] - pred[sl_r]
            dtarget = target[sl_l] - target[sl_r]
            valid = mask[sl_l] * mask[sl_r]
            
            denom = valid.sum().clamp(min=1.0)
            loss = loss + (torch.abs(dpred - dtarget) * valid).sum() / denom
        return loss / 3.0
