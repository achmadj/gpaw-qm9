"""Compact 3D U-Net for QM9 `v_ext -> n_r` regression (dense PyTorch backend)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8) -> None:
        super().__init__()
        groups = _group_count(out_channels, norm_groups)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResidualBlock3D(out_channels + skip_channels, out_channels, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _crop_or_pad_to_match(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))
class SmallUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.05,
        norm_groups: int = 8,
        final_activation: str = "relu",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.gradient_checkpointing = gradient_checkpointing

        self.enc1 = ResidualBlock3D(in_channels, c, norm_groups=1)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResidualBlock3D(c, c * 2, norm_groups=norm_groups)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResidualBlock3D(c * 2, c * 4, norm_groups=norm_groups)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ResidualBlock3D(c * 4, c * 8, norm_groups=norm_groups)
        self.dropout = nn.Dropout3d(dropout)

        self.up3 = UpBlock3D(c * 8, c * 4, c * 4, norm_groups=norm_groups)
        self.up2 = UpBlock3D(c * 4, c * 2, c * 2, norm_groups=norm_groups)
        self.up1 = UpBlock3D(c * 2, c, c, norm_groups=norm_groups)
        self.head = nn.Conv3d(c, out_channels, kernel_size=1)

        if final_activation == "softplus":
            self.output_activation = nn.Softplus()
        elif final_activation == "relu":
            self.output_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown final_activation: {final_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._ckpt(self.enc1, x)
        e2 = self._ckpt(self.enc2, self.pool1(e1))
        e3 = self._ckpt(self.enc3, self.pool2(e2))
        bottleneck = self.dropout(self._ckpt(self.bottleneck, self.pool3(e3)))
        d3 = self._ckpt(lambda b, s: self.up3(b, s), bottleneck, e3)
        d2 = self._ckpt(lambda b, s: self.up2(b, s), d3, e2)
        d1 = self._ckpt(lambda b, s: self.up1(b, s), d2, e1)
        return self.output_activation(self.head(d1))

    def _ckpt(self, module, *args):
        """Helper: run module with checkpoint if active, normal otherwise."""
        if self.gradient_checkpointing and self.training:
            # use_reentrant=False is modern and more stable with AMP/DDP
            return torch_checkpoint(module, *args, use_reentrant=False)
        return module(*args)


def _crop_or_pad_to_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    _, _, rx, ry, rz = reference.shape
    _, _, sx, sy, sz = source.shape
    if (sx, sy, sz) == (rx, ry, rz):
        return source

    def _center_crop(t: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
        cur = t.shape[dim]
        if cur <= target_size:
            return t
        start = (cur - target_size) // 2
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(start, start + target_size)
        return t[idx]

    # Crop dimensi yang terlalu besar
    source = _center_crop(source, 2, rx)
    source = _center_crop(source, 3, ry)
    source = _center_crop(source, 4, rz)
    sx, sy, sz = source.shape[2], source.shape[3], source.shape[4]

    # Center-pad dimensi yang masih kurang
    if (sx, sy, sz) != (rx, ry, rz):
        out = source.new_zeros((source.shape[0], source.shape[1], rx, ry, rz))
        px = (rx - sx) // 2
        py = (ry - sy) // 2
        pz = (rz - sz) // 2
        out[:, :, px:px+sx, py:py+sy, pz:pz+sz] = source
        return out

    return source
