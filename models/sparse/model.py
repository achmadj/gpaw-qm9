"""Small sparse 3D U-Net for QM9 using MinkowskiEngine."""

from __future__ import annotations

import sys
from importlib import import_module

import torch

from models.config import MINKOWSKI_BUILD_LIB, MINKOWSKI_REPO


def _import_minkowski():
    errors = []
    for extra_path in (None, str(MINKOWSKI_BUILD_LIB), str(MINKOWSKI_REPO)):
        if extra_path is not None and extra_path not in sys.path:
            sys.path.insert(0, extra_path)
        try:
            me = import_module("MinkowskiEngine")
            mf = import_module("MinkowskiEngine.MinkowskiFunctional")
            return me, mf
        except Exception as exc:  # pragma: no cover
            errors.append(f"{extra_path or 'default'}: {exc}")

    raise ImportError(
        "Could not import MinkowskiEngine. Install it into the active environment "
        "or provide a compatible local build. Attempts: " + " | ".join(errors)
    )


ME, MF = _import_minkowski()


class SparseUNet3D(ME.MinkowskiNetwork):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16, dropout: float = 0.05, D: int = 3):
        super().__init__(D)
        c = base_channels
        self.dropout = dropout

        self.enc1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(in_channels, c, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(c),
        )
        self.enc2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(c, c * 2, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c * 2),
        )
        self.enc3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(c * 2, c * 4, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c * 4),
        )

        self.bottleneck = torch.nn.Sequential(
            ME.MinkowskiConvolution(c * 4, c * 8, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c * 8),
        )
        self.dropout_layer = ME.MinkowskiDropout(dropout)

        self.up3 = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(c * 8, c * 4, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c * 4),
        )
        self.dec3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(c * 8, c * 4, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(c * 4),
        )
        self.up2 = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(c * 4, c * 2, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c * 2),
        )
        self.dec2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(c * 4, c * 2, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(c * 2),
        )
        self.up1 = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(c * 2, c, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(c),
        )
        self.dec1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(c * 2, c, kernel_size=3, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(c),
        )
        self.final = ME.MinkowskiConvolution(c, out_channels, kernel_size=1, bias=True, dimension=D)

    def forward(self, x):
        e1 = MF.relu(self.enc1(x))
        e2 = MF.relu(self.enc2(e1))
        e3 = MF.relu(self.enc3(e2))
        b = MF.relu(self.bottleneck(e3))
        b = self.dropout_layer(b)

        d3 = MF.relu(self.up3(b))
        d3 = ME.cat(d3, e3)
        d3 = MF.relu(self.dec3(d3))

        d2 = MF.relu(self.up2(d3))
        d2 = ME.cat(d2, e2)
        d2 = MF.relu(self.dec2(d2))

        d1 = MF.relu(self.up1(d2))
        d1 = ME.cat(d1, e1)
        d1 = MF.relu(self.dec1(d1))

        out = self.final(d1)
        return out
