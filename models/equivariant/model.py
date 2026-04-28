"""SO(3)-equivariant 3D U-Net using escnn R3Conv.

Handles rotational equivariance for v_ion -> n_r regression.
Uses trivial (scalar, l=0) + vector (l=1) representations for
richer feature learning while maintaining SO(3) equivariance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from escnn import gspaces, nn as enn


def _build_field_type(gspace, n_scalar: int, n_vector: int = 0):
    """Helper to build FieldType from scalar + vector multiplicities."""
    reps = n_scalar * [gspace.trivial_repr]
    if n_vector > 0:
        reps += n_vector * [gspace.irrep(1)]
    return enn.FieldType(gspace, reps)


class EquivariantResBlock3D(nn.Module):
    """Residual block with R3Conv + BatchNorm + ELU."""

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType):
        super().__init__()
        self.conv1 = enn.R3Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = enn.IIDBatchNorm3d(out_type)
        self.act1 = enn.ELU(out_type, inplace=True)
        self.conv2 = enn.R3Conv(out_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = enn.IIDBatchNorm3d(out_type)
        self.act2 = enn.ELU(out_type, inplace=True)

        if in_type != out_type:
            self.skip = enn.R3Conv(in_type, out_type, kernel_size=1, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Residual addition (element-wise sum of tensors of same type)
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        return self.act2(out)


class EquivariantUpBlock3D(nn.Module):
    """Upsample + concat skip + R3Conv residual block."""

    def __init__(self, in_type: enn.FieldType, skip_type: enn.FieldType,
                 out_type: enn.FieldType):
        super().__init__()
        self.up = enn.R3Upsampling(in_type, scale_factor=2)
        # After concat: in_type + skip_type channels
        cat_type = in_type + skip_type
        self.conv = EquivariantResBlock3D(cat_type, out_type)

    def forward(self, x: enn.GeometricTensor,
                skip: enn.GeometricTensor) -> enn.GeometricTensor:
        x = self.up(x)
        x = _crop_or_pad_geometric(x, skip)
        x = enn.tensor_directsum([x, skip])
        return self.conv(x)


class EquivariantUNet3D(nn.Module):
    """
    SO(3)-equivariant 3D U-Net for scalar field regression.

    Architecture: 3-level encoder-decoder with skip connections.
    Uses escnn R3Conv with SO(3) group for rotational equivariance.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        max_freq: int = 2,
        last_activation: str = "relu",
    ):
        super().__init__()
        self.gspace = gspaces.rot3dOnR3(maximum_frequency=max_freq)
        gs = self.gspace
        c = base_channels

        # Define field types (all scalar for simplicity + stability)
        self.in_type = enn.FieldType(gs, in_channels * [gs.trivial_repr])
        t1 = enn.FieldType(gs, c * [gs.trivial_repr])
        t2 = enn.FieldType(gs, (c * 2) * [gs.trivial_repr])
        t3 = enn.FieldType(gs, (c * 4) * [gs.trivial_repr])
        t_bn = enn.FieldType(gs, (c * 8) * [gs.trivial_repr])
        self.out_type = enn.FieldType(gs, out_channels * [gs.trivial_repr])

        # Encoder
        self.enc1 = EquivariantResBlock3D(self.in_type, t1)
        self.pool1 = enn.PointwiseAvgPool3D(t1, kernel_size=2, stride=2)

        self.enc2 = EquivariantResBlock3D(t1, t2)
        self.pool2 = enn.PointwiseAvgPool3D(t2, kernel_size=2, stride=2)

        self.enc3 = EquivariantResBlock3D(t2, t3)
        self.pool3 = enn.PointwiseAvgPool3D(t3, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = EquivariantResBlock3D(t3, t_bn)

        # Decoder
        self.up3 = EquivariantUpBlock3D(t_bn, t3, t3)
        self.up2 = EquivariantUpBlock3D(t3, t2, t2)
        self.up1 = EquivariantUpBlock3D(t2, t1, t1)

        # Final 1x1 conv to output channels
        self.head = enn.R3Conv(t1, self.out_type, kernel_size=1)

        # Output activation (enforce non-negativity)
        if last_activation == "softplus":
            self.last_act = nn.Softplus()
        elif last_activation == "abs":
            self.last_act = torch.abs
        else:
            self.last_act = torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap raw tensor as GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        bn = self.bottleneck(self.pool3(e3))

        # Decoder path with skip connections
        d3 = self.up3(bn, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Output
        out = self.head(d1)
        # Extract raw tensor and apply activation (density must be non-negative)
        return self.last_act(out.tensor)


def _crop_or_pad_geometric(
    source: enn.GeometricTensor, reference: enn.GeometricTensor
) -> enn.GeometricTensor:
    """Crop or pad source GeometricTensor to match reference spatial dims."""
    s = source.tensor
    r = reference.tensor
    _, _, rx, ry, rz = r.shape
    _, _, sx, sy, sz = s.shape

    if (sx, sy, sz) == (rx, ry, rz):
        return source

    # Center crop if source is larger
    def _cc(t, dim, target):
        cur = t.shape[dim]
        if cur <= target:
            return t
        start = (cur - target) // 2
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(start, start + target)
        return t[tuple(idx)]

    s = _cc(s, 2, rx)
    s = _cc(s, 3, ry)
    s = _cc(s, 4, rz)
    sx, sy, sz = s.shape[2], s.shape[3], s.shape[4]

    # Center pad if source is smaller
    if (sx, sy, sz) != (rx, ry, rz):
        out = s.new_zeros((s.shape[0], s.shape[1], rx, ry, rz))
        px = (rx - sx) // 2
        py = (ry - sy) // 2
        pz = (rz - sz) // 2
        out[:, :, px : px + sx, py : py + sy, pz : pz + sz] = s
        s = out

    return enn.GeometricTensor(s, source.type)
