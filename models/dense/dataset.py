"""QM9 HDF5 dataset loader for the dense 3D U-Net backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from models.config import INPUT_DATASET, PAD_MULTIPLE, TARGET_DATASET
from models.utils import formula_to_electron_count, compute_voxel_volume

SYMLOG_EPS = 1e-3

def symlog(x: np.ndarray, eps: float = SYMLOG_EPS) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x) / eps)

def symlog_inv(y: np.ndarray, eps: float = SYMLOG_EPS) -> np.ndarray:
    return np.sign(y) * eps * np.expm1(np.abs(y))


@dataclass(frozen=True)
class SampleMeta:
    key: str
    formula: str
    original_shape: tuple[int, int, int]
    dv: float
    n_electrons: int


class QM9DensityDataset(Dataset):
    """Loads top-level QM9 molecule groups from the fp32 HDF5 dataset."""

    def __init__(
        self,
        h5_path: str,
        keys: Sequence[str] | None = None,
        input_dataset: str = INPUT_DATASET,
        target_dataset: str = TARGET_DATASET,
        use_symlog: bool = False,
    ) -> None:
        self.h5_path = str(h5_path)
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self.use_symlog = use_symlog
        self._h5_file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as handle:
            all_keys = sorted(handle.keys()) if keys is None else list(keys)
            self.keys = [key for key in all_keys if key in handle]
            self.formulas = [
                _decode_attr(handle[key].attrs.get("formula", "unknown"))
                for key in self.keys
            ]

    def __len__(self) -> int:
        return len(self.keys)

    def _require_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, index: int):
        handle = self._require_file()
        key = self.keys[index]
        group = handle[key]

        input_array = np.asarray(group[self.input_dataset][...], dtype=np.float32)
        target_array = np.asarray(group[self.target_dataset][...], dtype=np.float32)
        
        if self.use_symlog:
            target_array = symlog(target_array)

        if input_array.shape != target_array.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {input_array.shape} vs {target_array.shape}"
            )

        input_tensor = torch.from_numpy(input_array).unsqueeze(0)
        target_tensor = torch.from_numpy(target_array).unsqueeze(0)
        
        formula = _decode_attr(group.attrs.get("formula", "unknown"))
        cell_angstrom = np.asarray(group.attrs["cell_angstrom"], dtype=np.float64)
        original_shape = tuple(int(v) for v in input_array.shape)
        
        meta = SampleMeta(
            key=key,
            formula=formula,
            original_shape=original_shape,
            dv=compute_voxel_volume(cell_angstrom, original_shape),
            n_electrons=formula_to_electron_count(formula),
        )
        return input_tensor, target_tensor, meta


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _pad_spatial(tensor: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    _, x, y, z = tensor.shape
    tx, ty, tz = target_shape
    padded = torch.zeros((tensor.shape[0], tx, ty, tz), dtype=tensor.dtype)
    padded[:, :x, :y, :z] = tensor
    return padded


def round_up_to_multiple(value: int, multiple: int = PAD_MULTIPLE) -> int:
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)


def dynamic_pad_collate(batch):
    """Pad each batch to a shape divisible by 8 and create a physical mask."""
    inputs, targets, metas = zip(*batch)

    max_x = max(t.shape[1] for t in inputs)
    max_y = max(t.shape[2] for t in inputs)
    max_z = max(t.shape[3] for t in inputs)
    padded_shape = (
        round_up_to_multiple(max_x),
        round_up_to_multiple(max_y),
        round_up_to_multiple(max_z),
    )

    batch_size = len(batch)
    batch_input = torch.zeros((batch_size, 1, *padded_shape), dtype=torch.float32)
    batch_target = torch.zeros((batch_size, 1, *padded_shape), dtype=torch.float32)
    batch_mask = torch.zeros((batch_size, 1, *padded_shape), dtype=torch.float32)

    for index, (input_tensor, target_tensor, meta) in enumerate(batch):
        x, y, z = meta.original_shape
        batch_input[index] = _pad_spatial(input_tensor, padded_shape)
        batch_target[index] = _pad_spatial(target_tensor, padded_shape)
        batch_mask[index, :, :x, :y, :z] = 1.0

    return batch_input, batch_target, batch_mask, list(metas)
