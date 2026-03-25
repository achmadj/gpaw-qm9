"""Sparse QM9 dataset utilities for MinkowskiEngine backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from models.config import INPUT_DATASET, TARGET_DATASET


@dataclass(frozen=True)
class SampleMeta:
    key: str
    formula: str
    original_shape: tuple[int, int, int]
    num_active: int


class QM9SparseDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        keys: Sequence[str] | None = None,
        input_dataset: str = INPUT_DATASET,
        target_dataset: str = TARGET_DATASET,
    ) -> None:
        self.h5_path = str(h5_path)
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self._h5_file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as handle:
            all_keys = sorted(handle.keys()) if keys is None else list(keys)
            self.keys = [key for key in all_keys if key in handle]
            self.formulas = [_decode_attr(handle[key].attrs.get("formula", "unknown")) for key in self.keys]

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

        v_ext = np.asarray(group[self.input_dataset][...], dtype=np.float32)
        n_r = np.asarray(group[self.target_dataset][...], dtype=np.float32)
        if v_ext.shape != n_r.shape:
            raise ValueError(f"Shape mismatch for {key}: {v_ext.shape} vs {n_r.shape}")

        active_mask = (v_ext != 0.0) | (n_r != 0.0)
        coords = np.argwhere(active_mask).astype(np.int32)
        if coords.size == 0:
            coords = np.zeros((1, 3), dtype=np.int32)

        feats = v_ext[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)[:, None]
        target = n_r[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)[:, None]

        meta = SampleMeta(
            key=key,
            formula=_decode_attr(group.attrs.get("formula", "unknown")),
            original_shape=tuple(int(v) for v in v_ext.shape),
            num_active=int(coords.shape[0]),
        )
        return coords, feats, target, meta


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sparse_collate_fn(batch):
    coords_list, feats_list, targets_list, metas = zip(*batch)
    return list(coords_list), list(feats_list), list(targets_list), list(metas)
