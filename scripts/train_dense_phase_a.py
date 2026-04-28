"""
Training script for the standard (non-equivariant) Dense 3D U-Net on QM9 v_ion -> n_r.

Identical preprocessing, loss, and training protocol as equivariant_v1.
Only the architecture changes: SmallUNet3D instead of EquivariantUNet3D.

Usage:
  conda activate 3d-unet-qm9
  python scripts/train_dense_phase_a.py \
      --data dataset/qm9_1000_phase_a.h5 \
      --experiment-name dense_phase_a_v1 \
      --epochs 100 --batch-size 8 --lr 3e-4 --base-channels 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

_project_root = "/home/achmadjae/gpaw-qm9"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.dense.model import SmallUNet3D


# ── Dataset ──────────────────────────────────────────────────────────
PAD_MULTIPLE = 8


def round_up(v, m=PAD_MULTIPLE):
    return v if v % m == 0 else v + (m - v % m)


class PhaseADataset(Dataset):
    """Loads v_ion (input) and n_r (target) from Phase A HDF5."""

    def __init__(self, h5_path: str, keys: list[str] | None = None):
        self.h5_path = str(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self.keys = sorted(f.keys()) if keys is None else list(keys)

    def __len__(self):
        return len(self.keys)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        f = self._file()
        key = self.keys[idx]
        grp = f[key]
        v_ion = np.asarray(grp["v_ion"], dtype=np.float32)
        n_r = np.asarray(grp["n_r"], dtype=np.float32)
        mu = v_ion.mean()
        sigma = v_ion.std() + 1e-8
        v_ion = (v_ion - mu) / sigma
        inp = torch.from_numpy(v_ion).unsqueeze(0)
        tgt = torch.from_numpy(n_r).unsqueeze(0)
        return inp, tgt, v_ion.shape


def pad_collate(batch):
    inputs, targets, shapes = zip(*batch)
    mx = max(t.shape[1] for t in inputs)
    my = max(t.shape[2] for t in inputs)
    mz = max(t.shape[3] for t in inputs)
    ps = (round_up(mx), round_up(my), round_up(mz))
    B = len(batch)
    bi = torch.zeros(B, 1, *ps)
    bt = torch.zeros(B, 1, *ps)
    bm = torch.zeros(B, 1, *ps)
    for i, (inp, tgt, sh) in enumerate(batch):
        x, y, z = sh
        bi[i, :, :x, :y, :z] = inp
        bt[i, :, :x, :y, :z] = tgt
        bm[i, :, :x, :y, :z] = 1.0
    return bi, bt, bm


# ── Loss ─────────────────────────────────────────────────────────────
class MaskedL1Loss(nn.Module):
    def forward(self, pred, target, mask):
        diff = torch.abs(pred - target) * mask
        return diff.sum() / mask.sum().clamp(min=1)


# ── Helpers ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_keys(keys, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    rng.shuffle(idx)
    n_val = max(1, int(len(keys) * val_frac))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


# ── Training ─────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    steps = 0
    for inputs, targets, mask in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            preds = model(inputs)
            loss = criterion(preds, targets, mask)
        if not torch.isfinite(loss):
            continue
        if training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser(description="Train standard Dense 3D U-Net on Phase A")
    parser.add_argument("--data", required=True, help="Path to Phase A HDF5")
    parser.add_argument("--experiment-name", default="dense_phase_a_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--last-activation", default="softplus",
                        choices=["relu", "softplus"],
                        help="Output activation to enforce non-negativity")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(_project_root) / "models/experiments" / args.experiment_name
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = PhaseADataset(args.data)
    train_idx, val_idx = split_keys(dataset.keys, args.val_split, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=pad_collate,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=pad_collate,
                            pin_memory=True)

    print(f"Dataset: {len(dataset)} total, {len(train_set)} train, {len(val_set)} val")

    model = SmallUNet3D(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
        final_activation=args.last_activation,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SmallUNet3D, {n_params:,} parameters")
    print(f"  base_channels={args.base_channels}, last_act={args.last_activation}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = MaskedL1Loss()
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["n_params"] = n_params
    config["train_samples"] = len(train_set)
    config["val_samples"] = len(val_set)
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    history_path = log_dir / "history.jsonl"
    print(f"\nStarting training for {args.epochs} epochs on {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               device, scaler)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, None,
                                 device, None)
        dt = time.time() - t0
        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_s": dt,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        tag = ""
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }, ckpt_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }, ckpt_dir / "best.pt")
            tag = " ★ best"

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {dt:.1f}s{tag}")

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
