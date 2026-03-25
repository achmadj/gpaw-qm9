#!/usr/bin/env python3
"""Unified training entry point for QM9 3D U-Net.

Usage:
    # Dense backend (standard PyTorch Conv3d) on specific GPU
    python models/train.py --backend dense --gpus 1 --data dataset/...

    # Multi-GPU (DDP) on all available GPUs
    python models/train.py --backend dense --gpus -1 --batch-size 32

    # Sparse backend (MinkowskiEngine) - automatically falls back to GPU 0
    python models/train.py --backend sparse --gpus -1

    # Resume dari last.pt (otomatis dicari di experiment dir)
    python models/train.py --backend dense --experiment-name dense_v3 --resume
"""

from __future__ import annotations

import os
import tempfile

# Redirect TMPDIR ke local disk node untuk menghindari NFS silly rename
_local_tmp_candidates = ["/tmp", "/var/tmp", "/scratch"]
for _candidate in _local_tmp_candidates:
    if os.path.isdir(_candidate) and os.access(_candidate, os.W_OK):
        os.environ["TMPDIR"] = _candidate
        tempfile.tempdir = _candidate
        break

import argparse
from contextlib import nullcontext
import importlib
import importlib.util
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Ensure the project root is on sys.path so `models.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.config import (
    AMP,
    BASE_CHANNELS,
    BATCH_SIZE,
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_EVERY,
    DATA_PATH,
    DIMENSION,
    DROPOUT,
    EPOCHS,
    EXPERIMENTS_ROOT,
    GRAD_CLIP_NORM,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    GRADIENT_WEIGHT,
    IN_CHANNELS,
    INPUT_DATASET,
    LAST_CHECKPOINT_NAME,
    LEARNING_RATE,
    L1_WEIGHT,
    MAX_SAMPLES,
    MSE_WEIGHT,
    NORM_GROUPS,
    NUM_WORKERS,
    OUT_CHANNELS,
    PIN_MEMORY,
    RANDOM_SEED,
    TARGET_DATASET,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from models.utils import ensure_dir, load_checkpoint, masked_mae, save_checkpoint, write_json

_tqdm_mod = importlib.import_module("tqdm") if importlib.util.find_spec("tqdm") else None


def _tqdm(iterable, **kwargs):
    if _tqdm_mod is None or (dist.is_initialized() and dist.get_rank() != 0):
        return iterable
    return _tqdm_mod.tqdm(iterable, **kwargs)


# ─── DDP Helpers ─────────────────────────────────────────────────────

def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def reduce_metrics(metrics: dict[str, float], world_size: int) -> dict[str, float]:
    """Average metrics across all processes."""
    if world_size <= 1:
        return metrics

    reduced = {}
    for key, val in metrics.items():
        tensor = torch.tensor(val, device=torch.cuda.current_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item() / world_size
    return reduced


# ─── Argument parsing ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QM9 3D U-Net (dense or sparse backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend
    parser.add_argument("--backend", choices=["dense", "sparse"], default="dense",
                        help="Convolution backend")
    parser.add_argument("--experiment-name", default="default",
                        help="Subdirectory under models/experiments/ untuk output")

    # GPU
    parser.add_argument("--gpus", type=int, default=0,
                        help="GPU index. Gunakan -1 untuk semua GPU (DDP)")

    # Data
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="Path ke HDF5 dataset")
    parser.add_argument("--input-dataset", default=INPUT_DATASET,
                        help="Nama dataset input di HDF5")
    parser.add_argument("--target-dataset", default=TARGET_DATASET,
                        help="Nama dataset target di HDF5")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT,
                        help="Fraksi data untuk validasi")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Batasi jumlah sampel (None = semua)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Jumlah DataLoader worker")
    try:
        parser.add_argument("--use-symlog", action=argparse.BooleanOptionalAction, default=False,
                            help="Gunakan symmetric log transform pada target density")
    except AttributeError:
        parser.add_argument("--use-symlog", action="store_true", default=False,
                            help="Gunakan symmetric log transform pada target density")
        parser.add_argument("--no-use-symlog", action="store_false", dest="use_symlog",
                            help="Nonaktifkan symlog transform")

    # Model
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS,
                        help="Base channel width U-Net")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="Dropout probability di bottleneck")
    parser.add_argument("--norm-groups", type=int, default=NORM_GROUPS,
                        help="Jumlah GroupNorm groups (dense backend only)")
    parser.add_argument("--final-activation", choices=["softplus", "relu"], default="softplus",
                        help="Aktivasi output layer")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=GRADIENT_CHECKPOINTING,
                        help="Hemat VRAM dengan recompute activations saat backward pass (~40-60%% reduction)")

    # Optimization
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Jumlah epoch training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help="Akumulasi gradient selama N step. Effective batch = batch_size * N")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate awal (sebelum warmup)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM,
                        help="Max gradient norm untuk clipping")
    parser.add_argument("--amp", action="store_true", default=AMP,
                        help="Aktifkan automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Nonaktifkan AMP")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed untuk reproducibility")

    # Loss
    parser.add_argument("--l1-weight", type=float, default=L1_WEIGHT,
                        help="Weight untuk komponen L1 loss")
    parser.add_argument("--mse-weight", type=float, default=MSE_WEIGHT,
                        help="Weight untuk komponen MSE loss")
    parser.add_argument("--gradient-weight", type=float, default=GRADIENT_WEIGHT,
                        help="Weight Sobolev gradient loss (dense only)")
    parser.add_argument("--conservation-weight", type=float, default=0.1,
                        help="Weight electron conservation penalty")
    try:
        parser.add_argument("--use-density-weighting", action=argparse.BooleanOptionalAction, default=True,
                            help="Inverse-density weighting pada L1 loss. Nonaktifkan jika --use-symlog aktif")
        parser.add_argument("--use-pnll", action=argparse.BooleanOptionalAction, default=False,
                            help="Gunakan Poisson Negative Log-Likelihood (PNLL) sebagai loss utama")
    except AttributeError:
        parser.add_argument("--use-density-weighting", action="store_true", default=True,
                            help="Inverse-density weighting pada L1 loss")
        parser.add_argument("--no-use-density-weighting", action="store_false", dest="use_density_weighting",
                            help="Nonaktifkan density weighting (gunakan ini jika --use-symlog aktif)")
        parser.add_argument("--use-pnll", action="store_true", default=False,
                            help="Gunakan Poisson Negative Log-Likelihood (PNLL) sebagai loss utama")
        parser.add_argument("--no-use-pnll", action="store_false", dest="use_pnll")

    # Checkpointing
    # ── PERUBAHAN: --resume sekarang flag boolean, bukan path ────
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume dari last.pt di experiment dir. "
                             "Diabaikan jika last.pt tidak ditemukan.")

    return parser.parse_args()


# ─── Shared helpers ──────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_keys_and_formulas(h5_path: str, max_samples: int | None = None):
    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle.keys())
        if max_samples is not None:
            keys = keys[:max_samples]
        formulas = []
        for key in keys:
            formula = handle[key].attrs.get("formula", "unknown")
            if isinstance(formula, bytes):
                formula = formula.decode("utf-8")
            formulas.append(str(formula))
    return keys, formulas


def stratified_split(formulas, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    by_formula = defaultdict(list)
    for index, formula in enumerate(formulas):
        by_formula[formula].append(index)

    train_indices = []
    val_indices = []
    for indices in by_formula.values():
        indices = np.asarray(indices)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_split))) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    return sorted(train_indices), sorted(val_indices)


def create_grad_scaler(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    warnings.warn("AMP requested, but GradScaler is unavailable in this PyTorch build. Disabling AMP.")
    return None


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()

    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=True)

    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)

    return nullcontext()


# ─── Dense backend epoch ─────────────────────────────────────────────

def run_epoch_dense(model, loader, criterion, optimizer, device, scaler, grad_clip=None, world_size=1, accumulation_steps=1):
    """One training/validation epoch for the dense backend."""
    training = optimizer is not None
    model.train(training)
    running = defaultdict(float)
    steps = 0
    total_batches = len(loader)

    iterator = _tqdm(loader, leave=False, dynamic_ncols=True)
    for batch_idx, (inputs, targets, mask, _meta) in enumerate(iterator):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        batch_dv = sum(m.dv for m in _meta) / len(_meta)
        batch_n_electrons = torch.tensor([m.n_electrons for m in _meta], device=device, dtype=torch.float32)

        is_last_accumulation_step = (
            (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches
        )

        if training and batch_idx % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=(scaler is not None)):
            preds = model(inputs)

            use_symlog = getattr(loader.dataset, "use_symlog", False)
            if hasattr(loader.dataset, "dataset"):
                use_symlog = getattr(loader.dataset.dataset, "use_symlog", False)

            if use_symlog:
                preds_clamped = torch.clamp(preds, min=-200.0, max=200.0)
                eps = 1e-3
                pred_physical = torch.sign(preds_clamped) * eps * torch.expm1(torch.abs(preds_clamped))
            else:
                pred_physical = None

            loss, stats = criterion(
                preds, targets, mask,
                dv=batch_dv, n_electrons=batch_n_electrons,
                pred_physical=pred_physical
            )
            mae = masked_mae(preds, targets, mask)
            loss_normalized = loss / accumulation_steps

        if not math.isfinite(loss.item()):
            continue

        if training:
            if scaler is not None:
                scaler.scale(loss_normalized).backward()
                if is_last_accumulation_step:
                    scaler.unscale_(optimizer)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss_normalized.backward()
                if is_last_accumulation_step:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        running["loss"] += float(loss.item())
        running["mae"] += float(mae.item())
        for key, value in stats.items():
            running[key] += float(value.item())
        steps += 1
        if dist.is_initialized() and dist.get_rank() == 0:
            iterator.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae.item():.4f}")
        elif not dist.is_initialized():
            iterator.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae.item():.4f}")

    if steps == 0:
        epoch_stats = {"loss": float("inf"), "mae": float("inf")}
    else:
        epoch_stats = {key: value / steps for key, value in running.items()}

    return reduce_metrics(epoch_stats, world_size)


# ─── Sparse backend epoch ────────────────────────────────────────────

def run_epoch_sparse(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None, ME=None, world_size=1):
    """One training/validation epoch for the sparse (MinkowskiEngine) backend."""
    training = optimizer is not None
    model.train(training)
    running = defaultdict(float)
    steps = 0

    iterator = _tqdm(loader, leave=False, dynamic_ncols=True)
    for coords_list, feats_list, targets_list, metas in iterator:
        coordinates = ME.utils.batched_coordinates(coords_list, dtype=torch.int32)
        features = torch.from_numpy(np.concatenate(feats_list, axis=0)).float()
        coordinates = coordinates.to(device)
        features = features.to(device)
        field = ME.TensorField(features=features, coordinates=coordinates, device=device)

        targets = torch.from_numpy(np.concatenate(targets_list, axis=0)).float().to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=(scaler is not None)):
            sparse_input = field.sparse()
            pred_sparse = model(sparse_input)
            pred_field = pred_sparse.slice(field)
            pred_features = pred_field.F
            loss, stats = criterion(pred_features, targets)

        if not torch.isfinite(loss):
            continue

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        running["loss"] += float(loss.item())
        for key, value in stats.items():
            running[key] += float(value.item())
        running["active_points"] += float(sum(meta.num_active for meta in metas)) / max(len(metas), 1)
        steps += 1
        if not dist.is_initialized() or dist.get_rank() == 0:
            iterator.set_postfix(loss=f"{loss.item():.4f}", active=f"{running['active_points'] / steps:.1f}")

    if steps == 0:
        epoch_stats = {"loss": float("inf")}
    else:
        epoch_stats = {key: value / steps for key, value in running.items()}

    return reduce_metrics(epoch_stats, world_size)


# ─── Worker ──────────────────────────────────────────────────────────

def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    set_seed(args.seed + rank)

    if world_size > 1:
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    scaler = create_grad_scaler(device, args.amp)

    experiment_dir = EXPERIMENTS_ROOT / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    if rank == 0:
        ensure_dir(checkpoint_dir)
        ensure_dir(log_dir)

    # Load data
    keys, formulas = collect_keys_and_formulas(args.data, args.max_samples)
    train_indices, val_indices = stratified_split(formulas, args.val_split, args.seed)

    # ── Backend dispatch ──────────────────────────────────────────
    if args.backend == "dense":
        from models.dense.dataset import QM9DensityDataset, dynamic_pad_collate
        from models.dense.loss import PhysicsAwareLoss
        from models.dense.model import SmallUNet3D

        dataset = QM9DensityDataset(
            args.data, keys=keys,
            input_dataset=args.input_dataset, target_dataset=args.target_dataset,
            use_symlog=args.use_symlog,
        )
        collate_fn = dynamic_pad_collate

        model = SmallUNet3D(
            in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
            base_channels=args.base_channels, dropout=args.dropout,
            norm_groups=args.norm_groups,
            final_activation=args.final_activation,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        criterion = PhysicsAwareLoss(
            l1_weight=args.l1_weight,
            conservation_weight=args.conservation_weight,
            gradient_weight=args.gradient_weight,
            use_density_weighting=args.use_density_weighting,
            use_pnll=args.use_pnll,
        )
        if rank == 0 and args.use_pnll and args.final_activation == "relu":
            print("Warning: --use-pnll aktif tetapi --final-activation adalah 'relu'.")
            print("         PNLL membutuhkan output > 0 agar log() stabil. Direkomendasikan 'softplus'.")
        run_epoch_fn = lambda model, loader, criterion, optimizer, device, scaler, grad_clip: \
            run_epoch_dense(
                model, loader, criterion, optimizer, device, scaler, grad_clip,
                world_size=world_size,
                accumulation_steps=args.gradient_accumulation,
            )

    elif args.backend == "sparse":
        from models.sparse.dataset import QM9SparseDataset, sparse_collate_fn
        from models.sparse.loss import SparseDensityLoss
        from models.sparse.model import ME, SparseUNet3D

        dataset = QM9SparseDataset(
            args.data, keys=keys,
            input_dataset=args.input_dataset, target_dataset=args.target_dataset,
        )
        collate_fn = sparse_collate_fn

        model = SparseUNet3D(
            in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
            base_channels=args.base_channels, dropout=args.dropout,
            D=DIMENSION,
        ).to(device)
        criterion = SparseDensityLoss(l1_weight=args.l1_weight, mse_weight=args.mse_weight)
        _ME = ME
        run_epoch_fn = lambda model, loader, criterion, optimizer, device, scaler, grad_clip: \
            run_epoch_sparse(model, loader, criterion, optimizer, device, scaler, grad_clip, ME=_ME, world_size=world_size)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # ── Data loaders ──────────────────────────────────────────────
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn, sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn, sampler=val_sampler,
    )

    # ── Optimizer ─────────────────────────────────────────────────
    raw_model = model.module if world_size > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val = float("inf")

    # ── Resume: auto-resolve ke last.pt di experiment dir ─────────
    if args.resume:
        last_ckpt = checkpoint_dir / LAST_CHECKPOINT_NAME
        if last_ckpt.exists():
            if rank == 0:
                print(f"Resuming from {last_ckpt}")
            start_epoch, best_val = load_checkpoint(last_ckpt, raw_model, optimizer, device=device)
            start_epoch += 1
            if rank == 0:
                print(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")
        else:
            if rank == 0:
                print(f"Warning: --resume specified but {last_ckpt} not found. Starting from scratch.")

    warmup_epochs = 5
    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    for _ in range(start_epoch):
        scheduler.step()

    # ── Logging config ────────────────────────────────────────────
    if rank == 0:
        model_summary = {
            # ── Experiment ────────────────────────────────────────
            "experiment_name": args.experiment_name,
            "resumed_from_epoch": start_epoch,

            # ── Runtime ───────────────────────────────────────────
            "backend": args.backend,
            "device": str(device),
            "world_size": world_size,

            # ── Data ──────────────────────────────────────────────
            "data": args.data,
            "input_dataset": args.input_dataset,
            "target_dataset": args.target_dataset,
            "val_split": args.val_split,
            "max_samples": args.max_samples,
            "num_workers": args.num_workers,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "use_symlog": args.use_symlog,

            # ── Model ─────────────────────────────────────────────
            "base_channels": args.base_channels,
            "dropout": args.dropout,
            "norm_groups": args.norm_groups,
            "final_activation": args.final_activation,
            "parameters": int(sum(p.numel() for p in raw_model.parameters() if p.requires_grad)),

            # ── Optimization ──────────────────────────────────────
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "amp": args.amp,
            "seed": args.seed,

            # ── Efficiency ────────────────────────────────────────
            "gradient_checkpointing": args.gradient_checkpointing,

            # ── Loss ──────────────────────────────────────────────
            "l1_weight": args.l1_weight,
            "mse_weight": args.mse_weight,
            "gradient_weight": args.gradient_weight,
            "conservation_weight": args.conservation_weight,
            "use_density_weighting": args.use_density_weighting,
            "use_pnll": args.use_pnll,
        }
        write_json(log_dir / "run_config.json", model_summary)

    # ── Training loop ─────────────────────────────────────────────
    history_path = log_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_metrics = run_epoch_fn(model, train_loader, criterion, optimizer, device, scaler, args.grad_clip)

        with torch.no_grad():
            val_metrics = run_epoch_fn(model, val_loader, criterion, None, device, None, None)

        elapsed = time.time() - epoch_start
        scheduler.step()

        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "time_s": elapsed,
                "lr": scheduler.get_last_lr()[0],
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"time={elapsed:.1f}s"
            )

            save_checkpoint(checkpoint_dir / LAST_CHECKPOINT_NAME, raw_model, optimizer, epoch, best_val)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(checkpoint_dir / BEST_CHECKPOINT_NAME, raw_model, optimizer, epoch, best_val)

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt", raw_model, optimizer, epoch, best_val)

    if world_size > 1:
        cleanup_ddp()


def main() -> None:
    args = parse_args()

    if args.backend == "sparse" and args.gpus == -1:
        print("Warning: Sparse backend (MinkowskiEngine) does not support DDP. Falling back to GPU 0.")
        args.gpus = 0

    if args.gpus == -1:
        world_size = torch.cuda.device_count()
        if world_size < 1:
            print("No GPUs found. Falling back to CPU.")
            main_worker(0, 1, args)
        else:
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
