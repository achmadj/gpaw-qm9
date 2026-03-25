#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.config import (
    BASE_CHANNELS,
    DATA_PATH,
    DROPOUT,
    EXPERIMENTS_ROOT,
    INPUT_DATASET,
    NORM_GROUPS,
    TARGET_DATASET,
)
from models.dense.dataset import QM9DensityDataset, dynamic_pad_collate
from models.dense.model import SmallUNet3D
from models.utils import ensure_dir, load_checkpoint


def parse_args():
    default_experiment = EXPERIMENTS_ROOT / "default"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument(
        "--checkpoint", default=str(default_experiment / "checkpoints" / "best.pt")
    )
    parser.add_argument("--output-dir", default=str(default_experiment / "eval"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--norm-groups", type=int, default=NORM_GROUPS)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)

    dataset = QM9DensityDataset(args.data)
    if args.limit:
        dataset.keys = dataset.keys[: args.limit]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynamic_pad_collate,
    )

    model = SmallUNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        dropout=args.dropout,
        norm_groups=args.norm_groups,
    ).to(device)

    print(f"Loading checkpoint {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    model.eval()

    results = []

    print("Evaluating dataset...")
    with torch.no_grad():
        for inputs, targets, masks, metas in tqdm(loader):
            inputs = inputs.to(device)
            preds = model(inputs)

            # Move to CPU for analysis
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            masks = masks.cpu().numpy()

            for i, meta in enumerate(metas):
                pred = preds[i, 0]
                target = targets[i, 0]
                mask = masks[i, 0]

                # Unpad using original shape
                ox, oy, oz = meta.original_shape
                pred = pred[:ox, :oy, :oz]
                target = target[:ox, :oy, :oz]

                true_flat = target.reshape(-1)
                pred_flat = pred.reshape(-1)

                true_max = float(true_flat.max())
                pred_max = float(pred_flat.max())

                non_zero = true_flat[true_flat > 1e-2]
                median_val = float(np.median(non_zero)) if non_zero.size > 0 else 0.0
                ratio = true_max / median_val if median_val > 0 else 0.0

                mae = float(np.mean(np.abs(pred_flat - true_flat)))

                results.append(
                    {
                        "key": meta.key,
                        "formula": meta.formula,
                        "true_max": true_max,
                        "pred_max": pred_max,
                        "median_nonzero": median_val,
                        "ratio_max_to_median": ratio,
                        "mae": mae,
                    }
                )

    # Analysis
    ratios = [r["ratio_max_to_median"] for r in results]
    true_maxs = [r["true_max"] for r in results]

    stats = {
        "total_samples": len(results),
        "avg_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "max_ratio": float(np.max(ratios)),
        "min_ratio": float(np.min(ratios)),
        "percent_extreme_ratio_gt_100": float(np.mean(np.array(ratios) > 100) * 100),
        "avg_true_max": float(np.mean(true_maxs)),
    }

    with open(output_dir / "distribution_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Distribution Stats ---")
    print(json.dumps(stats, indent=2))

    # Plot histogram of ratios
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Ratio (Max Density / Median Non-Zero Density)")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_histogram.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved stats and histogram to {output_dir}")


if __name__ == "__main__":
    main()
