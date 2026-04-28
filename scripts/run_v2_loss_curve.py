#!/usr/bin/env python3
"""Plot training loss curve from equivariant_v2 history."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HISTORY_PATH = "/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/logs/history.jsonl"
OUT_DIR = Path("/home/achmadjae/3d-unet-qm9/figures")

with open(HISTORY_PATH, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

epochs = [d["epoch"] for d in data]
train_loss = [d["train_loss"] for d in data]
val_loss = [d["val_loss"] for d in data]

best_idx = int(np.argmin(val_loss))
best_epoch = epochs[best_idx]
best_val = val_loss[best_idx]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss (MAE)", linewidth=2)
plt.plot(epochs, val_loss, label="Validation Loss (MAE)", linewidth=2)
plt.scatter([best_epoch], [best_val], color="red", zorder=5, s=50)
plt.annotate(f"Best: {best_val:.5f}\nEpoch {best_epoch}",
             xy=(best_epoch, best_val), xytext=(best_epoch+10, best_val+0.005),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="red"))
plt.yscale("log")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Absolute Error (log scale)", fontsize=12)
plt.title("Training and Validation Loss: SO(3)-Equivariant 3D U-Net", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / "loss_vs_epoch.png")
plt.savefig(OUT_DIR / "loss_vs_epoch.pdf")
plt.close()

print(f"Best validation loss: {best_val:.5f} at epoch {best_epoch}")
print(f"Saved: {OUT_DIR / 'loss_vs_epoch.png'}")
