import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Plot Training Loss Curve
with open('models/experiments/equivariant_v1/logs/history.jsonl', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

epochs = [d['epoch'] for d in data]
train_loss = [d['train_loss'] for d in data]
val_loss = [d['val_loss'] for d in data]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss (MAE)', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss (MAE)', linewidth=2)
plt.yscale('log')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Absolute Error (log scale)', fontsize=12)
plt.title('Training and Validation Loss: SO(3)-Equivariant 3D U-Net', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('loss_curve.pdf')
plt.close()

print("Generated loss_curve.pdf")
