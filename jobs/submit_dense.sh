#!/bin/bash
#SBATCH --job-name=3d_unet_dense
#SBATCH --output=/clusterfs/students/achmadjae/gpaw-qm9/models/experiments/dense_baseline_v1/logs/slurm_%j.out
#SBATCH --error=/clusterfs/students/achmadjae/gpaw-qm9/models/experiments/dense_baseline_v1/logs/slurm_%j.err
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi08
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT=/clusterfs/students/achmadjae/gpaw-qm9
cd "${ROOT}"
source /clusterfs/students/achmadjae/miniconda3/bin/activate research

mkdir -p models/experiments/dense_baseline_v1/logs

echo "Starting Dense 3D UNet training..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

python -u models/train.py \
    --backend dense \
    --experiment-name dense_baseline_v2 \
    --data "${ROOT}/dataset/gpaw_qm9_20k.h5" \
    --epochs 80 \
    --batch-size 32 \
    --lr 3e-4 \
    --base-channels 16 \
    --dropout 0.05 \
    --val-split 0.10 \
    --num-workers 8

echo "Training completed at $(date)"
