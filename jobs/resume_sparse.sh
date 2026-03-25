#!/bin/bash
#SBATCH --job-name=3d_unet_sparse_resume
#SBATCH --output=/clusterfs/students/achmadjae/gpaw-qm9/models/experiments/sparse_minkowski_v1/logs/slurm_%j.out
#SBATCH --error=/clusterfs/students/achmadjae/gpaw-qm9/models/experiments/sparse_minkowski_v1/logs/slurm_%j.err
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi08
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

ROOT=/clusterfs/students/achmadjae/gpaw-qm9
cd "${ROOT}"
source /clusterfs/students/achmadjae/miniconda3/bin/activate cuda118

mkdir -p models/experiments/sparse_minkowski_v1/logs

echo "Resuming Sparse 3D UNet (MinkowskiEngine) training..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

python -u models/train.py \
    --backend sparse \
    --experiment-name sparse_minkowski_v1 \
    --data "${ROOT}/dataset/gpaw_qm9_all.h5" \
    --resume "${ROOT}/models/experiments/sparse_minkowski_v1/checkpoints/best.pt" \
    --epochs 300 \
    --batch-size 64 \
    --lr 3e-4 \
    --base-channels 16 \
    --dropout 0.05 \
    --val-split 0.10 \
    --num-workers 8 \
    --no-amp

echo "Training completed at $(date)"
