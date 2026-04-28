#!/bin/bash
#SBATCH --job-name=dense_phase_a
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/dense_phase_a_train_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/dense_phase_a_train_%j.err

set -e

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"

export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

python /home/achmadjae/gpaw-qm9/scripts/train_dense_phase_a.py \
    --data /home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5 \
    --experiment-name dense_phase_a_v1 \
    --epochs 100 \
    --batch-size 8 \
    --lr 3e-4 \
    --base-channels 32 \
    --last-activation softplus \
    --val-split 0.10 \
    --seed 42 \
    --amp \
    --gpu 0 \
    --num-workers 2

echo "=== Job finished at $(date) ==="
