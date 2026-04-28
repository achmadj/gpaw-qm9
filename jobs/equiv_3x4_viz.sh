#!/bin/bash
#SBATCH --job-name=equiv_3x4_viz
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/equiv_3x4_viz_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/equiv_3x4_viz_%j.err

set -e

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"

export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

python /home/achmadjae/gpaw-qm9/scripts/plot_equivariant_rotation_3x4_grid.py

echo "=== Job finished at $(date) ==="
