#!/bin/bash
#SBATCH --job-name=equiv_v2_viz
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/equiv_v2_3x4_viz_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/equiv_v2_3x4_viz_%j.err

set -e

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

echo "Python: $(which python)"

python /home/achmadjae/gpaw-qm9/scripts/plot_equivariant_v2_3x4_grid.py

echo "=== Job finished at $(date) ==="
