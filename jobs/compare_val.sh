#!/bin/bash
#SBATCH --job-name=compare_val
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/compare_val_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/compare_val_%j.err

set -e
echo "=== Job started at $(date) ==="
export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9
python /home/achmadjae/gpaw-qm9/scripts/compare_equiv_vs_dense_rotation_val.py
echo "=== Job finished at $(date) ==="
