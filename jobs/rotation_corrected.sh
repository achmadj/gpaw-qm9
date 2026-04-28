#!/bin/bash
#SBATCH --job-name=rot_correct
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/rot_correct_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/rot_correct_%j.err
set -e
export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9
python /home/achmadjae/gpaw-qm9/scripts/rotation_corrected.py
echo "=== Job finished ==="
