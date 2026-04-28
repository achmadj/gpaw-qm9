#!/bin/bash
#SBATCH --job-name=v2_progress_report
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/v2_progress_report_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/v2_progress_report_%j.err

set -e

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"

export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

echo ""
echo "--- 1. Loss Curve ---"
python /home/achmadjae/gpaw-qm9/scripts/run_v2_loss_curve.py

echo ""
echo "--- 2. Rotation R2 (10 mol) ---"
python /home/achmadjae/gpaw-qm9/scripts/run_v2_rotation_r2.py

echo ""
echo "--- 3. 3x4 Visualization ---"
python /home/achmadjae/gpaw-qm9/scripts/plot_equivariant_v2_3x4_grid.py

echo ""
echo "--- 4. V2 vs Dense Comparison ---"
python /home/achmadjae/gpaw-qm9/scripts/run_v2_vs_dense_val.py

echo ""
echo "--- 5. Prediction Metrics (5 mol) ---"
python /home/achmadjae/gpaw-qm9/scripts/run_v2_pred_metrics.py

echo ""
echo "=== Job finished at $(date) ==="
