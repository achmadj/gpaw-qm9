#!/bin/bash
#SBATCH --job-name=gpaw_shard1
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/gpaw_p_shard1_fix_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/gpaw_p_shard1_fix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=dual_4090
#SBATCH --nodelist=node19

set -eo pipefail
source /home/achmadjae/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR="/home/achmadjae/gpaw-qm9"
INPUT_DB="$PROJECT_DIR/dataset/raw/qm9_full.h5"
SHARD_OUT="$PROJECT_DIR/dataset/shards/gpaw_pseudo_paw_pbe_full_shard_1.h5"

echo "Running shard 1 on $(hostname)"

srun --ntasks=1 python $PROJECT_DIR/src/run_gpaw_pseudo_density.py \
  --db_path "$INPUT_DB" --out_path "$SHARD_OUT" \
  --n_mols 0 --selection all \
  --shard_index 1 --num_shards 32 \
  --padding_bohr 4.0 --h 0.2 --mode fd --xc PBE \
  --resume \
  > "$PROJECT_DIR/logs/pseudo_paw_pbe_full_shard_1.log" 2>&1

echo "Done"
