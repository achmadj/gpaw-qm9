#!/bin/bash
#SBATCH --job-name=gpaw_8c16
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/gpaw_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/gpaw_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=48:00:00
#SBATCH --partition=8C16GB_sapphire

set -euo pipefail
source /home/achmadjae/miniconda3/etc/profile.d/conda.sh
conda activate gpaw
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR="/home/achmadjae/gpaw-qm9"
INPUT_DB="$PROJECT_DIR/dataset/raw/qm9_smallest_20k.h5"
NUM_SHARDS=32
OFFSET=30

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/dataset/shards"
cd "$PROJECT_DIR"

srun --ntasks=2 bash -c '
  IDX=$(('"$OFFSET"' + SLURM_PROCID))
  SHARD_OUT="'"$PROJECT_DIR"'/dataset/shards/gpaw_shard_${IDX}.h5"
  echo "Starting shard=$IDX on $(hostname)"
  python '"$PROJECT_DIR"'/src/run_gpaw_from_h5.py \
    --db_path "'"$INPUT_DB"'" --out_path "$SHARD_OUT" \
    --n_mols 0 --selection all \
    --shard_index "$IDX" --num_shards '"$NUM_SHARDS"' \
    --padding_bohr 4.0 --h 0.2 --resume \
    > "'"$PROJECT_DIR"'/logs/shard_${IDX}.log" 2>&1
  echo "Done shard=$IDX on $(hostname)"
'
echo "Done job 8C16GB_sapphire"
