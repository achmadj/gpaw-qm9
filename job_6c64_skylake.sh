#!/bin/bash
#SBATCH --job-name=gpaw_6c64
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/gpaw_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/gpaw_%j.err
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --partition=6C64GB_skylake

set -euo pipefail
source /home/achmadjae/miniconda3/etc/profile.d/conda.sh
conda activate gpaw
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR="/home/achmadjae/gpaw-qm9"
INPUT_DB="$PROJECT_DIR/dataset/raw/qm9_smallest_20k.h5"
NUM_SHARDS=32
OFFSET=24

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/dataset/shards"
cd "$PROJECT_DIR"

srun --ntasks=6 bash -c '
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
echo "Done job 6C64GB_skylake"
