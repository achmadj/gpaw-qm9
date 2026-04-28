#!/bin/bash
#SBATCH --job-name=gpaw_p_4c16s
#SBATCH --output=/home/achmadjae/gpaw-qm9/logs/gpaw_p_%j.out
#SBATCH --error=/home/achmadjae/gpaw-qm9/logs/gpaw_p_%j.err
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=48:00:00
#SBATCH --partition=4C16GB_skylake

set -eo pipefail
source /home/achmadjae/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR="/home/achmadjae/gpaw-qm9"
INPUT_DB="${INPUT_DB:-$PROJECT_DIR/dataset/raw/qm9_smallest_20k.h5}"
NUM_SHARDS="${NUM_SHARDS:-32}"
OFFSET="${OFFSET:-4}"
SHARD_PREFIX="${SHARD_PREFIX:-gpaw_pseudo_shard}"
LOG_PREFIX="${LOG_PREFIX:-pseudo_shard}"
MODE="${MODE:-fd}"
XC="${XC:-LDA}"
SETUPS="${SETUPS:-}"
PADDING_BOHR="${PADDING_BOHR:-4.0}"
H="${H:-0.2}"
SELECTION="${SELECTION:-all}"
N_MOLS="${N_MOLS:-0}"
RESUME="${RESUME:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_SHARDS="${SKIP_SHARDS:-}"

if [ "$RESUME" = "1" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

if [ -n "$SETUPS" ]; then
  SETUPS_FLAG="--setups $SETUPS"
else
  SETUPS_FLAG=""
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/dataset/shards"
cd "$PROJECT_DIR"

echo "Pseudo generation config: MODE=$MODE XC=$XC SETUPS=${SETUPS:-default} NUM_SHARDS=$NUM_SHARDS OFFSET=$OFFSET PREFIX=$SHARD_PREFIX SKIP_SHARDS='${SKIP_SHARDS}'"

srun --ntasks=20 bash -c '
  IDX=$(('"$OFFSET"' + SLURM_PROCID))
  if [[ " '"$SKIP_SHARDS"' " == *" ${IDX} "* ]]; then
    echo "Skipping shard=$IDX due to SKIP_SHARDS"
    exit 0
  fi
  SHARD_OUT="'"$PROJECT_DIR"'/dataset/shards/'"$SHARD_PREFIX"'_${IDX}.h5"
  echo "Starting pseudo shard=$IDX on $(hostname)"
  python '"$PROJECT_DIR"'/src/run_gpaw_pseudo_density.py \
    --db_path "'"$INPUT_DB"'" --out_path "$SHARD_OUT" \
    --n_mols "'"$N_MOLS"'" --selection "'"$SELECTION"'" \
    --shard_index "$IDX" --num_shards '"$NUM_SHARDS"' \
    --padding_bohr "'"$PADDING_BOHR"'" --h "'"$H"'" --mode "'"$MODE"'" --xc "'"$XC"'" \
    '"$SETUPS_FLAG"' '"$RESUME_FLAG"' '"$EXTRA_ARGS"' \
    > "'"$PROJECT_DIR"'/logs/'"$LOG_PREFIX"'_${IDX}.log" 2>&1
  echo "Done pseudo shard=$IDX on $(hostname)"
'
echo "Done job 4C16GB_skylake (pseudo)"
