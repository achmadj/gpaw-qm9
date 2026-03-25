#!/bin/bash
#SBATCH --job-name=gpaw_qm9_sharded
#SBATCH --output=/clusterfs/students/achmadjae/gpaw-qm9/gpaw_logs/slurm_%x_%j.out
#SBATCH --error=/clusterfs/students/achmadjae/gpaw-qm9/gpaw_logs/slurm_%x_%j.err
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi09
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

ROOT=/clusterfs/students/achmadjae/gpaw-qm9
RUN_ID=qm9_test_sharded_$(date +%Y%m%d_%H%M%S)
RUN_DIR=${ROOT}/gpaw_logs/runs/${RUN_ID}

cd "${ROOT}"
source /clusterfs/students/achmadjae/miniconda3/bin/activate gpaw
PYTHON=/clusterfs/students/achmadjae/miniconda3/envs/gpaw/bin/python
NUM_WORKERS=2

mkdir -p "${RUN_DIR}/shards" "${RUN_DIR}/merged" "${RUN_DIR}/logs"

for worker in $(seq 0 $((NUM_WORKERS - 1))); do
    ${PYTHON} -u ${ROOT}/src/run_gpaw_from_h5.py \
        --seed 42 \
        --n_mols 2 \
        --selection random \
        --resume \
        --shard_index ${worker} \
        --num_shards ${NUM_WORKERS} \
        --db_path ${ROOT}/data/raw/qm9_smallest_20k.h5 \
        --out_path ${RUN_DIR}/shards/gpaw_qm9_shard_${worker}.h5 \
        --log_dir ${RUN_DIR}/logs/worker_${worker} &
done

wait

${PYTHON} -u ${ROOT}/src/merge_h5_shards.py \
    --input_glob "${RUN_DIR}/shards/gpaw_qm9_shard_*.h5" \
    --out_path ${RUN_DIR}/merged/gpaw_qm9_merged.h5
