#!/bin/bash
#SBATCH --job-name=gen_phaseA
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi09
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_phaseA_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpaw

cd /clusterfs/students/achmadjae/gpaw-qm9

python src/generate_dataset_phase_a.py \
    --sdf data/raw/gdb9.sdf \
    --out dataset/qm9_1000_phase_a.h5 \
    --n_mols 1000 \
    --h 0.2 \
    --padding_bohr 4.0 \
    --xc LDA
