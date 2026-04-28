#!/bin/bash
#SBATCH --job-name=gen_vion
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi09
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vion_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpaw  

python src/generate_pre_scf_vion.py \
    --db_path /home/achmadjae/gpaw-qm9/dataset/gpaw_qm9_merged.h5 \
    --out_path dataset/vion_qm9_20k.h5 \
    --padding_bohr 4.0 --h 0.2 --xc LDA
