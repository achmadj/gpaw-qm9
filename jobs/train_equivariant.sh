#!/bin/bash
#SBATCH --job-name=train_equiv
#SBATCH --partition=qdisk
#SBATCH --nodelist=quasi08
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_equiv_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnist-torch

cd /clusterfs/students/achmadjae/gpaw-qm9

python models/equivariant/train.py \
    --data dataset/qm9_1000_phase_a.h5 \
    --experiment-name equivariant_v1 \
    --epochs 100 \
    --batch-size 8 \
    --lr 3e-4 \
    --base-channels 32 \
    --max-freq 2 \
    --last-activation softplus \
    --gpu 0
