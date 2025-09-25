#!/usr/bin/env bash

#SBATCH --job-name=olmo-20M-CPT
#SBATCH --output=logs/cpt/%j.out
#SBATCH --error=logs/cpt/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo
for val in 32; do
  for lr in 1e-3 2e-4 4e-5; do
    echo "Running OLMo-20M-${val}B-CPT with lr ${lr}"
    torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py new_configs/cpt/lr_${lr}/OLMo-20M-${val}B-CPT.yaml --save_overwrite
  done
done