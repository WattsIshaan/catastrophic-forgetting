#!/usr/bin/env bash

#SBATCH --job-name=OLMo-20M-64B
#SBATCH --output=logs/pretrain/%j.out
#SBATCH --error=logs/pretrain/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo

torchrun --nproc_per_node=4 scripts/train.py new_configs/pretrain/OLMo-20M-64B-PreTrain.yaml --save_overwrite