#!/usr/bin/env bash

#SBATCH --job-name=test
#SBATCH --output=logs/ft_master_%j.out
#SBATCH --error=logs/ft_master_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# create logs dir if it doesn't exist (safe on compute nodes)
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo
# Optional: activate conda environment
# If you use conda, make sure to initialize shell integration on the login node
# and that the environment exists on the compute node image.


# Use torchrun to launch across the 4 local GPUs
# --nproc_per_node should equal the number of GPUs reserved on the node (4)
torchrun --nproc_per_node=4 scripts/train.py configs/tiny/OLMo-20M-test.yaml --save_overwrite