#!/bin/bash
#SBATCH --job-name=finetune_%j
#SBATCH --output=logs/finetune/%j.out
#SBATCH --error=logs/finetune/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Specfic to user environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310

# Run evaluation
python /home/{USER}/catastrophic-forgetting/finetune/finetune_starcoder.py --model_path {MODEL_PATH}  --output_dir {OUTPUT} --learning_rate {LR}