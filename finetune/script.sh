#!/bin/bash
#SBATCH --job-name=finetune_%j
#SBATCH --output=/logs/finetune/%j.err
#SBATCH --error=/logs/finetune/%j.err
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
python /home/{ANDREW_ID}/catastrophic-forgetting/finetune/finetune_starcoder.py --model_path "{MODEL PATH}"  --output_dir "{FT OUTPUT PATH}" --learning_rate {LR}