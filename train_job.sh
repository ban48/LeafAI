#!/bin/bash
#SBATCH --job-name=leafai_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00


module load anaconda3/2023.09-0-none-none
module load cuda/12.6.3
module load cudnn/9.8.0.87-12-none-none-cuda-12.6.3

source activate leafai_env

cd /homes/ztesta/leafai_project

mkdir -p logs

python main.py