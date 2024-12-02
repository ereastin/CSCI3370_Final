#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --output=./_logs/%x_%j.out
#SBATCH --error=./_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evan.eastin@bc.edu

module load miniconda

conda activate zhang_repr

python train.py cnn train 50
