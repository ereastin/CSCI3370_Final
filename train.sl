#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --output=./_logs/%x_%j.out
#SBATCH --error=./_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evan.eastin@bc.edu

module load miniconda

conda activate zhang_repr

echo | which python

python train.py cnn train
