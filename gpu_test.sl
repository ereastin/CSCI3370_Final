#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eastinev@bc.edu

module load miniconda

conda activate zhang_repr

python gpu_test.py
