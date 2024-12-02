#!/bin/bash
#SBATCH --job-name=era_mon
#SBATCH --partition=short
#SBATCH --output=./_logs/%x_%j.out
#SBATCH --error=./_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=10gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eastinev@bc.edu
module load miniconda

conda activate zhang_repr

for i in {2001..2020}; do
	for j in {0..11}; do
		python preprocess.py $i $j
	done
done
