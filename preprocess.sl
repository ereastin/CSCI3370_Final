#!/bin/bash
#SBATCH --job-name=pp_01_18
#SBATCH --partition=medium
#SBATCH --output=./test_logs/%x_%j.out
#SBATCH --error=./test_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eastinev@bc.edu

module load miniconda/3

conda activate zhang_repr

for i in {2001..2018}; do
	for j in {0..11}; do
		python monthly_mswep.py $i $j
	done
done
