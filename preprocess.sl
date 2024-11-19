#!/bin/bash
#SBATCH --job-name=mswep_mon
#SBATCH --partition=short
#SBATCH --output=./_logs/%x_%j.out
#SBATCH --error=./_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=10gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eastinev@bc.edu

module load miniconda

conda activate zhang_repr

for i in {2004..2020..4}; do
	for j in {0..11}; do
		srun --nodes=1 --ntasks=1 --cpus-per-task --exclusive python monthly_mswep.py $i $j &
	done
done
wait
