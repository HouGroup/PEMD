#!/bin/bash
#SBATCH -J pemd
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p standard

conda activate foyer 
module load GROMACS 
module load Gaussian

python md.py 

