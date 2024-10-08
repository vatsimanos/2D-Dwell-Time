#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --export=NONE
##SBATCH -C a100_80

unset SLURM_EXPORT_ENV

module load python/tensorflow-2.7.0py3.9

srun python Deep_Dwell_time_regression.py