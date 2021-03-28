#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4 
#SBATCH --output=supervised.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet
python train_supervised.py MLP_sound_64 
