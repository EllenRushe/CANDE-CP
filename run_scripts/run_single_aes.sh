#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --output=aes.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet

./run_scripts/aes_sound.sh 3
