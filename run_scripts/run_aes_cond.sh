#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 24:00:00 
#SBATCH --output=aes_cond_embed.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet

./run_scripts/ae_sound_film_embed_64.sh 3 


