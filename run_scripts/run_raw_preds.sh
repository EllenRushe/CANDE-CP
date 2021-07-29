#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --output=aes_cond_embed_pred_raw.log
#SBATCH -t 12:00:00
cd $SLURM_SUBMIT_DIR
source  activate pt1.8
srun -N 1 ./run_scripts/ae_sound_film_embed_64_raw_preds.sh 2
