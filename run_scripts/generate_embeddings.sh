#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 8 
#SBATCH --output=generate_embeddings.log
cd $SLURM_SUBMIT_DIR
source  activate contextNet
best_ckpt_64=$(cat 'logs/mlp_sound_64/MLP_sound_64.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
srun -N 1 -n 1 python generate_embedding.py --model_name MLP_sound_64 --ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt_64" 

