#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi
best_ckpt_64=$(cat 'logs/mlp_sound_64/MLP_sound_64.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
# Runs from 1 to $NUM_ITERS inclusive
for i in $(seq 3 "$NUM_ITERS")
do 
	python train.py AE_sound_FiLM_one_hot "" $i
	val_json="logs/ae_sound_film_one_hot/AE_sound_FiLM_one_hot__$i.json"
 	python eval_one_hot.py  $val_json $i  --window_size 50  --decay_factor 0.1 --context_ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt_64"
    python eval_one_hot_acc.py  $val_json $i  --window_size 50  --decay_factor 0.1 --context_ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt_64"

done
