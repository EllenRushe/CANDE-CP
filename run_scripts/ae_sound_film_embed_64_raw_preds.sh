#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi

best_ckpt_64=$(cat 'logs/mlp_sound_64/MLP_sound_64.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
# Runs from 1 to $NUM_ITERS inclusive - edited to 2 because process was accdidentally interrupted
for i in $(seq "$NUM_ITERS")
do 
	val_json="logs/ae_sound_film_embed_64/AE_sound_FiLM_embed_64__$i.json"
	python  eval_embed_raw_preds.py  $val_json $i  --context_ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt_64"
done

