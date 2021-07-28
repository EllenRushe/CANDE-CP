#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi

# Runs from 1 to $NUM_ITERS inclusive
for i in $(seq "$NUM_ITERS")
do 
	val_json="logs/ae_sound_film_embed_64/AE_sound_FiLM_embed_64__$i.json"
	python  eval_embed_oracle.py  $val_json $i  
done

