#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi

# Runs from 1 to $NUM_ITERS inclusive
for i in $(seq "$NUM_ITERS")
do
	# Runs from 1 to $NUM_ITERS inclusive
	for context in {0..15}
	do 
		#python train.py AE_sound $context $i
		val_json="logs/ae_sound/AE_sound_"$context"_"$i".json"
		python eval.py $val_json $i
	done

done
