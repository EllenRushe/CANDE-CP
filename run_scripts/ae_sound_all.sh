#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi
# Runs from 1 to $NUM_ITERS inclusive
for i in $(seq "$NUM_ITERS")
do 
	#python train.py AE_sound_all "" $i
	val_json="logs/ae_sound_all/AE_sound_all__$i.json"
	python eval.py $val_json $i
done
