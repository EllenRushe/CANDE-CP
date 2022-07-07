# Deep Contextual Novelty Detection with Context Prediction.
Implementation of *"Deep Contextual Novelty Detection with Context Prediction"*.
## Requirements installation
**pip**
```sh
$ pip -r install requirements.txt
```
## __Data__
**Download data**
- The MIMII open source version of the MIMII dataset was used for experiments. Data can be downloaded from [zenodo](https://zenodo.org/record/3384388#.XuEE2GpKhhE)
- Feature extraction based on code from original [MIMII baseline repository]( https://github.com/MIMII-hitachi/mimii_baseline/). Any modifications made are clearly indicated in code. 

**Run feature extraction**
Fill out ```baseline.yaml```, then run:
```sh
python data_utils/baseline_mod_add.py 
```
**Set up data for contextual novelty detection** 
- ```source_dir``` (str): Directory here pickle data is written
- ```target_dir``` (str): Where NumPY dataset is to be written to. 
```sh
python python data_utils/write_all_data.py --source_dir "..." --target_dir "..."
```

## __Quick training/validation/evaluation__
**To quickly run train/validation/evaluation, run bash scripts in "run_scripts" directory. These are available for all models.**

```sh
$ num_iterations=3 # Number of random initializations to run training and evaluation.
$ ./run_scripts/ae_sound_film_one_hot.sh $num_iterations
```

## __Stand-alone scripts__
Use evaluation section to load all pretrained checkpoints and evaluate models.
## Autoencoders
### Training 
```train.py```
**args:**
- ```model_name``` (str): Model name as defined in ```hparams.yaml```.
- ```context``` (int): If no context is being defined leave as empty string. 
-``` iteration``` (int): Defines iteration of runs for different random initializations (1 if only using one random initialization).

```sh
$ python train.py AE_sound_all "" 1
```

### Evaluation
```eval.py```
```eval_one_hot.py```
```eval_embed.py```

All evaluation files use the following two arguments. 
- ```iteration``` (int): If running multiple initialisations of training (keep as 1 if only a single iteration  was run)
- ```val_json``` (str): define log file for trained model to be evaluated (this will load correct model etc.)
```sh
$ iteration=1
$ val_json="logs/ae_sound_film_one_hot/AE_sound_FiLM_one_hot__$iteration.json"
```




#### Unconditioned
```sh
$ python eval.py $val_json $iteration
```
#### Conditioned

**args:**
- ```window_size``` (int): Number of past examples to use.
- ```context_ckpt_name``` (str): Name of pretrained context classifer used to predict context. 

**One-hot conditioned**
```sh
$ python eval_one_hot.py  $val_json $iteration  --window_size 50 --context_ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt"
```
**Embedding conditioned**
```sh
$ python  eval_embed.py  $val_json $iteration --window_size 50 --context_ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt"
```

## Context Predictor
### Training
**args:**
- ```model_name``` (str): Model name as defined in hparams.yaml.

```sh
$ python train_supervised.py MLP_sound_64 
```
### Generate Embeddings
Fill ```embedding_params.yaml``` and run
```sh
$ best_ckpt = "..."
$ python generate_embedding.py --model_name MLP_sound_64 --ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt" 
```
## Pre-trained Models
**All pre-trained models can be found in at [this google drive link](https://drive.google.com/drive/folders/1jLKILWg3IAZlQtKm8gQIu1lNBxheibFP). The checkpoint with the best validation performance are within the training logs. The ```run_scripts``` load this checkpoint automatically.**


**Note for the data train/val/test configuration:**  This is achieved by adding 50% of the normal files for each machine ID from the dataset to the test data, leaving the other half for training. This means that overall 50% of data was used for testing while the remaining 50% was used for the training. A portion of normal examples from the test set was used for validation. Please not that these normal validation examples were also used in the evaluations for all models. This portion’s size was equal to 10% of the number of files in the training data (which are all normal files).
