import os
import time
import json
import glob
import argparse
import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import models 
from data_utils import Datasets
from data_utils.params import Params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "val_json", type=str, help="Directory of validation json file which indictates the best epoch.")
    parser.add_argument(
            "iter", type=int, default=1, help="Evaluation iteration.")
    parser.add_argument(
            "--decay_factor", type=float, help="Decay factor.")
    parser.add_argument(
            "--window_size", type=int, default=1, help="Window size for evaluation.")
    parser.add_argument(
            "--context_ckpt_name", type=str, help="checkpoint name for contextual model.")
    args = parser.parse_args()

    with open(args.val_json) as json_file:  
        model_params  = json.load(json_file)  
    params = Params("hparams.yaml", model_params["model"])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    log_dir = os.path.join(params.log_dir, "testing_logs")
    results_dir = os.path.join("results", model_params["model"], model_params["context"]) 
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    context_model_module = __import__('.'.join(['models', params.context_model_name]),  fromlist=['object'])
    Dataset = getattr(Datasets, params.dataset_class)

    test_data_dir = os.path.join(params.data_dir, "test")
    test_files = glob.glob(os.path.join(test_data_dir, "eval_files_*_*_*dB.npy"))
    test_labels = glob.glob(os.path.join(test_data_dir, "eval_labels_*_*_*dB.npy"))
    # checkpoint belonging to the current evaluation iteration
    iter_checkpoint = os.path.join(
        params.checkpoint_dir, 
        "checkpoint_{}_{}_iter_{}_epoch_{}".format(
            model_params["model"], 
            model_params["context"],  
            args.iter, 
            model_params["best_val_epoch"]
            )
    )
    with open(os.path.join(params.data_dir, 'machine_ids_map')) as f:
        machine_ids_map = json.load(f)
    test_dirs = glob.glob(os.path.join(test_data_dir, "*/*/*"))    

    with open(os.path.join(params.data_dir, 'machine_ids_map')) as f:
        machine_ids_map = json.load(f)
    
    results = {}
    results_log = []
    window_size = args.window_size

    context_model = context_model_module.net()
    context_model = nn.DataParallel(context_model).to(device)
    checkpoint_name = os.path.join(params.context_checkpoint_dir, args.context_ckpt_name)
    model = model_module.net()
    model = nn.DataParallel(model).to(device)
    test = model_module.test
    window_preds = []
    
    for test_dir in test_dirs:
        machine_data = test_dir.split('/')[-3:] 
        dB, machine_name, machine_id = machine_data
        # Check if this is a context specific model. 
        if model_params["context"]:
            # Skip these files if they aren't in the context being evaluated. 
            if machine_ids_map["{}_{}".format(machine_name, machine_id)] != int(model_params["context"]):
                continue
        dir_labels = []
        dir_mses = []

        # Initial embedding size. 
        # For every window, we should empty the accumulated activations.        
        print("Testing on files in {}".format(test_dir), "Number of files: {}".format(len(glob.glob(test_dir+"/*.npy"))))
        test_files = np.array(glob.glob(test_dir+"/*.npy"))
        perms = np.random.RandomState(seed=1).permutation(len(test_files))        
        shuffled_test_files = test_files[perms]
        for idx, (test_file) in enumerate(shuffled_test_files):
            file_data = test_file.split('/')[-1].split('.')[0].split('_')
            ID, machine_id, label = file_data[1], int(file_data[2]), int(file_data[3])
            print("MACHINE_ID {} LABEL {}".format(machine_id, label))
            # Get normal/anomalous label for given file. 
            dir_labels.append(label)
            test_data = Dataset(
                test_dir,
                "X_{}_{}_{}.npy".format(ID, machine_id, label),
                context_filename=None,
             )

            test_loader = DataLoader(
                test_data,
                batch_size=params.batch_size, # Needed to be sequential to mimic stream. 
                shuffle=False,
                num_workers=1
             )
            test_context_net =context_model_module.test
            prediction = test_context_net(
                context_model,
                device,
                test_loader,
                checkpoint=checkpoint_name
                )


            if len(window_preds) == window_size:
                # Make room for new example if window is full
                window_preds.pop(0)
                assert len(window_preds) == (window_size - 1)
            decay_factor = args.decay_factor
            window_preds.append(prediction)
            # Use len(window_preds) because at the start we will have less than the window size 
            weights_per_t = [(1 - decay_factor)**(len(window_preds)-t) for t in range(1, len(window_preds)+1)]
            context_scores = np.zeros(16)
            for context in range(len(context_scores)):
                context_scores[context] = sum(weights_per_t*(1*(np.array(window_preds)==context)))
            context_class = np.argmax(context_scores)

            context = np.zeros(16)
            context[context_class] = 1
            context = context.astype(np.float32)
            context = torch.from_numpy(context)
            print(context_class, context)
            file_mse = test(model, context, device, test_loader, checkpoint=iter_checkpoint)
            dir_mses.append(file_mse)     
        fpr, tpr, thresholds = roc_curve(dir_labels, dir_mses)
        test_auc_score =  auc(fpr, tpr) 
        results["{}_{}_{}".format(machine_name, machine_id, dB)] = {"AUC": float(test_auc_score)}
        results_file_name = "results_{}.json".format(iter_checkpoint.split('/')[-1])
        with open(os.path.join(results_dir, results_file_name), "w") as f:
            json.dump(results, f)                
                        
        results_log.append(float(test_auc_score))
       
    logs = {"model": model_params["model"], 
            "checkpoint_name": iter_checkpoint, 
            "val_json": args.val_json,
            "auc_scores": results_log,
        }

    with open(
            os.path.join(
                    log_dir, 
                    "test_{}_{}_iter_{}_{}.json".format(
                            model_params["model"], 
                            model_params["context"],
                            args.iter, 
                            time.strftime("%d%m%y_%H%M%S"))), 'w') as f:
            json.dump(logs, f)

if __name__ == '__main__':

        main()
