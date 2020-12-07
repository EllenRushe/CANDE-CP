from __future__ import print_function
import os
import time
from tqdm import tqdm
import json
import glob
import argparse
import pickle
import random
import numpy as np
import random
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc

import models 
from data_utils import Datasets
from data_utils.params import Params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, help="Pass name of model as defined in hparams.yaml.")
    parser.add_argument(
        "context", type=str, default="", help="Context to train separate autoencoders on.")
    parser.add_argument(
        "iter", type=int, default=1, help="Indicates which training run this is (multiple iterations with different random initialisations for reporting variance between runs).")
    args = parser.parse_args()
    params = Params("hparams.yaml", args.model_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Device", device)
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    train_data_dir = os.path.join(params.data_dir, "train", args.context)
    val_data_dir = os.path.join(params.data_dir, "val", args.context)
    
    Dataset = getattr(Datasets, params.dataset_class)

    train_data = Dataset(
        train_data_dir, 
        'X.npy', 
        context_filename=params.context_filename
        )
        
    val_data = Dataset(
        val_data_dir, 
        'X.npy', 
        context_filename=params.context_filename
        )
    

    train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size, 
        shuffle=True, 
        num_workers=1
        )

    val_loader = DataLoader(
        val_data, 
        batch_size=params.batch_size, 
        shuffle=False, 
        num_workers=1
        )

    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    if params.context_embedding_file:
        embeddings = torch.from_numpy(np.load(params.context_embedding_file)).float()
    else:
        embeddings = None
    model = model_module.net(embeddings=embeddings)
    model = nn.DataParallel(model).to(device)
    train = model_module.train
    val = model_module.val

    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)

    val_mses = []
    for epoch in range(1, params.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        val_error  = val(model, device, val_loader)
        val_mses.append(float(val_error))
        torch.save(
            model.state_dict(), os.path.join(
                params.checkpoint_dir,
                "checkpoint_{}_{}_iter_{}_epoch_{}".format(
                args.model_name, 
                args.context,
                args.iter,
                epoch
                )
            )
        )

        logs ={
            "iter": args.iter,
            "val_mses": val_mses,
            "best_val_epoch": int(np.argmin(val_mses)+1),
            "model": args.model_name,
            "lr": params.lr,
            "batch_size":params.batch_size,
            "context": args.context
        }

        with open(
            os.path.join(
                params.log_dir, "{}_{}_{}.json".format(args.model_name, args.context, args.iter)), 'w') as f:
            json.dump(logs, f)



if __name__ == '__main__':
    main()

