from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import models 
from data_utils import Datasets
from data_utils.params import Params

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, help="Pass name of model as defined in hparams.yaml.")
    args = parser.parse_args()
    params = Params("hparams.yaml", args.model_name)

    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)

    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net().to(device)
   
    model = nn.DataParallel(model)
    model.to(device)
    
    train = model_module.train
    val = model_module.val

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    train_data_dir = os.path.join(params.data_dir, "train")
    val_data_dir = os.path.join(params.data_dir, "val")
    
    Dataset = getattr(Datasets, params.dataset_class)

    train_data = Dataset(train_data_dir, 'X.npy', 'contexts.npy')
    val_data = Dataset(val_data_dir, 'X.npy', 'contexts.npy')
    
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
    val_f1s = []
    val_accs = []
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    for epoch in range(1, params.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        val_acc = val(model, device, val_loader)
        val_accs.append(float(val_acc))
        torch.save(
            model.state_dict(), os.path.join(
                params.checkpoint_dir,
                "checkpoint_{}_epoch_{}".format(
                args.model_name, 
                epoch
                )
            )
        )
        logs ={
            "val_accs": val_accs, 
            "best_val_epoch": int(np.argmax(val_accs)+1),
            "model": args.model_name, 
            "lr": params.lr, 
            "batch_size": params.batch_size,
        }
   
        with open(
            os.path.join(
                params.log_dir, 
                "{}.json".format(args.model_name)), 'w') as f:
            json.dump(logs, f)

if __name__ == '__main__':
    main()
