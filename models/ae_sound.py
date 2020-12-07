from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
class net(nn.Module):
    def __init__(self, **kwargs):
        super(net, self).__init__()
        self.en_1 = nn.Linear(320, 64)
        self.en_2 = nn.Linear(64, 64)
        self.en_3 = nn.Linear(64, 8)
        self.de_1= nn.Linear(8, 64)
        self.de_2 = nn.Linear(64, 64)
        self.de_3 = nn.Linear(64, 320)


    def forward(self, x):
        encoder  = F.relu(self.en_1(x))
        encoder = F.relu(self.en_2(encoder))
        encoder = F.relu(self.en_3(encoder))
        decoder = F.relu(self.de_1(encoder))
        decoder = F.relu(self.de_2(decoder))
        decoder = self.de_3(decoder)
        return decoder


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cost = nn.MSELoss()
    losses = []
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data) in tqdm(enumerate(train_loader)): 
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, data)        
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress_bar.update(1)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, np.mean(losses)))

def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)      
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as progress_bar:
            for batch_idx,(data) in enumerate(val_loader):
                data = data.to(device)
                target = data.clone()
                output = model(data)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())    
                progress_bar.update(1)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    errors = np.mean((preds - targets)**2, axis=(1))
    mean_error = np.mean(errors) 
    print("Validation error:", mean_error)  
    return mean_error

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)         
    model.eval()
    preds = []
    targets = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as progress_bar:
            for batch_idx, (data) in enumerate(test_loader): 
                data = data.to(device)
                target = data.clone()
                output = model(data)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                progress_bar.update(1)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    errors = np.mean((preds - targets)**2, axis=(1))
    return errors


