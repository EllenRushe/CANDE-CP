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

# Format: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py


class FiLM(nn.Module):
    ''''
        Form: https://github.com/ethanjperez/film/blob/master/vr/models/filmed_net.py
    '''
    def forward(self, layer, gammas, betas):
        return (gammas * layer) + betas


class FiLMBlock(nn.Module):
    def __init__(self, in_size, out_size, in_betas, in_gammas):
        super(FiLMBlock, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.betas_fc = nn.Linear(in_betas, out_size)
        self.gammas_fc = nn.Linear(in_gammas, out_size)
        self.film = FiLM()

    def forward(self, x, gs, bs):
        fc = F.relu(self.fc(x))
        gammas = self.gammas_fc(gs)
        betas =  self.betas_fc(bs)
        return F.relu(self.film(fc, gammas, betas))


class net(nn.Module):
    def __init__(self, embeddings):
        super(net, self).__init__()
        num_embed, embed_size = embeddings.size()
        self.embeddings_gamma = nn.Embedding.from_pretrained(embeddings)
        self.embeddings_beta = nn.Embedding.from_pretrained(embeddings)
        self.en_1 = nn.Linear(320, 64)
        self.en_2 = FiLMBlock(64, 64, embed_size, embed_size)
        self.en_3 = FiLMBlock(64, 8, embed_size, embed_size)
        self.de_1= FiLMBlock(8, 64, embed_size, embed_size)
        self.de_2 = FiLMBlock(64, 64, embed_size, embed_size)
        self.de_3 = nn.Linear(64, 320)

    def forward(self, x, context):
        gammas = self.embeddings_gamma(context)
        betas =  self.embeddings_beta(context)       
        encoder  = F.relu(self.en_1(x))
        encoder = self.en_2(encoder, gammas, betas)
        encoder = self.en_3(encoder, gammas, betas)
        decoder = self.de_1(encoder, gammas, betas)
        decoder = self.de_2(decoder, gammas, betas)
        decoder = self.de_3(decoder)
        return decoder

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cost = nn.MSELoss()
    losses = []
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, context) in tqdm(enumerate(train_loader)):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            context_int = torch.argmax(context, dim=1)
            context_int = context_int.to(device)
            optimizer.zero_grad()
            output = model(data, context_int)
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
            for batch_idx,(data, context) in tqdm(enumerate(val_loader)):            
                data = data.to(device)
                context_int = torch.argmax(context, dim=1)
                context_int = context_int.to(device)
                target = data.clone()
                output = model(data, context_int)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                progress_bar.update(1)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    errors = np.mean((preds - targets)**2, axis=(1))
    mean_error = np.mean(errors)
    print("Validation error:", mean_error)  
    return mean_error

def test(model, context_int,  device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as progress_bar:
            for batch_idx, (data) in tqdm(enumerate(test_loader)):
                data = data.to(device)
                context_int = context_int.to(device)
                target = data.clone()
                output = model(data, context_int)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                progress_bar.update(1)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    errors = np.mean((preds - targets)**2, axis=1)
    mean_error = torch.mean(torch.from_numpy(errors))
    return mean_error
