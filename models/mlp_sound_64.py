from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(320, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 16)
    
    def forward(self, x):
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))
        fc4 = F.relu(self.fc4(fc3))
        fc5 = F.relu(self.fc5(fc4))
        fc6 = F.relu(self.fc6(fc5))
        fc7 = self.fc7(fc6)
        return fc7

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cost = nn.CrossEntropyLoss()
    losses = []
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)
                label = torch.squeeze(label)
                label = np.argmax(label, axis=1)
                optimizer.zero_grad()
                output = model(data)
                loss = cost(output, label)        
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
            for batch_idx, (data, label) in enumerate(val_loader):
                    data = data.to(device)
                    target = label.clone()
                    softmax = torch.nn.Softmax(dim=1)
                    output = softmax(model(data))
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())     
                    progress_bar.update(1) 
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    preds = np.argmax(preds, axis=1)
    targets = np.argmax(targets, axis=1)
    acc = accuracy_score(targets, preds)
    print("Validation accuracy:", acc)
    return acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
            model_state = torch.load(checkpoint)
            model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as progress_bar:
            for batch_idx, (data) in enumerate(test_loader):
                    data = data.to(device)
                    softmax = torch.nn.Softmax(dim=1)
                    output = softmax(model(data))
                    preds.append(output.cpu().numpy())
                    progress_bar.update(1) 
    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)
    (values,counts) = np.unique(preds,return_counts=True)
    ind=np.argmax(counts)
    return values[ind]

def get_mean_embedding(model, device, test_loader,context_layer_name, window_size, checkpoint=None):
    if checkpoint is not None:
            model_state = torch.load(checkpoint)
            model.load_state_dict(model_state)    
    # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
    # This is a global dict TODO: Tidy this up. 
    activations = {}
    def get_activation(name, model, activation):
        def hook(model, inputs, output):
            activation[name] = output.detach()
        return hook
    embeddings_layer = getattr(model.module, context_layer_name)
    embedding = embeddings_layer.register_forward_hook(
    get_activation(
        context_layer_name, model, activations
        )
    )
    embedding_size = getattr(model.module, context_layer_name).out_features
    model.eval()
    preds = []
    targets = []
    context_activations = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as progress_bar:
            for batch_idx, (data) in enumerate(test_loader):
                    data = data.to(device)
                    _ = model(data)
                    example_i_activation = activations[context_layer_name].clone()
                    context_activations.append(example_i_activation.cpu().numpy())
                    progress_bar.update(1) 
    context_activations = np.concatenate(context_activations)
    mean_context_activations = torch.mean(torch.from_numpy(context_activations), 0)
    # Check that the embedding has the expected dimension. 
    assert mean_context_activations.shape[-1] == embedding_size
    return mean_context_activations
