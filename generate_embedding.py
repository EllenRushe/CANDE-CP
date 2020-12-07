import os
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from data_utils.params import Params
from data_utils import Datasets 



parser = argparse.ArgumentParser()
parser.add_argument(
      "--model_name", type=str, help="Pass name of model as defined in embedding_params.yaml.")
parser.add_argument(
      "--ckpt_name", type=str, help="Pass name of checkpoint.")
args = parser.parse_args()

params = Params("embedding_params.yaml", args.model_name)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
use_gpu= torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print("GPU AVAILABLE", use_gpu)
if not os.path.exists(params.embedding_dir): os.mkdir(params.embedding_dir)

Dataset = getattr(Datasets, params.dataset_class)

train_data = Dataset(
            params.data_dir, 
            "X.npy", 
            "contexts.npy"
            )

train_loader = DataLoader(
            train_data,
            num_workers=1,
            shuffle=True,  
            batch_size=params.batch_size
            )

model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
model = model_module.net()
model = nn.DataParallel(model)
model.to(device) 
embeddings_layer = getattr(model.module, params.layer_name)
model_state = torch.load(os.path.join(params.checkpoint_dir, args.ckpt_name))
model.load_state_dict(model_state)

# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
# This is a global dict TODO: Tidy this up. 
activations = {}
def get_activation(name, model, activation):
    def hook(model, inputs, output):
        activation[name] = output.detach()
    return hook

context_activations = {}
batched_context_ints = []
batched_activations = []
model.eval()
with torch.no_grad():
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, context) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data, context = data.to(device), context.to(device)
            embeddings_layer.register_forward_hook(get_activation(params.layer_name, model, activations))
            output = model(data)
            batch_context_ints= torch.argmax(context, dim=1)
            batch_activations = activations[params.layer_name].clone()
            batched_activations.append(batch_activations.cpu().numpy())
            batched_context_ints.append(batch_context_ints.cpu().numpy())
            progress_bar.update(1)
all_activations = np.concatenate(batched_activations)
all_context_ints = np.concatenate(batched_context_ints)

context_activations = {}
for context_int in sorted(set(all_context_ints)):
    context_activations[context_int] = all_activations[all_context_ints==context_int]

# Shape of embeddings: (number of contexts, size of last layer) 
embeddings = np.zeros((len(context_activations.keys()),context_activations[0].shape[-1]))
for k,v in context_activations.items():
    embeddings[k] = np.mean(v, axis=(0))
min_embed = embeddings.min(axis=0)
max_embed = embeddings.max(axis=0)

norm_embeddings = (embeddings - embeddings.min(axis=0))/(embeddings.max(axis=0) - embeddings.min(axis=0))

embedding_file_name = args.ckpt_name
np.save(os.path.join(params.embedding_dir,embedding_file_name), norm_embeddings)
np.save(os.path.join(params.embedding_dir,embedding_file_name+"_raw"), embeddings)
print("Embedding shape:", norm_embeddings.shape)
print("Embedding name:", embedding_file_name)
