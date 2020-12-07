import os
import torch
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset


class MachineSoundDataset(Dataset):
    def __init__(self, data_dir, X_filename, context_filename=None):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        context_filename (str): Name of file containing context labels.
        """
        self.data = torch.from_numpy(
            np.load(os.path.join(data_dir, X_filename)).astype(np.float32))
        self.context_filename = context_filename
        if self.context_filename  is not None:
            self.contexts_int = torch.from_numpy(np.load(os.path.join(data_dir, context_filename))).long()
            # One-hot-encoding
            contexts = np.zeros((self.contexts_int.shape[0], 16))                
            contexts[np.arange(self.contexts_int.shape[0]), self.contexts_int] = 1
            contexts = contexts.astype(np.float32)
            self.contexts = torch.from_numpy(contexts)
                
    def __getitem__(self, index):
        X = self.data[index]
        if self.context_filename is not None:
            context = self.contexts[index]
            return X, context
        
        return X

    def __len__(self):
        return len(self.data)
