import torch
from torch.utils.data import Dataset
import os
import numpy as np

class NanoporeReadDataset(Dataset):
    def __init__(self, root, label):
        self.root = root
        self.label = label
        self.files = sorted([os.path.join(root, f) 
                             for f in os.listdir(root) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = np.load(self.files[idx])   # shape (7, L)
        x = torch.tensor(tensor, dtype=torch.float32)
        y = torch.tensor(self.label, dtype=torch.long)
        return x, y
