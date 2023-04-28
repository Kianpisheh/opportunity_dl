import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, samples):
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x = sample[:,1:-1].T
        y = int(sample[0,-1])

        return x, y