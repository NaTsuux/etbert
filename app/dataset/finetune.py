import logging
import pickle

import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger("Dataset")


class FineTuneDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # item: (src, label)
        item = self.data[idx]
        return item
