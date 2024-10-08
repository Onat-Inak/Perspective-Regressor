from torch.utils.data import Dataset
import torch


class Dataset(Dataset):
    def __init__(self, X, y, device):
        self.X = X.to(device)
        self.y = y.to(device)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return (X, y)

    def __len__(self):
        count = self.X.shape[0]
        return count
