import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TorchDataset(Dataset):
    def __init__(self, csv_file, start=0, stop=None, seed=0, gpu=0):
        self.data = pd.read_csv(csv_file)
        y = self.data[["rhoT", "phiT"]].values[start:stop]
        x = self.data[["rho0", "phi0", "rho1", "phi1", "quad", "poisson"]]
        np.random.seed(seed)
        x.loc[:, "quad"] = (
            x["quad"].values + np.random.uniform(low=-0.5, high=0.5, size=len(x))
        ) / 4
        x.loc[:, 'poisson'] = (
            x["poisson"].values + np.random.uniform(low=-0.5, high=0.5, size=len(x))
        ) / 16
        x = x.values[start:stop]

        self.y = torch.from_numpy(y).float()
        self.x = torch.from_numpy(x).float()

        # self.y.cuda(gpu)
        # self.x.cuda(gpu)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
