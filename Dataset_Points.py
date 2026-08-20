import numpy as np
import torch
from torch.utils.data import Dataset

from Constants import DATA_POINTS, NOISE, DEVICE


class Points:
    def __init__(self):
        self.w = 3

    def create_data(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-self.w,self.w,size=(DATA_POINTS,2))
        y = 10 *  np.sin(X[:,0]*X[:,1])     +    2 * X[:,0] ** 2     +     rng.normal(0,NOISE,size=DATA_POINTS)
        return X, y

    def create_data_inference(self, n_points=100):
        x1 = np.linspace(-self.w, self.w, n_points)
        x2 = np.linspace(-self.w, self.w, n_points)
        xx1, xx2 = np.meshgrid(x1, x2)
        X = np.column_stack((xx1.ravel(), xx2.ravel()))
        return X, xx1, xx2


class Dataset_Points(Dataset):
    def __init__(self,X,y):
        X = torch.from_numpy(X).type(torch.float32).to(DEVICE)
        y = torch.from_numpy(y).type(torch.float32).unsqueeze(1).to(DEVICE) # [N] -> [N,1]
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    