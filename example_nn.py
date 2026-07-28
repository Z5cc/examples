from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time






DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {DEVICE}')
BATCH_SIZE = 32
EPOCHS = 2000
SEED = 42
if SEED is not None:
    torch.manual_seed(SEED)
NOISE = 1 # losses shall get close to that noise
NEURONS = 16
print(f'neurons: {NEURONS}')
WEIGHT_DECAY = 0.00 # regularization
print(f'weight_decay: {WEIGHT_DECAY}')





class Visualizer:
    def __init__(self,y_train,y_val,y_inf):
        self.fig = plt.figure()

        self.ax_inference = self.fig.add_subplot(2,2,1)
        self.ax_height = self.fig.add_subplot(1,2,2, projection='3d')
        self.ax_loss = self.fig.add_subplot(2,2,3)

        self.ax_inference.set_xlabel('Feature 1')
        self.ax_inference.set_ylabel('Feature 2')
        self.cmap = 'RdBu'
        self.norm = Normalize(vmin=min(y_train.min(), y_val.min(), y_inf.min()), vmax=max(y_train.max(), y_val.max(), y_inf.max()))

    def plot_colorsurface(self,xx1,xx2,y_inf):
        surface = self.ax_inference.contourf(xx1, xx2, y_inf, levels=50, cmap=self.cmap, norm=self.norm)
        self.fig.colorbar(surface, label='y_inf')

    def plot_colorsurface_points(self,X,y,edgecolors=None):
        self.ax_inference.scatter(X[:,0],X[:,1],c=y,cmap=self.cmap, norm=self.norm, edgecolors=edgecolors)


    def plot_height(self,xx1,xx2,y_inf):
        self.ax_height.plot_surface(xx1, xx2, y_inf,cmap=self.cmap,norm=self.norm,alpha=0.7)
        # self.ax_height.plot_wireframe(xx1, xx2, y_inf)

    def plot_height_points(self,X,y,edgecolors=None):
        self.ax_height.scatter(X[:,0],X[:,1],y,norm=self.norm,edgecolors=edgecolors)

    def plot_history(self,history):
        self.ax_loss.plot(history['epochs'], history['train'], label='Train')
        self.ax_loss.plot(history['epochs'], history['val'], label='Validation')
        self.ax_loss.set_xlabel('epochs')
        self.ax_loss.set_ylabel('RMSE')
        self.ax_loss.legend()

    def show(self):
        plt.show()





class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2,NEURONS//2)
        self.lin2 = nn.Linear(NEURONS//2,NEURONS)
        self.lin3 = nn.Linear(NEURONS,NEURONS//2)
        self.lin4 = nn.Linear(NEURONS//2,1)

    def forward(self, x):
        x = F.relu(self.lin1(x)) # hidden layer 1
        x = F.relu(self.lin2(x)) # hidden layer 2
        x = F.relu(self.lin3(x)) # hidden layer 3
        x = self.lin4(x)         # output layer
        return x














# dimension of X with X features and y with Y features
# pytorch with batch:   X:[N,X]   y:[N,Y]
# pytorch without batch X:[X]     y:[Y]

class Model:
    def __init__(self):
        self.model = NeuralNetwork().to(DEVICE)
        # neural network
        # increase dimensions of neural network until dimension of my 3d bbox problem. and for 3d bbox problem also try trees and polynomials.
        # increase size and dimension of problem until i can see that neural network helps over tree or polynomial regression
        # get intuition about inductive bias of CNN and Transformers

    def train(self, dataloader_train, dataloader_val_tr, dataloader_val):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, weight_decay=WEIGHT_DECAY)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        criterion = nn.MSELoss()
        history = {'train':[],'val':[],'epochs':[]}

        for i in range(EPOCHS):
            self.model.train()
            for X, y in dataloader_train:
                # forward
                y_pred= self.model(X)
                loss = criterion(y_pred, y)
                # backwards
                loss.backward()
                # update weights according to gradients of weights from backward pass
                optimizer.step()
                optimizer.zero_grad()
            if dataloader_val and i%100==0:
                rmse_train = self.validate(dataloader_val_tr)
                rmse_val = self.validate(dataloader_val)
                history['train'].append(rmse_train)
                history['val'].append(rmse_val)
                history['epochs'].append(i)
            # scheduler.step()
        return history

    def validate(self, dataloader):
        self.model.eval()
        criterion = nn.MSELoss()
        with torch.inference_mode():
            X, y = next(iter(dataloader))
            y_pred = self.model(X)
            rmse = torch.sqrt(criterion(y_pred, y))
        return rmse.item() # item() puts from GPU to CPU
    
    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).type(torch.float32).to(DEVICE)
        with torch.inference_mode():
            y_pred = self.model(X)
        y_pred = y_pred.squeeze(1).cpu().numpy() # [N,1] -> [N]
        return y_pred # cpu() puts from GPU to CPU










class Points:
    def __init__(self):
        self.w = 3
        self.n = 50

    def create_data(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-self.w,self.w,size=(self.n,2))
        y = 10 *  np.sin(X[:,0]*X[:,1])     +    2 * X[:,0] ** 2     +     rng.normal(0,NOISE,size=self.n)
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









start = time.time()



# create data
points = Points()
X,y = points.create_data(SEED)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
dataset_train = Dataset_Points(X_train, y_train)
dataset_val = Dataset_Points(X_val, y_val)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE)
dataloader_val_tr = DataLoader(dataset_train, shuffle=False, batch_size=len(dataset_train))
dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=len(dataset_val))


# train
model = Model()
history = model.train(dataloader_train, dataloader_val_tr, dataloader_val)


# test and error
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
train_error = root_mean_squared_error(y_train, y_train_pred)
test_error = root_mean_squared_error(y_val, y_val_pred)
print(f'train_error: {train_error}')
print(f'test_error: {test_error}\n')


# inference and visualization
X_inf, xx1, xx2 = points.create_data_inference()
y_inf = model.predict(X_inf).reshape(xx1.shape)
visualizer = Visualizer(y_train,y_val,y_inf)
visualizer.plot_colorsurface(xx1,xx2,y_inf)
visualizer.plot_colorsurface_points(X_train,y_train)
visualizer.plot_colorsurface_points(X_val,y_val,edgecolors='black')
visualizer.plot_height(xx1,xx2,y_inf)
visualizer.plot_height_points(X_train,y_train,edgecolors='yellow')
visualizer.plot_height_points(X_val,y_val,edgecolors='black')
visualizer.plot_history(history)
print(f'total time: {time.time()-start}')
visualizer.show()



# TODO:
# underfitting vs overfitting: stop training at right time (-> epochs. but also look for learning rate scheduler and optimizer)
# underfitting vs overfitting: right model size (-> nn architecture)
# regularizing
