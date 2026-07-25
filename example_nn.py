from sklearn.datasets import make_moons, make_friedman2, make_regression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn .tree import DecisionTreeRegressor

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time







class Visualizer:
    def __init__(self,y_train,y_val,y_inf):
        self.fig, self.ax = plt.subplots(1,2)
        self.ax_inference = self.ax[0]
        self.ax_loss = self.ax[1]

        self.ax_inference.set_xlabel('Feature 1')
        self.ax_inference.set_ylabel('Feature 2')
        self.cmap = 'RdBu'
        self.norm = Normalize(vmin=min(y_train.min(), y_val.min(), y_inf.min()), vmax=max(y_train.max(), y_val.max(), y_inf.max()))

    def plot_surface(self,xx1,xx2,y_inf):
        surface = self.ax_inference.contourf(xx1, xx2, y_inf, levels=50, cmap=self.cmap, norm = self.norm)
        self.fig.colorbar(surface, label='y_inf')

    def plot_points(self,X,y,edgecolors=None):
        points = self.ax_inference.scatter(X[:,0],X[:,1],c=y,cmap=self.cmap, norm=self.norm, edgecolors=edgecolors)

    def plot_history(self,history):
            self.ax_loss.plot(history['epochs'], history['train'], label='Train')
            self.ax_loss.plot(history['epochs'], history['val'], label='Validation')
            self.ax_loss.set_xlabel('epochs')
            self.ax_loss.set_ylabel('RMSE')
            self.ax_loss.legend()

    def show(self):
        plt.show()



class Data:
    def __init__(self):
        self.w = 3

    def create_data(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-self.w,self.w,size=(100,2))
        y = 10 *  np.sin(X[:,0]*X[:,1])     +    2 * X[:,0] ** 2     +     rng.normal(0,1,size=100)
        return X, y

    
    def create_data_inference(self, n_points=100):
        x1 = np.linspace(-self.w, self.w, n_points)
        x2 = np.linspace(-self.w, self.w, n_points)
        xx1, xx2 = np.meshgrid(x1, x2)
        X = np.column_stack((xx1.ravel(), xx2.ravel()))
        return X, xx1, xx2



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2,8)
        self.lin2 = nn.Linear(8,8)
        self.lin3 = nn.Linear(8,1)

    def forward(self, x):
        x = F.relu(self.lin1(x)) # hidden layer 1
        x = F.relu(self.lin2(x)) # hidden layer 1
        x = self.lin3(x)         # output layer
        return x



# dimension of X with X features and y with Y features
# pytorch with batch:   X:[N,X]   y:[N,Y]
# pytorch without batch X:[X]     y:[Y]

class Model:
    def __init__(self):
        self.model = NeuralNetwork()
        # neural network
        # increase dimensions of neural network until dimension of my 3d bbox problem. and for 3d bbox problem also try trees and polynomials.
        # increase size and dimension of problem until i can see that neural network helps over tree or polynomial regression
        # get intuition about inductive bias of CNN and Transformers

    def train(self, X, y, validation_data=None, epochs=100000):
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32).unsqueeze(1) # [N] -> [N,1]
        if validation_data:
            X_val, y_val = validation_data
            X_val = torch.from_numpy(X_val).type(torch.float32)
            y_val = torch.from_numpy(y_val).type(torch.float32).unsqueeze(1) # [N] -> [N,1]
            validation_data = (X_val, y_val)
        optimizer = torch.optim.SGD(self.model.parameters(),lr=0.001)
        criterion = nn.MSELoss()
        history = {'train':[],'val':[],'epochs':[]}

        for i in range(epochs):
            # forward
            y_pred= self.model(X)
            loss = criterion(y_pred, y)
            # backwards
            loss.backward()
            # update weights according to gradients of weights from backward pass
            optimizer.step()
            optimizer.zero_grad()
            if validation_data and i%100==0:
                rmse_train, rmse_val = self.validate(X, y , validation_data)
                history['train'].append(rmse_train)
                history['val'].append(rmse_val)
                history['epochs'].append(i)
        return history

    def validate(self, X, y, validation_data):
        criterion = nn.MSELoss()
        X_val, y_val = validation_data
        with torch.inference_mode():
            y_pred = self.model(X)
            y_val_pred = self.model(X_val)
            rmse_train = torch.sqrt(criterion(y_pred, y))
            rmse_val = torch.sqrt(criterion(y_val_pred, y_val))
        return rmse_train.item(), rmse_val.item()
    
    def predict(self, X):
        X = torch.from_numpy(X).type(torch.float32)
        with torch.inference_mode():
            y_pred = self.model(X)
        y_pred = y_pred.squeeze(1).numpy() # [N,1] -> [N]
        return y_pred
        
    







start = time.time()


seed = 42
if seed is not None:
    torch.manual_seed(seed)
data = Data()
model = Model()

# train
X,y = data.create_data(seed)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
history = model.train(X_train,y_train,validation_data=(X_val,y_val))

# test and error
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
train_error = root_mean_squared_error(y_train, y_train_pred)
test_error = root_mean_squared_error(y_val, y_val_pred)
print(f'train_error: {train_error}')
print(f'test_error: {test_error}\n')

# inference and visualization
X_inf, xx1, xx2 = data.create_data_inference()
y_inf = model.predict(X_inf).reshape(xx1.shape)
visualizer = Visualizer(y_train,y_val,y_inf)
visualizer.plot_surface(xx1,xx2,y_inf)
visualizer.plot_points(X_train,y_train)
visualizer.plot_points(X_val,y_val,edgecolors='black')
visualizer.plot_history(history)
print(f'total time: {time.time()-start}')
visualizer.show()



# TODO:
# underfitting vs overfitting: stop training at right time (-> epochs. but also look for learning rate scheduler and optimizer)
# underfitting vs overfitting: right model size (-> nn architecture)
# regularizing
