from sklearn.datasets import make_moons, make_friedman2, make_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn .tree import DecisionTreeRegressor

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np







class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")
        self.cmap = "RdBu"


    def plot(self,X,y,xx1,xx2,y_pred):
        # TODO: norm
        norm = Normalize(vmin=min(y.min(), y_pred.min()), vmax=max(y.max(), y_pred.max()))

        surface = self.ax.contourf(xx1, xx2, y_pred, levels=50, cmap=self.cmap, norm = norm)
        self.fig.colorbar(surface, label="y_inf")

        points = self.ax.scatter(X[:,0],X[:,1],c=y,cmap=self.cmap, norm=norm)







    def show(self):
        plt.show()



class Data:
    def __init__(self):
        self.w = 3

    def create_data(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-self.w,self.w,size=(200,2))
        y = 10 *  np.sin(X[:,0]*X[:,1])     +    2 * X[:,0] ** 2     +     rng.normal(0,1,size=200)
        return X, y

    
    def create_data_inference(self, n_points=200):
        x1 = np.linspace(-self.w, self.w, n_points)
        x2 = np.linspace(-self.w, self.w, n_points)
        xx1, xx2 = np.meshgrid(x1, x2)
        X = np.column_stack((xx1.ravel(), xx2.ravel()))
        return X, xx1, xx2







class Model:
    def __init__(self):
        # self.model = LinearRegression()
        # self.model = make_pipeline(PolynomialFeatures(degree=4, include_bias=False),LinearRegression())
        self.model = DecisionTreeRegressor()
        # neural network
        # increase dimensions of neural network until dimension of my 3d bbox problem. and for 3d bbox problem also try trees and polynomials.
        # increase size and dimension of problem until i can see that neural network helps over tree or polynomial regression
        # get intuition about inductive bias of CNN and Transformers

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    




class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2,10)
        self.lin2 = nn.Linear(10,10)
        self.lin3 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.lin1(x)) # hidden layer 1
        x = F.relu(self.lin2(x)) # hidden layer 2
        x = self.lin3(x)         # output layer
        return x








visualizer = Visualizer()
data = Data()
model = Model()

X,y = data.create_data()
model.train(X,y)

X_inf, xx1, xx2 = data.create_data_inference()# RESHAPING / CHANGING SHAPE
y_inf = model.predict(X_inf).reshape(xx1.shape)
visualizer.plot(X,y,xx1,xx2,y_inf)
visualizer.show()
