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







class Visualizer:
    def __init__(self,y_train,y_test,y_inf):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")
        self.cmap = "RdBu"
        self.norm = Normalize(vmin=min(y_train.min(), y_test.min(), y_inf.min()), vmax=max(y_train.max(), y_test.max(), y_inf.max()))

    def plot_surface(self,xx1,xx2,y_inf):
        surface = self.ax.contourf(xx1, xx2, y_inf, levels=50, cmap=self.cmap, norm = self.norm)
        self.fig.colorbar(surface, label="y_inf")

    def plot_points(self,X,y,edgecolors=None):
        points = self.ax.scatter(X[:,0],X[:,1],c=y,cmap=self.cmap, norm=self.norm, edgecolors=edgecolors)

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
        # self.model = DecisionTreeRegressor()
        self.model = NeuralNetwork()
        # neural network
        # increase dimensions of neural network until dimension of my 3d bbox problem. and for 3d bbox problem also try trees and polynomials.
        # increase size and dimension of problem until i can see that neural network helps over tree or polynomial regression
        # get intuition about inductive bias of CNN and Transformers

    def train(self, X, y):
        if isinstance(self.model, nn.Module):
            self.train_nn(X, y)
        else:
            self.model.fit(X, y)

    def train_nn(self, X, y):
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32)
        optimizer = torch.optim.SGD(self.model.parameters())
        criterion = nn.MSELoss()
        for _ in range(1000):
            # forward
            y_pred= self.model(X)
            loss = criterion(y_pred, y)
            # backwards
            loss.backward()
            # update weights according to gradients of weights from backward pass
            optimizer.step()
            optimizer.zero_grad()



    def predict(self, X):
        if isinstance(self.model, nn.Module):
            y_pred = self.predict_nn(X)
        else:
            y_pred = self.model.predict(X)
        return y_pred
    
    def predict_nn(self, X):
        X = torch.from_numpy(X).type(torch.float32)
        y_pred = self.model(X)
        y_pred = y_pred.detach().numpy()
        return y_pred
        
    




class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2,10)
        self.lin2 = nn.Linear(10,50)
        self.lin3 = nn.Linear(50,50)
        self.lin4 = nn.Linear(50,1)

    def forward(self, x):
        x = F.relu(self.lin1(x)) # hidden layer 1
        x = F.relu(self.lin2(x)) # hidden layer 2
        x = F.relu(self.lin3(x)) # hidden layer 3
        x = self.lin4(x)         # output layer
        return x








data = Data()
model = Model()

# train
X,y = data.create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.train(X_train,y_train)

# test and error
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_error = root_mean_squared_error(y_train, y_train_pred)
test_error = root_mean_squared_error(y_test, y_test_pred)
print(f"train_error: {train_error}")
print(f"test_error: {test_error}\n")

# inference and visualization
X_inf, xx1, xx2 = data.create_data_inference()
y_inf = model.predict(X_inf).reshape(xx1.shape)
visualizer = Visualizer(y_train,y_test,y_inf)
visualizer.plot_surface(xx1,xx2,y_inf)
visualizer.plot_points(X_train,y_train)
visualizer.plot_points(X_test,y_test,edgecolors='black')
visualizer.show()
