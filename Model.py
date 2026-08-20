import torch
import torch.nn as nn
import torch.nn.functional as F

from Constants import NEURONS, DEVICE, P, WEIGHT_DECAY, EPOCHS



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2,NEURONS//2)
        self.lin2 = nn.Linear(NEURONS//2,NEURONS)
        self.lin3 = nn.Linear(NEURONS,NEURONS//2)
        self.lin4 = nn.Linear(NEURONS//2,1)
        self.drop1 = nn.Dropout(p=P)
        self.drop2 = nn.Dropout(p=P)
        self.drop3 = nn.Dropout(p=P)

    def forward(self, x):
        x = self.drop1(F.relu(self.lin1(x))) # hidden layer 1
        x = self.drop2(F.relu(self.lin2(x))) # hidden layer 2
        x = self.drop3(F.relu(self.lin3(x))) # hidden layer 3
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






