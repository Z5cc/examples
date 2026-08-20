import numpy as np
import time

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Dataset_Points import Points, Dataset_Points
from Model import Model
from Visualizer import Visualizer
from Constants import SEED, BATCH_SIZE









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
