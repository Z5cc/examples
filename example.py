from sklearn.datasets import make_moons, make_friedman2, make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np






class Visualizer:
    def __init__(self):
        pass

    def scatter(self,X,y):
        plt.scatter(X[:,0],X[:,1],c=y,cmap="viridis")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.colorbar(label="Target y")
        plt.show(block=False)
    
    def scatter_inference(self,X,y):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=0.001) # TODO: maybe continue with pcolormesh.... instead of scatter here
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.colorbar(label="Predicted y")
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
        return X






class Model:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred








visualizer = Visualizer()
data = Data()
model = Model()

X,y = data.create_data()
visualizer.scatter(X,y)
model.train(X,y)

X_inf = data.create_data_inference()
y_inf = model.predict(X_inf)
visualizer.scatter_inference(X_inf,y_inf)
