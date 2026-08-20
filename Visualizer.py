import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



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