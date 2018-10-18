import autograd.numpy as np
from scipy.optimize import brute

from algorithms import FletcherILS
from functions import func4
from vis_functions import plot_3d

interval    = [-np.pi, +np.pi]
numPoints   = 1000

x = np.linspace(interval[0], interval[1], num=numPoints)
y = np.linspace(interval[0], interval[1], num=numPoints)
X, Y = np.meshgrid(x, y)

Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = func4([X[i,j], Y[i,j]])

plot_3d(X, Y, Z, save='png', fig_name="3D Plot", show=True)
