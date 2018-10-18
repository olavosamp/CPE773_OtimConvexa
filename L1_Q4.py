import autograd.numpy as np
from scipy.optimize import brute
import matplotlib.pyplot as plt

from algorithms import FletcherILS
from functions import func4
from vis_functions import plot_3d, plot_contour

interval    = [-np.pi, +np.pi]
numPoints   = 1000

x = np.linspace(interval[0], interval[1], num=numPoints)
y = np.linspace(interval[0], interval[1], num=numPoints)
X, Y = np.meshgrid(x, y)

Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = func4([X[i,j], Y[i,j]])

# plot_3d(X, Y, Z, save='png', fig_name="3D_Plot", show=True)
fig, ax = plot_contour(X, Y, Z, save=False, fig_name="Contour_Plot", show=False)

# Compute Fletcher's Inexact Line Search
initialX = np.array([np.pi, -np.pi])
initalDir = np.array([1.0, -1.3])
xtol = 1e-5
maxIters = 500

fletcher = FletcherILS(func4, interval, xtol=xtol, maxIters=maxIters, initialX=initialX)
xOpt = fletcher.optimize()

alphaList = np.array(fletcher.alpha0List)
dirList = np.array(fletcher.dirList)
print("alpha: ", alphaList.shape)
print("dir: ", dirList.shape)
xLen = np.shape(alphaList)[0]

xList = np.zeros((xLen+1, 2))
xList[0] = initialX
for i in range(1, xLen):
    xList[i] = xList[i-1] + alphaList[i]*dirList[i]

print(xList)
print(np.shape(xList))
#
# ax.plot(initialX, 'rx')
# plt.show()
