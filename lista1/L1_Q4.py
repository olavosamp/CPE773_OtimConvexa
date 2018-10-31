import autograd.numpy as np
from scipy.optimize import brute
import matplotlib.pyplot as plt

from libs import dirs
from libs.line_search import FletcherILS, BacktrackingLineSearch
from libs.functions import func4
from libs.vis_functions import plot_3d, plot_contour, plot_line_search

interval    = [-np.pi, +np.pi]
numPoints   = 1000

x = np.linspace(interval[0], interval[1], num=numPoints)
y = np.linspace(interval[0], interval[1], num=numPoints)
X, Y = np.meshgrid(x, y)

Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = func4([X[i,j], Y[i,j]])


# 3D Plot
# plot_3d(X, Y, Z, save='png', fig_name="3D_Plot", show=True)

# Contour Plot
fig, ax = plot_contour(X, Y, Z, save=False, fig_name="Contour_Plot", show=False)

## Compute Fletcher's Inexact Line Search
initialX = np.array([-np.pi, +np.pi])
initialDir = np.array([1.0, -1.1])
xtol = 1e-5
maxIters = 500

fletcher = FletcherILS(func4, interval, xtol=xtol, maxIters=maxIters, initialX=initialX, initialDir=initialDir)
xOpt = fletcher.optimize()

alphaList = np.array(fletcher.alpha0List)
dirList = np.array(fletcher.dirList)
bestF = fletcher.f0
# print("alpha: ", alphaList.shape)
# print("dir: ", dirListline_search
xLen = np.shape(alphaList)[0]

xList = np.zeros((xLen, 2))
xList[0] = initialX
for i in range(1, xLen):
    xList[i] = xList[i-1] + alphaList[i]*dirList[i]

# Clip values to search space: may be cheating
# xList = np.clip(xList, interval[0], interval[1])

print(xList)
print(np.shape(xList))
print(alphaList)

print("x* = ", xList[-1])
print("f(x*) = ", bestF)

ax.plot(xList[0,0], xList[0,1], 'bo', markersize=10, label='Starting point')
ax.plot(xList[:,0], xList[:,1], 'rx', markersize=7, label='Line Search', linestyle='--', linewidth=1)

fig = plt.gcf()
ax.set_title("Contour Plot and Fletcher's Inexact Line Search")
fig.set_size_inches(18, 10)
fig.legend(loc='upper right')
fig.savefig(dirs.figures+"Contour_Plot_ILS"+".png", orientation='portrait', bbox_inches='tight')

plt.clf()
plot_line_search(alphaList, initialX, initialDir, func4,fig_name="Line_search_plot_ILS", save='png', show=False)

## Compute Backtracking Line Search
print("\nBacktracking Line Search")
xtol = 1e-5
maxIters = 500

backtrack = BacktrackingLineSearch(func4, interval, xtol=xtol, maxIters=maxIters, initialX=initialX, initialDir=initialDir)
xOpt = backtrack.optimize()

# Backtrack Figure
fig, ax = plot_contour(X, Y, Z, save=False, fig_name="Contour_Plot", show=False)

alphaList = np.array(backtrack.alphaList)
dirList = np.array(backtrack.dirList)
bestF = backtrack.fx
print("alpha shape: ", alphaList.shape)
# print("dir: ", dirList.shape)
xLen = np.shape(alphaList)[0]

xList = np.zeros((xLen, 2))
xList[0] = initialX
for i in range(1, xLen):
    xList[i] = xList[i-1] + alphaList[i]*dirList[i]

# Clip values to search space: may be cheating
# xList = np.clip(xList, interval[0], interval[1])

print(xList)
print(np.shape(xList))

print("x* = ", xList[-1])
print("f(x*) = ", bestF)

ax.plot(xList[0,0], xList[0,1], 'bo', markersize=10, label='Starting point')
ax.plot(xList[:,0], xList[:,1], 'rx', markersize=7, label='Line Search', linestyle='--', linewidth=1)

fig = plt.gcf()
ax.set_title("Contour Plot and Backtracking Line Search")
fig.set_size_inches(18, 10)
fig.legend(loc='upper right')
fig.savefig(dirs.figures+"Contour_Plot_BT"+".png", orientation='portrait', bbox_inches='tight')

plt.clf()
plot_line_search(alphaList, initialX, initialDir, func4, fig_name="Line_search_plot_BT", save='png', show=False)
