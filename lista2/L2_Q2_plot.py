import autograd.numpy       as np
import matplotlib.pyplot    as plt
from autograd               import grad
import scipy.optimize       as spo

from libs import dirs
from libs.line_search       import FletcherILS, BacktrackingLineSearch
from libs.functions         import func5
from libs.vis_functions     import plot_3d, plot_contour, plot_line_search

function = func5

interval     = [-4.1, 4.1]
numPoints    = 1000

initialXList = [[+4., +4.],
                [+4., -4.],
                [-4., +4.],
                [-4., -4.],]
# initialDir = np.array([1.0, -1.3])
initialDir   = None
maxIters     = 500
xtol         = 1e-6

# Assemble coordinate variables for 3D plot
x = np.linspace(interval[0], interval[1], num=numPoints)
y = np.linspace(interval[0], interval[1], num=numPoints)
X, Y = np.meshgrid(x, y)

Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = function([X[i,j], Y[i,j]])

## 3D Plot
# plot_3d(X, Y, Z, save='png', fig_name="3D_Plot", show=True)
# input()


## Contour Plot for Backtrack optimization
fig, ax = plot_contour(X, Y, Z, save=False, fig_name="Contour_Plot", show=False)
optimList = []
for initialX in initialXList:
    ## Compute Backtracking Line Search
    print("\nBacktracking Line Search")
    backtrack = BacktrackingLineSearch(function, interval, xtol=xtol, maxIters=maxIters, initialX=initialX, initialDir=initialDir)
    xOpt = backtrack.optimize()

    alphaList   = np.array(backtrack.alphaList)
    dirList     = np.array(backtrack.dirList)
    bestF       = backtrack.fx
    # print("alpha shape: ", alphaList.shape)
    # print("dir: ", dirList.shape)
    xLen  = np.shape(alphaList)[0]
    xList = np.zeros((xLen+1, 2))
    xList[0] = initialX
    for i in range(1, xLen+1):
        xList[i] = xList[i-1] + alphaList[i-1]*dirList[i-1]

    # Clip values to search space (may be cheating)
    # xList = np.clip(xList, interval[0], interval[1])

    print(xList)
    print(np.shape(xList))
    # print(alphaList)
    # print(np.shape(alphaList))
    # print(dirList)
    # print(np.shape(dirList))

    optimResult = spo.minimize(function, initialX, method='BFGS', tol=xtol)
    xRef = optimResult.x
    fRef = optimResult.fun

    print("x* = ", xList[-1])
    print("f(x*) = ", bestF)
    print("FEvals: ", backtrack.fevals)
    print("fRef shape ", np.shape(fRef))
    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)
    print("Delta f(x) = ", np.abs(bestF - fRef))
    optimList.append(xList)


print(np.shape(optimList))
# input()
ax.plot(xList[0,0], xList[0,1], 'bo', markersize=10, label='Starting point')
for xList in optimList:
    ax.plot(xList[0,0], xList[0,1], 'bo', markersize=10)
    ax.plot(xList[:,0], xList[:,1], 'x', markersize=7, label='Line Search', linestyle='--', linewidth=1)
# ax.plot(optimList, 'x', markersize=7, label='Line Search', linestyle='--', linewidth=1)

fig = plt.gcf()
ax.set_title("Steepest Descent with Backtracking over Contour Plot")
fig.set_size_inches(10, 10)
fig.legend(loc='right')
plt.show()
fig.savefig(dirs.figures+"Contour_Plot_BT"+".png", orientation='portrait', bbox_inches='tight')

# plt.clf()
# initialAlpha = 1
# plot_line_search(alphaList, initialAlpha, initialX, grad(function)(initialX), function, fig_name="L2_Q2_Line_search_plot_BT", save=False, show=True)
