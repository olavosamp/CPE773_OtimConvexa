import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs           as dirs
from libs.functions        import func5
from libs.gradient_methods import *


xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func5
interval   = [-1e15, 1e15]
savePath = dirs.results+"L2_Q4.xls"


# Q 5.17
initialX = [+4, +4]
# initialX = [+4, -4]
# initialX = [-4, +4]
# initialX = [-4, -4]
initialXList = [[+4., +4.],
                [+4., -4.],
                [-4., +4.],
                [-4., -4.],]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []

for initialX in initialXList:
    sd_algorithm = NewtonRaphson(function, initialX, interval=interval, xtol=xtol,
                                 maxIters=maxIters, maxItersLS=maxItersLS)
    xOpt, fOpt, fevals = sd_algorithm.optimize()

    print("Initial X:", initialX)
    print("f(x*): ", fOpt)
    print("x*: ", xOpt)
    print("FEvals: ", fevals)

    optimResult = spo.minimize(function, initialX, method='BFGS', tol=xtol)
    xRef = optimResult.x
    fRef = optimResult.fun
    deltaF = np.abs(fOpt - fRef)

    print("Delta f(x) = ", deltaF)
    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)
    print("CondVal: ", sd_algorithm.condVal)

    xList.append(xOpt)
    fxList.append(fOpt)
    fevalsList.append(fevals)
    deltaFList.append(deltaF)

xList = np.array(xList)
resultsDf = pd.DataFrame(data={"Initial x": initialXList,
                                "x*_1"    : xList[:, 0],
                                "x*_2"    : xList[:, 1],
                                "f(x*)"   : fxList,
                                "FEvals"  : fevalsList,
                                "Delta F" : deltaFList})

print(resultsDf)
resultsDf.to_excel(savePath)
