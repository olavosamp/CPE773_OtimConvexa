import pandas              as pd

import autograd.numpy      as np
# import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func8
from libs.conjugate_direction import *
from libs.gradient_methods    import *


xtol       = 1e-6
maxIters   = 500
maxItersLS = 200
function   = func8
interval   = [-1e2, 1e2]
savePath = dirs.results+"L3_Q2.xls"


# Q 6.2
initialXList = [[+2., -2.],
                [-2., +2.],
                [-2., -2.]]
# initialXList = [[-2., +2.]]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []

for initialX in initialXList:
    # input()
    sd_algorithm = FletcherReeves(function, initialX, interval=interval, xtol=xtol,
                                     maxIters=maxIters, maxItersLS=maxItersLS)
    # sd_algorithm = FletcherReeves(function, initialX, interval=interval, xtol=xtol,
    #                              maxIters=maxIters, maxItersLS=maxItersLS)
    xOpt, fOpt, fevals = sd_algorithm.optimize()

    print("Initial X:", initialX)
    print("f(x*): ", fOpt)
    print("x*: ", xOpt)
    print("FEvals: ", fevals)

    xRef = np.array([1, 1])
    fRef = func8(xRef)
    deltaF = np.abs(fOpt - fRef)

    print("Delta f(x) = ", deltaF)
    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)

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
