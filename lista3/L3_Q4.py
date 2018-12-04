import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func8
from libs.quasi_newton        import *
# from libs.gradient_methods    import *


ftol       = 1e-6
maxIters   = 500
maxItersLS = 200
function   = func8
interval   = [-1e1, 1e1]
savePath = dirs.results+"L3_Q4.xls"


# Q 7.7
# initialXList = [np.random.uniform(low=interval[0], high=interval[1], size=2),
#                 np.random.uniform(low=interval[0], high=interval[1], size=2),
#                 np.random.uniform(low=interval[0], high=interval[1], size=2)]

initialXList = [[+2., -2.],
                [-2., +2.],
                [-2., -2.]]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
for initialX in initialXList:
    # print(initialX.shape)
    # input()
    sd_algorithm = QuasiNewtonDFP(function, initialX, interval=interval, ftol=ftol,
                                     maxIters=maxIters, maxItersLS=maxItersLS)

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
