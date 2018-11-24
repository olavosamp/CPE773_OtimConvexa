import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func9
from libs.quasi_newton        import *
from libs.gradient_methods    import *


xtol       = 3e-7
maxIters   = 500
maxItersLS = 200
function   = func9
interval   = [-1e2, 1e2]
savePath = dirs.results+"L3_Q5.xls"


# Q 7.8
initialXList = [[0,0]
]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
for initialX in initialXList:
    sd_algorithm = QuasiNewtonDFP(function, initialX, interval=interval, xtol=xtol,
                                     maxIters=maxIters, maxItersLS=maxItersLS)

    xOpt, fOpt, fevals = sd_algorithm.optimize()

    print("Initial X:", initialX)
    print("f(x*): ", fOpt)
    print("x*: ", xOpt)
    print("FEvals: ", fevals)

    optimResult = spo.minimize(function, initialX, method='BFGS', tol=xtol)
    xRef = optimResult.x
    fRef = optimResult.fun
    fevalRef = optimResult.nfev
    deltaF = np.abs(fOpt - fRef)

    print("Delta f(x) = ", deltaF)
    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)
    print("Ref FEvals = ", fevalRef)

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
