import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func9
from libs.conjugate_direction import *
from libs.gradient_methods    import *


xtol       = 3e-7
maxIters   = 500
maxItersLS = 200
function   = func9
interval   = [-1e2, 1e2]
savePath = dirs.results+"L3_Q3.xls"


# Q 6.3
# initialXList = [[1, 1]]
initialX = [1, 1]
maxItersList = [1500]

initialXList = []

xListSD        = []
fxListSD       = []
fevalsListSD   = []
deltaFListSD   = []

xListCG        = []
fxListCG       = []
fevalsListCG   = []
deltaFListCG   = []

for maxIters in maxItersList:
    # input()
    cg_algorithm = ConjugateGradient(function, initialX, interval=interval, xtol=xtol,
                                     maxIters=maxIters, maxItersLS=maxItersLS)
    # sd_algorithm = FletcherReeves(function, initialX, interval=interval, xtol=xtol,
    #                              maxIters=maxIters, maxItersLS=maxItersLS)
    xOptCG, fOptCG, fevalsCG = cg_algorithm.optimize()

    sd_algorithm = SteepestDescentAnalytical(function, initialX, interval=interval, xtol=xtol,
                                            maxIters=maxIters, maxItersLS=maxItersLS)
    xOptSD, fOptSD, fevalsSD = sd_algorithm.optimize()
    print(sd_algorithm.k)
    input()

    print("Initial X:", initialX)
    print("\nConjugateGradient")
    print("f(x*): ", fOptCG)
    print("x*: ", xOptCG)

    print("\nSteepestDescent")
    print("FEvals: ", fevalsCG)
    print("f(x*): ", fOptSD)
    print("x*: ", xOptSD)
    print("FEvals: ", fevalsSD)

    optimResult = spo.minimize(function, initialX, method='BFGS', tol=xtol)
    xRef = optimResult.x
    fRef = optimResult.fun

    deltaFSD = np.abs(fOptSD - fRef)
    deltaFCG = np.abs(fOptCG - fRef)

    print("Delta f(x) = ", deltaFCG)
    # print("Ref x* = ", xRef)
    # print("Ref f(x*) = ", fRef)

    xListSD.append(xOptSD)
    fxListSD.append(fOptSD)
    fevalsListSD.append(fevalsSD)
    deltaFListSD.append(deltaFSD)

    xListCG.append(xOptCG)
    fxListCG.append(fOptCG)
    fevalsListCG.append(fevalsCG)
    deltaFListCG.append(deltaFCG)

    initialXList.append(initialX)

xListSD = np.array(xListSD)
xListCG = np.array(xListCG)

runData = { "Iteration"                  : maxItersList,
            "Initial x"                  : initialXList,
            "f(x*) SteepestDescent"      : fxListSD,
            "f(x*) ConjugateGradient"    : fxListCG,
            "FEvalsSD"                   : fevalsListSD,
            "FEvalsCG"                   : fevalsListCG,
            "Delta F SD"                 : deltaFListSD,
            "Delta F CG"                 : deltaFListCG}

resultsDf = pd.DataFrame(data=runData)

print(resultsDf)
resultsDf.to_excel(savePath)
