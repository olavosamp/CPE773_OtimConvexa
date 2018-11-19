import pandas              as pd

import autograd.numpy      as np
# import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func8
from libs.conjugate_direction import *


xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func8
interval   = [-1e15, 1e15]
savePath = dirs.results+"L3_Q2.xls"


# Q 6.2
initialXList = [[+2., -2.],
                [-2., +2.],
                [-2., -2.],]

b = -np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])

Q1 = np.array([[12,8,7,6],
               [8,12,8,7],
               [7,8,12,8],
               [6,7,8,12],
])
Q2 = np.array([[3,2,1,0],
               [2,3,2,1],
               [1,2,3,2],
               [0,1,2,3],
])
Q3 = np.array([[2,1,0,0],
               [1,2,1,0],
               [0,1,2,1],
               [0,0,1,2],
])
Q4 = np.eye(4)

Q = np.block([[Q1,Q2,Q3,Q4],
              [Q2,Q1,Q2,Q3],
              [Q3,Q2,Q1,Q2],
              [Q4,Q3,Q2,Q1],
])

xList      = []
fxList     = []
fevalsList = []
deltaFList = []

for initialX in initialXList:
    sd_algorithm = FletcherReeves(function, initialX, b, Q, interval=interval, xtol=xtol,
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
