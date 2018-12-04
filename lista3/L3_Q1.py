import pandas              as pd

import autograd.numpy      as np
# import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func7, func8
from libs.conjugate_direction import *

ftol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func7
interval   = [-10, 10]
savePath = dirs.results+"L3_Q1.xls"

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

b = -np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])

xLen = 16
initialX = np.random.uniform(interval[0], interval[1], size=(xLen))

times = 5
deltaFSum = []
for _ in range(times):
    # Q 6.1
    cd_alg = ConjugateGradient(function, initialX, interval=interval, ftol=ftol,
                                maxIters=maxIters, maxItersLS=maxItersLS)
    xOpt, fOpt, fevals = cd_alg.optimize()

    print("Initial X:", initialX)
    print("f(x*): ", fOpt)
    print("x*: ", xOpt)
    print("FEvals: ", fevals)

    # xRef   = np.zeros(xLen)
    xRef = -np.linalg.inv(Q) @ b
    fRef   = func7(xRef)
    deltaF = np.abs(fOpt - fRef)

    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)
    print("Delta f(x) = ", deltaF)
    deltaFSum.append(deltaF)
print("Average Delta F ", np.mean(deltaFSum))
print("Variance DeltaF ", np.var(deltaFSum))
