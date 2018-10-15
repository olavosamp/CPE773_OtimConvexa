import time
import numpy  as np
import pandas as pd
from   tqdm   import tqdm

import dirs
from algorithms import *#(DichotomousSearch, FibonacciSearch,GoldenSectionSearch,
                        #QuadraticInterpolation)
from functions import poly1

algList = [(DichotomousSearch, "Dichotomous Search"),
            (FibonacciSearch, "Fibonacci Search"),
            (GoldenSectionSearch, "Golden Section Search"),
            (QuadraticInterpolation, "Quadratic Interpolation"),
            (CubicInterpolation, "Cubic Interpolation"),
            (DSCAlgorithm, "Davies Swann Campey Algorithm"),
            ]

algListStochastic = [(BacktrackingLineSearch, "Backtracking Line Search")]

targetFunction  = poly1
maxIters        = int(1e3)
interval        = [-0.5, 0.5]
xtol            = 1e-5
xSolution       = 0.10986
runtimeEvals    = 10
stochasticEvals = 500
savePath = dirs.results+"L1_Q1_"

# Compute Deterministic Algorithms
resultsList = []
for algData in algList:
    dataDict = dict()

    alg     = algData[0](targetFunction, interval, xtol=xtol, maxIters=maxIters)
    algName = algData[1]
    print("\nRunning ", algName)

    optimum = alg.optimize()
    fevals  = alg.fevals
    dataDict["Algorithm"] = algName
    dataDict["Delta X"]   = np.abs(xSolution - optimum)
    dataDict["FEvals"]    = fevals

    runtimeList = np.empty(runtimeEvals)
    for i in range(runtimeEvals):
        start = time.perf_counter()
        alg.optimize()
        end  = time.perf_counter()
        runtimeList[i] = end - start

    dataDict["Runtime"]   = np.mean(runtimeList)
    dataDict["Success Rate"] = '-'
    resultsList.append(dataDict)


# Compute Results for Stochastic Algorithms
for algData in algListStochastic:
    dataDict = dict()

    alg     = algData[0](targetFunction, interval, xtol=xtol, maxIters=maxIters)
    algName = algData[1]
    print("\nRunning ", algName)

    # stochList is a list of [xOptimum, fevals, runtime]
    stochList = np.zeros((stochasticEvals, 3))
    for i in tqdm(range(stochasticEvals)):
        start = time.perf_counter()
        stochList[i, 0] = alg.optimize()
        end  = time.perf_counter()

        stochList[i, 1]  = alg.fevals
        stochList[i, 2] = end - start

    dataDict["Algorithm"] = algName

    deltaX = np.abs(xSolution - stochList[:, 0])
    mask   = deltaX < xtol
    dataDict["Delta X"]   = np.mean(deltaX)
    dataDict["FEvals"]    = np.mean(stochList[mask, 1])

    dataDict["Runtime"]   = np.mean(runtimeList)
    dataDict["Success Rate"] = np.mean(mask)
    resultsList.append(dataDict)

results = pd.DataFrame(resultsList)
print(results)

savePath += "results_table.xlsx"
print("Table1 saved at\n{}\n".format(savePath))
results.to_excel(savePath, float_format="%.6f")
