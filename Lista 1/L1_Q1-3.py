import time
import autograd.numpy  as np
import pandas as pd
from   tqdm   import tqdm
from scipy.optimize import brute

import dirs
from line_search import *#(DichotomousSearch, FibonacciSearch,GoldenSectionSearch,
                        #QuadraticInterpolation)
from functions import poly1, func2, func3

algList = [(brute, "Brute Force"),
            (DichotomousSearch, "Dichotomous Search"),
            (FibonacciSearch, "Fibonacci Search"),
            (GoldenSectionSearch, "Golden Section Search"),
            (QuadraticInterpolation, "Quadratic Interpolation"),
            (CubicInterpolation, "Cubic Interpolation"),
            (DSCAlgorithm, "Davies Swann Campey Algorithm")
            ]

algListStochastic = [(BacktrackingLineSearch, "Backtracking Line Search")]

targetFunction  = poly1
interval        = [-0.5, 0.5]
savePath = dirs.results+"L1_Q_4-2_"

# targetFunction  = func2
# interval        = [6., 9.9]
# savePath = dirs.results+"L1_Q_4-3_"

# targetFunction = func3
# interval        = [0, 2*np.pi]
# savePath = dirs.results+"L1_Q_4-4_"

maxIters        = int(1e3)
xtol            = 1e-5
runtimeEvals    = 10
stochasticEvals = 500


# Compute Deterministic Algorithms
resultsList = []
for algData in algList:
    dataDict = dict()

    algName = algData[1]
    print("\nRunning ", algName)
    if algName == "Brute Force":
        numSteps = round(np.abs(interval[0] - interval[1])/xtol)
        optimum, _, _, Jout = brute(targetFunction, [(interval[0], interval[1])], Ns=numSteps, full_output=True)
        optimum   = optimum[0]
        xSolution = optimum
        fevals = len(Jout)
    else:
        alg = algData[0](targetFunction, interval, xtol=xtol, maxIters=maxIters)
        optimum = alg.optimize()
        fevals  = alg.fevals
    print("Result: \nx* = ", optimum)

    dataDict["Algorithm"] = algName
    dataDict["Delta X"]   = np.abs(xSolution - optimum)
    dataDict["FEvals"]    = fevals

    runtimeList = np.empty(runtimeEvals)
    for i in tqdm(range(runtimeEvals)):
        start = time.perf_counter()
        if algName == "Brute Force":
            brute(poly1, [(interval[0], interval[1])], Ns=round(1/xtol), full_output=True)
        else:
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
    dataDict["Delta X"]   = np.mean(deltaX[mask]) # Succesful results only
    # dataDict["Delta X"]   = np.mean(deltaX) # All results
    dataDict["FEvals"]    = np.mean(stochList[mask, 1])

    dataDict["Runtime"]   = np.mean(runtimeList)
    dataDict["Success Rate"] = np.mean(mask)
    resultsList.append(dataDict)

results = pd.DataFrame(resultsList)
print(results)

savePath += "results_table.xlsx"
print("Table1 saved at\n{}\n".format(savePath))
results.to_excel(savePath)
