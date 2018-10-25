import autograd.numpy as np
from scipy.optimize import brute

from line_search import FletcherILS
from functions import func4

def print_results(alg, string):
    print(string)
    xOpt = alg.optimize()
    print("x* = ", xOpt)
    print("Fevals: ", alg.fevals)
    print("xtol: ", alg.xtol)


xtol = 1e-5
maxIters = 500

targetFunction = func4
interval = np.array(((-np.pi, -np.pi), (np.pi, np.pi)))


# # Brute force
# print("Brute Force"+string)
# xOpt, fevals, _, Jout = brute(targetFunction, [(interval[0], interval[1])], Ns=round(1/xtol), full_output=True)
# solution = xOpt[0]
# print("x* = ", solution)
# print("Fevals: ", len(Jout))
# print("Ns: ", round(1/xtol))
# print("xtol: ", xtol)
# print(Jout)

# Fletcher's Inexact Line Search
print("\nFletcher's Inexact Line Search")
initialX = np.array([np.pi, -np.pi])
initalDir = np.array([1.0, -1.3])

fletcher = FletcherILS(func4, interval, xtol=xtol, maxIters=maxIters, initialX=initialX)
xOpt = fletcher.optimize()
print("x* = ", xOpt)
print("Fevals: ", fletcher.fevals)
print("xtol: ", xtol)

# # Fletcher's Inexact Line Search (Stochastic)
# print("\nFletcher's Inexact Line Search")
# evaluations = 300
# results = np.empty(evaluations)
# fevals = np.empty(evaluations)
# # solution = 0.10986
# for i in range(evaluations):
#     fletcher = FletcherILS(targetFunction, interval, xtol=xtol, maxIters=maxIters)
#     results[i] = fletcher.optimize()
#     fevals[i]  = fletcher.fevals
#
# mask = np.abs(results - solution) < xtol
# sr = np.mean(np.where(mask, 1, 0))
# meanFevals = np.mean(fevals)
# meanFevalsSuccess = np.mean(fevals[mask])
#
# print("SR: {:.2f}".format( sr))
# print("Fevals: {:.2f}".format( meanFevals))
# print("Fevals Succ: {:.2f}".format( meanFevalsSuccess))
# print(results[mask])
