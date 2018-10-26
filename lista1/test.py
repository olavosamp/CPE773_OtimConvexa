import numpy as np

from libs.line_search import *#(DichotomousSearch, FibonacciSearch,GoldenSectionSearch,
                        #QuadraticInterpolation)
from libs.functions import poly1, func2, func3
from scipy.optimize import brute

def print_results(alg, string):
    print(string)
    xOpt = alg.optimize()
    print("x* = ", xOpt)
    print("Fevals: ", alg.fevals)
    print("xtol: ", alg.xtol)


xtol = 1e-5
maxIters = 1000
# Function 1
string = "\nOptimize f(x) = -5x5 + 4x4 - 12x3 + 11x2 - 2x + 1\nSolution x = 0.10986"
# interval = [-0.5, 0.5]
# targetFunction = func2
# interval = [6., 9.9]
targetFunction = func3
interval = [0, 2*np.pi]

# Brute force
print("Brute Force"+string)
xOpt, fevals, _, Jout = brute(targetFunction, [(interval[0], interval[1])], Ns=round(1/xtol), full_output=True)
solution = xOpt[0]
print("x* = ", solution)
print("Fevals: ", len(Jout))
print("Ns: ", round(1/xtol))
print("xtol: ", xtol)
print(Jout)

# Dichotomous Search
# dichot = DichotomousSearch(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# print_results(dichot, "\nDichotomous Search"+string)
#
# # Fibonacci Search
# fib = FibonacciSearch(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# # print(fib.compute_fibonacci(10))
# print_results(fib, "\nFibonacci Search"+string)
#
# # Golden Section Search
# golden = GoldenSectionSearch(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# print_results(golden, "\nGolden Section Search"+string)
#
# # Quadratic Interpolation
# quadratic = QuadraticInterpolation(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# print_results(quadratic, "\nQuadratic Interpolation"+string)

# Cubic Interpolation
cubic = CubicInterpolation(targetFunction, interval, xtol=xtol, maxIters=maxIters )
print_results(cubic, "\nCubic Interpolation"+string)

# # Davies-Swann-Campey Algorithm
# dscAlg = DSCAlgorithm(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# print_results(dscAlg, "\nDavies-Swann-Campey Algorithm"+string)

# Backtracking Line Search
print("\nBacktracking Line Search")
evaluations = 300
results = np.empty(evaluations)
fevals = np.empty(evaluations)
# solution = 0.10986
for i in range(evaluations):
    backtrack = BacktrackingLineSearch(targetFunction, interval, xtol=xtol, maxIters=maxIters,
                                        alpha=0.01, beta=0.5)
    results[i] = backtrack.optimize()
    fevals[i]  = backtrack.fevals

mask = np.abs(results - solution) < xtol
sr = np.mean(np.where(mask, 1, 0))
meanFevals = np.mean(fevals)
meanFevalsSuccess = np.mean(fevals[mask])

print("SR: {:.2f}".format( sr))
print("Fevals: {:.2f}".format( meanFevals))
print("Fevals Succ: {:.2f}".format( meanFevalsSuccess))
print(results[mask])


# Backtracking Line Search
print("\nFletcher's Inexact Line Search")
evaluations = 300
results = np.empty(evaluations)
fevals = np.empty(evaluations)
# solution = 0.10986
for i in range(evaluations):
    fletcher = FletcherILS(targetFunction, interval, xtol=xtol, maxIters=maxIters)
    results[i] = fletcher.optimize()
    fevals[i]  = fletcher.fevals

mask = np.abs(results - solution) < xtol
sr = np.mean(np.where(mask, 1, 0))
meanFevals = np.mean(fevals)
meanFevalsSuccess = np.mean(fevals[mask])

print("SR: {:.2f}".format( sr))
print("Fevals: {:.2f}".format( meanFevals))
print("Fevals Succ: {:.2f}".format( meanFevalsSuccess))
print(results[mask])

# Algorithm
# interval = [,]
# alg = ALGCLASS(targetFunction, interval, xtol=xtol, maxIters=maxIters )
# print_results(alg, string)
