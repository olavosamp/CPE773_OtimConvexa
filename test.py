import numpy as np

from algorithms import *#(DichotomousSearch, FibonacciSearch,GoldenSectionSearch,
                        #QuadraticInterpolation)
from functions import poly1, quadratic

def print_results(alg, string):
    print(string)
    xOpt = alg.optimize()
    print("x* = ", xOpt)
    print("Fevals: ", alg.fevals)
    print("xtol: ", alg.xtol)


xtol = 1e-5
maxIters = 1000
string = "\nOptimize f(x) = -5x5 + 4x4 - 12x3 + 11x2 - 2x + 1\nSolution x = 0.10986"

# Dichotomous Search
interval = [-0.5, 0.5]
dichot = DichotomousSearch(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(dichot, "\nDichotomous Search"+string)

# Fibonacci Search
fib = FibonacciSearch(poly1, interval, xtol=xtol, maxIters=maxIters )
# print(fib.compute_fibonacci(10))
print_results(fib, "\nFibonacci Search"+string)

# Golden Section Search
golden = GoldenSectionSearch(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(golden, "\nGolden Section Search"+string)

# Quadratic Interpolation
quadratic = QuadraticInterpolation(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(quadratic, "\nQuadratic Interpolation"+string)

# Cubic Interpolation
cubic = CubicInterpolation(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(cubic, "\nCubic Interpolation"+string)

# Davies-Swann-Campey Algorithm
dscAlg = DSCAlgorithm(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(dscAlg, "\nDavies-Swann-Campey Algorithm Interpolation"+string)


# Algorithm
# interval = [,]
# alg = ALGCLASS(poly1, interval, xtol=xtol, maxIters=maxIters )
# print_results(alg, string)
