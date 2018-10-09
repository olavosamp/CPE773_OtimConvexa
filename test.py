import numpy as np

from algorithms import DichotomousSearch, FibonacciSearch
from functions import poly1, quadratic

def print_results(alg, string):
    print(string)
    xOpt = alg.optimize()
    print("x* = ", xOpt)
    print("Fevals: ", alg.fevals)
    print("xtol: ", alg.xtol)


xtol = 1e-5
maxIters = 1000
string = "\nOptimize f(x) = −5x5 + 4x4 − 12x3 + 11x2 − 2x + 1\nSolution x = 0.10986"

# Dichotomous Search
interval = [-0.5, 0.5]
dichot = DichotomousSearch(poly1, interval, xtol=xtol, maxIters=maxIters )
print_results(dichot, "\nDichotomous Search"+string)

# Fibonacci Search
# interval = [-0.5, 0.5]
fib = FibonacciSearch(poly1, interval, xtol=xtol, maxIters=maxIters )
# print(fib.compute_fibonacci(10))
print_results(fib, "\nFibonacci Search"+string)

# Algorithm
# interval = [,]
# alg = ALGCLASS(poly1, interval, xtol=xtol, maxIters=maxIters )
# print_results(alg, string)
