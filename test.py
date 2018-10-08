import numpy as np

from algorithms import DichotomousSearch
from functions import poly1, quadratic

interval = [-0.5, 0.5]
maxIters = 1000
xtol = 1e-5
alg = DichotomousSearch(poly1, interval, xtol=xtol, maxIters=maxIters )

print("\nOptimize f(x) = −5x5 + 4x4 − 12x3 + 11x2 − 2x + 1\n")
xOpt = alg.optimize()
print("x* = ", xOpt)
print("Fevals: ", alg.fevals)
print("xtol: ", alg.xtol)
