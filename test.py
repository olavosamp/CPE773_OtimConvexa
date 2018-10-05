import numpy as np

from algorithms import DichotomousSearch
from functions import poly1, quadratic

# f1 = function_evaluator(test_polynomial)
#
# x = 1.2
# ans = f1.eval(x)
# print(ans)
# print(f1.fevals)
interval = [-0.5, 0.5]
maxIters = 1000
xtol = 1e-5
alg = DichotomousSearch(poly1, interval, xtol=xtol, maxIters=maxIters )

# print("Evaluate one time")
# x = 2
# print(alg.evaluate(x))
# print(alg.fevals)
# print()
# print(alg.iteration())
print("Optimize f(x) = −5x5 + 4x4 − 12x3 + 11x2 − 2x + 1")
print("x* = ", alg.optimize())
print("Fevals: ", alg.fevals)
print("xtol: ", alg.xtol)
