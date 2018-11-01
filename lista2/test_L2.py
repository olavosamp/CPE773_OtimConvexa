import autograd.numpy as np

from libs.functions import func5
from libs.gradient_methods import steepest_descent


xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func5
interval   = [-1e15, 1e15]

initialX = [+4, +4]
# initialX = [+4, -4]
# initialX = [-4, +4]
# initialX = [-4, -4]

xOpt, fOpt, fevals = steepest_descent(function, initialX, interval=interval, xtol=xtol, maxIters=maxIters, maxItersLS=maxItersLS)

print("Optimal X: ", xOpt)
print("f(x*): ", fOpt)
print("FEvals: ", fevals)
