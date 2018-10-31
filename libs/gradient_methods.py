from autograd import grad, hessian
import autograd.numpy as np
from copy import copy

from libs.line_search import FletcherILS, BacktrackingLineSearch

def steepest_descent(func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3):
    k         = 0
    fevals    = 0
    maxIters  = int(maxIters)
    grad_func = grad(func)
    direction = np.zeros((maxIters, 2))
    x         = np.zeros((maxIters, 2))
    alpha     = np.zeros(maxIters)

    x[0] = initialX

    while k < maxIters-1:
        direction[k] = -grad_func(x[k])

        lineSearch = BacktrackingLineSearch(func, interval, xtol=xtol, maxIters=maxIters, initialX=x[k], initialDir=direction[k])
        lineSearch.optimize()
        fevals  += lineSearch.fevals
        alpha[k] = lineSearch.alphaList[-1]

        x[k+1] = x[k] + alpha[k]*direction[k]

        if np.linalg.norm(alpha[k]*direction[k], ord=2) < xtol:
            xOpt = x[k]
            return xOpt, fevals
        else:
            k += 1
    print("Algorithm did not converge.")
    return x[-1], fevals
