from autograd import grad, hessian
import autograd.numpy as np
from copy import copy

from libs.line_search import FletcherILS, BacktrackingLineSearch

def steepest_descent(func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
    k         = 0
    fevals    = 0
    maxIters  = int(maxIters)
    grad_func = grad(func)
    direction = np.zeros((maxIters, 2))
    x         = np.zeros((maxIters, 2))
    alpha     = np.zeros(maxIters)

    x[0] = initialX

    while k < maxIters-1:
        print("\nDescent Iter: ", k)
        print("x", x[k])
        direction[k] = -grad_func(x[k])
        print("direction", direction[k])

        lineSearch = BacktrackingLineSearch(func, interval, xtol=xtol, maxIters=maxItersLS, initialX=x[k], initialDir=direction[k])
        lineSearch.optimize()
        fevals  += lineSearch.fevals
        alpha[k] = lineSearch.alphaList[-1]
        print("alpha", alpha[k])

        x[k+1] = x[k] + alpha[k]*direction[k]
        print("x[k+1]", x[k+1])
        # input()

        if np.isnan(x[k]).any() or np.isnan(direction[k]).any():
            print("\nDescent diverged.")
            print("x: ", x[k])
            print("Grad(x): ", direction[k])
            return x[k], func(x[k]), None

        if np.linalg.norm(alpha[k]*direction[k], ord=2) < xtol:
            xOpt = x[k]
            return xOpt, func(x[k]), fevals
        else:
            k += 1
    print("\nAlgorithm did not converge.")
    return x[-1], func(x[-1]), fevals


def analytical_steepest_descent(func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
    k         = 1
    fevals    = 0
    maxIters  = int(maxIters)
    grad_func = grad(func)
    direction = np.zeros((maxIters, 2))
    x         = np.zeros((maxIters, 2))
    alpha     = np.zeros(maxIters)

    x[1]     = initialX
    alpha[0] = 1

    while k < maxIters-1:
        print("\nDescent Iter: ", k)
        print("x", x[k])

        # Search direction
        gradient     = grad_func(x[k])
        fevals += 1

        direction[k] = -gradient
        print("direction", direction[k])

        alphaProbe = alpha[k-1]
        fProbe     = func(x[k] - alphaProbe*gradient)
        fevals += 1

        f_k = func(x[k])
        fevals += 1

        # Compute optimal alpha
        alpha[k] = (np.transpose(gradient) @ gradient * alphaProbe**2)/(2*(fProbe - f_k + alphaProbe*np.transpose(gradient) @ gradient))
        print("alpha", alpha[k])

        # Update position
        x[k+1] = x[k] + alpha[k]*direction[k]
        print("x[k+1]", x[k+1])

        # Check for bad x or direction values
        if np.isnan(x[k]).any() or np.isnan(direction[k]).any():
            print("\nDescent diverged.")
            print("x: ", x[k])
            print("Grad(x): ", direction[k])
            return x[k], func(x[k]), None

        # Check stopping conditions
        if np.linalg.norm(alpha[k]*direction[k], ord=2) < xtol:
            xOpt = x[k]
            return xOpt, func(x[k]), fevals
        else:
            k += 1

    print("\nAlgorithm did not converge.")
    return x[-1], func(x[-1]), fevals
