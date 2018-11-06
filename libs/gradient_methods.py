from autograd import grad, hessian
import autograd.numpy as np
from copy import copy

from libs.line_search import FletcherILS, BacktrackingLineSearch

class SteepestDescent:
    def __init__(self, func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
        self.costFunc   = func
        self.gradFunc   = grad(self.evaluate)
        self.maxIters   = int(maxIters)
        self.maxItersLS = maxItersLS
        self.direction  = np.zeros((maxIters, 2))
        self.interval   = interval
        self.alpha      = np.zeros(maxIters)
        self.xtol       = xtol
        self.x          = np.zeros((maxIters, 2))

        self.x[0] = initialX

    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result

    def line_search(self):
        pass

    def optimize(self):
        self.fevals   = 0
        self.gradFunc = grad(self.evaluate)
        self.k = 0
        while self.k < self.maxIters-1:
            # Update search direction
            self.gradient          = self.gradFunc(self.x[self.k])
            self.direction[self.k] = -self.gradient

            # Compute new alpha via line search
            self.alpha[self.k] = self.line_search()

            # Update search position
            self.x[self.k+1] = self.x[self.k] + self.alpha[self.k]*self.direction[self.k]

            ## Debug
            # print("\nIter ", self.k)
            # print("x[k]",    self.x[self.k] )
            # print("alpha[k]",self.alpha[self.k] )
            # print("dir[k]",  self.direction[self.k] )
            # print("x[k+1]",  self.x[self.k+1] )
            # print("f(x)", self.evaluate(self.x[self.k+1] ))

            # Check for bad x or direction values
            if np.isnan(self.x[self.k]).any() or np.isnan(self.direction[self.k]).any():
                print("\nDescent diverged.")
                print("x: ", self.x[self.k])
                print("Grad(x): ", self.direction[self.k])
                return self.x[self.k], self.costFunc(self.x[self.k]), self.fevals

            # Check for stopping conditions
            if np.linalg.norm(self.alpha[self.k]*self.direction[self.k], ord=2) < self.xtol:
                self.xOpt = self.x[self.k]
                return self.xOpt, self.costFunc(self.xOpt), self.fevals
            else:
                self.k += 1

        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[-1]
        return self.xOpt, self.costFunc(self.x[-1]), self.fevals


class SteepestDescentBacktracking(SteepestDescent):
    def line_search(self):
        t = 1
        self.iter2 = 0
        self.alphaParam = 0.5
        self.betaParam  = 0.9
        self.fx = self.evaluate(self.x[self.k])

        while (self.evaluate(self.x[self.k] + t*self.direction[self.k]) > self.fx + self.alphaParam*t*(np.transpose(self.gradient) @ self.direction[self.k])) and (self.iter2 < self.maxItersLS):
            t = self.betaParam*t
            t = np.clip(t, 2*self.xtol, None)
            self.iter2 += 1

        self.alpha[self.k] = t
        return t

class SteepestDescentAnalytical(SteepestDescent):
    def line_search(self):
        if self.k == 0:
            alphaProbe = 1
        else:
            alphaProbe = self.alpha[self.k-1]

        fProbe  = self.evaluate(self.x[self.k] - alphaProbe*self.gradient)
        self.fx = self.evaluate(self.x[self.k])

        # Compute optimal alpha
        optimalAlpha = (np.transpose(self.gradient) @ self.gradient * alphaProbe**2)/(2*(fProbe - self.fx + alphaProbe*np.transpose(self.gradient) @ self.gradient))
        self.alpha[self.k] = optimalAlpha
        # print("alpha", self.alpha[self.k])
        return optimalAlpha

def steepest_descent(func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
    k         = 0
    fevals    = 0
    grad_func = grad(func)
    maxIters  = int(maxIters)
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
