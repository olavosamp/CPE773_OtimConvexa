from autograd         import grad, hessian, jacobian
import autograd.numpy as np

from libs.operators        import positive_definite
from libs.line_search      import FletcherILS, BacktrackingOptim

class SteepestDescent:
    def __init__(self, func, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
        self.costFunc   = func
        self.gradFunc   = grad(self.evaluate)
        self.maxIters   = int(maxIters)
        self.maxItersLS = maxItersLS
        self.interval   = interval
        self.fevals     = 0
        self.alpha      = np.zeros(maxIters)
        self.ftol       = ftol

        xShape = np.shape(initialX)[0]
        self.direction  = np.zeros((maxIters, xShape))
        self.x          = np.zeros((maxIters, xShape))
        self.x[0] = initialX


    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result

    def line_search(self):
        pass

    def get_direction(self, x):
        self.gradient          = self.gradFunc(x)
        return -self.gradient

    def stopping_cond(self):
        self.condVal = np.linalg.norm(self.alpha[self.k]*self.direction[self.k], ord=2)
        return self.condVal < self.ftol

    def check_bad_values(self):
        # Check for bad x or direction values
        if np.isnan(self.x[self.k]).any() or np.isnan(self.direction[self.k]).any():
            print("\nDescent algorithm diverged.")
            print("x: ", self.x[self.k])
            print("Grad(x): ", self.direction[self.k])

            return self.x[self.k], self.costFunc(self.x[self.k]), self.fevals


    def optimize(self):
        self.fevals   = 0
        self.gradFunc = grad(self.evaluate)
        self.k = 0
        while self.k < self.maxIters-1:
            # Update search direction
            self.direction[self.k] = self.get_direction(self.x[self.k])
            # return "DEBUG GAUSS-NEWTON"
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
            # input()

            # Check for NaN or Inf values
            self.check_bad_values()

            # Check for stopping conditions
            if self.stopping_cond() == True:
                print("\nStopping condition reached, descent algorithm terminating.")
                self.xOpt = self.x[self.k]
                return self.xOpt, self.costFunc(self.xOpt), self.fevals
            else:
                self.k += 1
            # print("CondVal: ", self.condVal)

        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[-1]
        return self.xOpt, self.costFunc(self.x[-1]), self.fevals


class SteepestDescentBacktracking(SteepestDescent):
    def line_search(self):
        t = 1
        self.iter2 = 0
        self.alphaParam = 0.5
        self.betaParam  = 0.7
        self.alphaMin   = 1e-10
        self.fx = self.evaluate(self.x[self.k])

        while (self.evaluate(self.x[self.k] + t*self.direction[self.k]) > self.fx + self.alphaParam*t*(np.transpose(self.gradient) @ self.direction[self.k])) and (self.iter2 < self.maxItersLS):
            t = self.betaParam*t
            # t = np.clip(t, self.alphaMin, None)
            self.iter2 += 1

        self.alpha[self.k] = t
        return t

    def check_bad_values(self):
        # Check if x or f(x) is Nan or inf - symptoms that the algorithm reached
        # the constraint barrier
        if np.isnan(self.x[self.k]).any() or np.isinf(self.x[self.k]).any() or \
        np.isnan(self.costFunc(self.x[self.k])).any() or np.isinf(self.costFunc(self.x[self.k])).any():
            self.alpha[self.k] *= 0.5

        return self.alpha[self.k]


class SteepestDescentAnalytical(SteepestDescent):
    def line_search(self):
        self.alphaMin   = 1e-10

        # print("Line search")
        if self.k == 0:
            alphaProbe = 1
        else:
            alphaProbe = self.alpha[self.k-1]

        fProbe  = self.evaluate(self.x[self.k] - alphaProbe*self.gradient)
        self.fx = self.evaluate(self.x[self.k])

        # Compute optimal alpha
        optimalAlpha = (np.transpose(self.gradient) @ self.gradient * alphaProbe**2)/(2*(fProbe - self.fx + alphaProbe*np.transpose(self.gradient) @ self.gradient))
        optimalAlpha = np.clip(optimalAlpha, self.alphaMin, None)
        self.alpha[self.k] = optimalAlpha
        # print("alpha", self.alpha[self.k])
        return optimalAlpha

class NewtonRaphson(SteepestDescentBacktracking):
    def compute_hessian(self, x):
        self.hessFunc = hessian(self.evaluate)
        hessVal = self.hessFunc(x)

        if positive_definite(hessVal) == True:
            beta = 1e15
        else:
            beta = 1e-15

        identity = np.eye(np.shape(hessVal)[0])
        hessModified = (hessVal + beta*identity)/(1 + beta)
        return hessModified

    def get_direction(self, x):
        self.gradient = self.gradFunc(x)
        self.hessian = self.compute_hessian(x)
        dir = -np.linalg.inv(self.hessian) @ self.gradient
        return dir

class GaussNewton(SteepestDescentAnalytical):
    def func_sum(self, x):
        fx = self.evaluate(x)
        self.fSum = np.transpose(fx) @ fx
        return self.fSum

    def get_direction(self, x):
        self.jacob_func = jacobian(self.evaluate)

        self.jacobian   = self.jacob_func(x)
        self.fx         = self.evaluate(x)
        self.gradient   = 2*np.transpose(self.jacobian) @ self.fx
        self.hessian    = np.transpose(self.jacobian) @ self.jacobian

        return np.zeros(4)
