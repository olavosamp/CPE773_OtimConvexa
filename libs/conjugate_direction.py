from tqdm             import tqdm
from autograd         import grad, hessian, jacobian
import autograd.numpy as np

from libs.operators        import positive_definite
from libs.line_search      import *


class ConjugateDirection:
    def __init__(self, func, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
        self.costFunc   = func
        self.gradFunc   = grad(self.evaluate)
        self.hessFunc   = hessian(self.evaluate)
        self.maxIters   = int(maxIters)
        self.maxItersLS = maxItersLS
        self.interval   = interval
        self.fevals     = 0
        self.ftol       = ftol

        self.xLen       = np.shape(initialX)[0]
        self.direction  = np.zeros((self.maxIters, self.xLen))
        self.x          = np.zeros((self.maxIters, self.xLen))
        self.x[0]       = initialX

        # self.gradFunc   = grad(self.evaluate)


    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result


    def line_search(self, x):
        def funcLS(alpha):
            return self.evaluate(x + alpha*self.direction[self.iter])

        lineSearch = FibonacciSearch(funcLS, self.interval, ftol=self.ftol, maxIters=self.maxItersLS)
        # lineSearch = BacktrackingOptim(funcLS, self.interval, ftol=self.ftol, maxIters=self.maxItersLS)
        self.alpha[self.iter] = lineSearch.optimize()
        self.xLS = self.x[self.iter] + self.alpha[self.iter]*self.direction[self.iter]
        self.fevals += lineSearch.fevals
        return self.xLS


    def get_direction(self, x):
        pass


    def optimize(self):
        pass


class ConjugateGradient(ConjugateDirection):
    def __init__(self, func, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
        super().__init__(func, initialX, interval=interval, ftol=ftol, maxIters=maxIters, maxItersLS=maxItersLS)

        self.hessFunc   = hessian(self.evaluate)

        self.g          = np.zeros((self.maxIters, self.xLen))
        self.direction  = np.zeros((self.maxIters, self.xLen))
        self.alpha      = np.zeros((self.maxIters))
        self.hessVal    = np.zeros((self.maxIters, self.xLen, self.xLen))

        self.b          = self.gradFunc(np.zeros(self.xLen))
        self.H          = self.hessFunc(np.zeros(self.xLen))


    def get_direction(self):
        self.g[self.iter+1] = self.b + self.H @ self.x[self.iter+1]
        self.beta = (np.transpose(self.g[self.iter+1]) @ self.g[self.iter+1])/(np.transpose(self.g[self.iter]) @ self.g[self.iter])

        newDirection = -self.g[self.iter+1] + self.beta*self.direction[self.iter]
        return newDirection


    def stopping_cond(self):
        norm2 = np.linalg.norm(self.alpha[self.iter]*self.direction[self.iter], ord=2)
        return norm2 < self.ftol


    def line_search(self):
        self.xLS = self.x[self.iter] + self.alpha[self.iter]*self.direction[self.iter]
        return self.xLS


    def optimize(self):
        self.iter = 0

        # Initial values for g and d
        self.g[0] = self.b + self.H @ self.x[0]
        self.direction[0] = -self.g[0]
        while self.iter < self.maxIters-2:
            self.hessVal[self.iter] = self.hessFunc(self.x[self.iter])

            # Check if gradient is zero. If true, skip to stopping condition check
            if not (self.g[self.iter] == 0).all():
                self.alpha[self.iter] = (np.transpose(self.g[self.iter]) @ self.g[self.iter])/ \
                (np.transpose(self.direction[self.iter]) @ self.hessVal[self.iter] @ self.direction[self.iter])

                self.x[self.iter+1] = self.line_search()
            else:
                self.x[self.iter+1] = self.x[self.iter]

            ## Debug
            # p|rint("\nIter", self.iter)
            # print("hessVal:\n", self.hessVal[self.iter])
            # print("g:\t", self.g[self.iter])
            # print("direction: ", self.direction[self.iter])
            # print("alpha[k]: ", self.alpha[self.iter])
            # print("x[k+1]: ", self.x[self.iter+1])
            # input()

            # Check stopping condition
            if self.stopping_cond() == True:
                print("\nStopping condition reached, algorithm terminating.")
                self.xOpt = self.x[self.iter+1]
                return self.xOpt, self.costFunc(self.xOpt), self.fevals

            # Compute new direction
            self.direction[self.iter+1] = self.get_direction()
            self.iter += 1

        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[-1]
        return self.xOpt, self.costFunc(self.x[-1]), self.fevals


class FletcherReeves(ConjugateGradient):
    def __init__(self, func, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
        super().__init__(func, initialX, interval=interval, ftol=ftol, maxIters=maxIters, maxItersLS=maxItersLS)

        self.restartIter = self.xLen
        # self.restartIter = 20

        self.g           = np.zeros((self.restartIter*self.maxIters, self.xLen))
        self.direction   = np.zeros((self.restartIter*self.maxIters, self.xLen))
        # self.hessVal     = np.zeros((self.restartIter*self.maxIters, self.xLen, self.xLen))

    def line_search(self, x):
        def funcLS(alpha):
            return self.evaluate(x + alpha*self.direction[self.iter])

        # lineSearch = BacktrackingOptim(funcLS, self.interval, ftol=self.ftol, maxIters=self.maxItersLS)
        lineSearch = FibonacciSearch(funcLS, self.interval, ftol=self.ftol, maxIters=self.maxItersLS)
        self.alpha[self.iter] = lineSearch.optimize()
        self.xLS = x + self.alpha[self.iter]*self.direction[self.iter]

        self.fevals += lineSearch.fevals
        # print("alpha ", self.alpha[self.iter])
        # print("xLS", self.xLS)
        # print("LS FEvals: ", lineSearch.fevals)
        return self.xLS

    def optimize(self):
        self.fevals = 0
        self.iter = 0
        self.totIter = 0

        # Initial values for g and d
        self.g[0] = self.b + self.H @ self.x[0]
        self.direction[0] = -self.g[0]

        # print("b ", self.b)
        # print("H ", self.H)
        # print("g ", self.g[0])
        # print("")
        while (self.totIter < self.maxIters-2):
            # print("Tot Iter: ", self.totIter)
            # self.hessVal[self.iter] = self.hessFunc(self.x[self.iter])

            self.x[self.iter+1] = self.line_search(self.x[self.iter])

            ## Debug
            # print("ITER ", self.iter)
            # print("direction ", self.direction[self.iter])
            # print("alpha ", self.alpha[self.iter])
            # print("x: ", self.x[self.iter])
            # print("x[k+1]: ", self.x[self.iter+1])
            # input()

            # Check stopping condition
            if self.stopping_cond() == True:
                print("\nStopping condition reached, algorithm terminating.")
                self.xOpt = self.x[self.iter+1]
                return self.xOpt, self.costFunc(self.xOpt), self.fevals

            if self.iter == self.restartIter-1:
                self.x[0] = self.x[self.iter+1]
                self.totIter += self.iter
                self.iter = 0
            else:
                # Compute new direction
                self.direction[self.iter+1] = self.get_direction()
                self.iter += 1

        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[self.iter]
        return self.xOpt, self.costFunc(self.xOpt), self.fevals
