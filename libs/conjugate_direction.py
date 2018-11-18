from autograd         import grad, hessian, jacobian
import autograd.numpy as np

from libs.operators        import positive_definite
from libs.line_search      import CubicInterpolation, BacktrackingLineSearch

class ConjugateDirection:
    def __init__(self, func, initialX, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
        self.costFunc   = func
        self.gradFunc   = grad(self.evaluate)
        self.maxIters   = int(maxIters)
        self.maxItersLS = maxItersLS
        self.interval   = interval
        self.fevals     = 0
        self.xtol       = xtol

        self.xLen       = np.shape(initialX)[0]
        self.direction  = np.zeros((self.maxIters, self.xLen))
        self.x          = np.zeros((self.maxIters, self.xLen))
        self.x[0]       = initialX

        self.gradFunc   = grad(self.evaluate)

    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result

    def line_search(self, x):
        lineSearch = CubicInterpolation(self.costFunc, self.interval, xtol=self.xtol, maxIters=self.maxItersLS)
        self.xLS = lineSearch.optimize()
        return self.xLS

    def get_direction(self, x):
        pass

    def optimize(self):
        pass


class CoordinateGradient(ConjugateDirection):
    def __init__(self, func, initialX, b, H, interval=[-1e15, 1e15], xtol=1e-6, maxIters=1e3, maxItersLS=200):
        super().__init__(func, initialX, interval=interval, xtol=xtol, maxIters=maxIters, maxItersLS=maxItersLS)

        self.hessFunc = hessian(self.evaluate)

        self.g       = np.zeros((self.maxIters, self.xLen))
        self.d       = np.zeros((self.maxIters, self.xLen))
        self.alpha   = np.zeros((self.maxIters))
        self.hessVal = np.zeros((self.maxIters, self.xLen, self.xLen))

        self.b = b
        self.H = H

    def get_direction(self):
        self.g[self.iter+1] = self.b + self.H @ self.x[self.iter+1]
        self.beta = (np.transpose(self.g[self.iter+1]) @ self.g[self.iter+1])/(np.transpose(self.g[self.iter]) @ self.g[self.iter])

        direction = -self.g[self.iter+1] + self.beta*self.d[self.iter]
        return direction

    def stopping_cond(self):
        norm2 = np.linalg.norm(self.alpha[self.iter]*self.d[self.iter], ord=2)
        return norm2 < self.xtol

    def optimize(self):
        self.iter = 0

        # Initial values for g and d
        self.g[0] = self.b + self.H @ self.x[0]
        self.d[0] = -self.g[0]

        while self.iter < self.maxIters-2:
            self.hessVal[self.iter] = self.hessFunc(self.x[self.iter])
            self.alpha[self.iter] = (np.transpose(self.g[self.iter]) @ self.g[self.iter])/(np.transpose(self.d[self.iter]) @ self.hessVal[self.iter] @ self.d[self.iter])

            self.x[self.iter+1] = self.x[self.iter] + self.alpha[self.iter]*self.d[self.iter]

            # Check stopping condition
            if self.stopping_cond() == True:
                print("\nStopping condition reached, algorithm terminating.")
                self.xOpt = self.x[self.iter+1]
                return self.xOpt, self.costFunc(self.xOpt), self.fevals

            # Compute new direction
            self.d[self.iter+1] = self.get_direction()

            self.iter += 1

        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[-1]
        return self.xOpt, self.costFunc(self.x[-1]), self.fevals
