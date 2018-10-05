import numpy as np

# class FunctionEvaluator:
#     def __init__(self, func):
#         self.func = func
#         self.fevals = 0
#     def eval(self, x):
#         result = self.func(x)
#         self.fevals += 1
#         return result

class OptimizationAlgorithm:
    def __init__(self, costFunc, xtol):
        self.costFunc = costFunc
        self.fevals = 0
        self.xtol = xtol

    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result

class DichotomousSearch(OptimizationAlgorithm):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = interval

    def get_test_points(self):
        self.intervalCenter = np.mean(self.interval)
        self.xA = self.intervalCenter - self.epsilon/2
        self.xB = self.intervalCenter + self.epsilon/2

        self.fA = self.evaluate(self.xA)
        self.fB = self.evaluate(self.xB)
        return self.xA, self.xB

    def iteration(self):
        # Compute xA, xB and f(xA), f(xB)
        self.get_test_points()

        if self.fA <= self.fB:
            self.interval = [self.interval[0], self.xB]
        elif self.fA > self.fB:
            self.interval = [self.xA, self.interval[1]]

        return self.interval

    def optimize(self):
        self.numIters = 0
        while (np.abs(self.interval[0] - self.interval[1]) > self.xtol) and (self.numIters < self.maxIters):
            self.iteration()
            self.numIters += 1

        xOpt = self.interval[0]
        return xOpt
