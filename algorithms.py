import numpy as np

class LineSearch:
    def __init__(self, costFunc, xtol):
        self.costFunc = costFunc
        self.fevals = 0
        self.xtol = xtol

    def evaluate(self, x):
        result = self.costFunc(x)
        self.fevals += 1
        return result

class DichotomousSearch(LineSearch):
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

        self.xOpt = self.interval[0]
        return self.xOpt

# Fibonacci Search
class FibonacciSearch(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = interval

        # Compute number of iterations required to reach epsilon
        self.compute_fibonacci(40) # Long Fibonacci will overflow and break the algorithm
        auxSeq = self.fibSequence[(self.fibSequence < 1/self.epsilon)]

        self.iterations = np.argmax(auxSeq)
        if self.iterations > self.maxIters:
            self.iterations = self.maxIters

        self.fibSequence = self.fibSequence[:self.iterations]

    def compute_fibonacci(self, size):
        self.fibSequence = np.zeros(size, dtype=np.int)
        self.fibSequence[0] = 1
        self.fibSequence[1] = 1
        for i in range(2, size):
            self.fibSequence[i] = self.fibSequence[i-1] + self.fibSequence[i-2]

        return self.fibSequence

    def optimize(self):
        # Initialize variables
        fA = np.zeros(self.iterations)
        fB = np.zeros(self.iterations)
        xA = np.zeros(self.iterations)
        xB = np.zeros(self.iterations)
        xU = np.zeros(self.iterations)
        xL = np.zeros(self.iterations)
        I  = np.zeros(self.iterations+1)

        xL[0] = self.interval[0]
        xU[0] = self.interval[1]

        k = 0 # Iteration counter
        I[0] = xU[0] - xL[0]
        I[1] = I[0]*self.fibSequence[self.iterations-2]/self.fibSequence[self.iterations-1]

        xA[0] = xU[0] - I[1]
        xB[0] = xL[0] - I[1]

        # Evaluate f(xA), f(xB)
        fA[0] = self.evaluate(xA[0])
        fB[0] = self.evaluate(xB[0])

        # print("\nStart iterating")
        while True:
            print("Iter: ", k)
            I[k+2] = I[k+1]*self.fibSequence[self.iterations - k - 2]/ self.fibSequence[self.iterations - k-1]

            if fA[k] >= fB[k]:
                xL[k+1] = xA[k]
                xU[k+1] = xU[k]
                xA[k+1] = xB[k]
                xB[k+1] = xA[k] + I[k+2]

                fA[k+1] = fB[k]
                fB[k+1] = self.evaluate(xB[k+1])
            elif fA[k] < fB[k]:
                xL[k+1] = xL[k]
                xU[k+1] = xB[k]
                xA[k+1] = xU[k+1] - I[k+2]
                xB[k+1] = xA[k]

                fA[k+1] = self.evaluate(xA[k+1])
                fB[k+1] = fA[k]

            if (k == self.iterations-2) or (xA[k+1] > fB[k+1]):
                self.xOpt = xA[k+1]
                return self.xOpt
            else:
                k += 1


# Golden Section Search

# Quadratic interpolation method

# Cubic interpolation method

# Davies-Swann-Campey Algorithm

# Inexact Line Search
