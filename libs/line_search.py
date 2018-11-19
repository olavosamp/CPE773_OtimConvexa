# import numpy as np
from autograd import grad, hessian
import autograd.numpy as np
from copy import copy

from libs.operators import norm2

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
        self.fevals = 0
        self.iter = 0
        while (np.abs(self.interval[0] - self.interval[1]) > self.xtol) and (self.iter < self.maxIters):
            self.iteration()
            self.iter += 1
            # print("interval: ", self.interval[0])

        self.xOpt = self.interval[0]
        # input()
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
        self.fevals = 0
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
        xB[0] = xL[0] + I[1]

        # Evaluate f(xA), f(xB)
        fA[0] = self.evaluate(xA[0])
        fB[0] = self.evaluate(xB[0])

        # print("\nStart iterating")
        while True:
            I[k+2] = I[k+1]*self.fibSequence[self.iterations - k - 2]/self.fibSequence[self.iterations - k-1]

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

            if (k == self.iterations-2) or (xA[k+1] > xB[k+1]):
                self.xOpt = xA[k+1]
                return self.xOpt
            else:
                k += 1


# Golden Section Search
class GoldenSectionSearch(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = interval

        # self.goldenRatio = 1.618034
        self.goldenRatio = (1 + np.sqrt(5))/2

    def optimize(self):
        self.fevals = 0
        # Initialize variables
        fA = np.zeros(self.maxIters-1)
        fB = np.zeros(self.maxIters-1)
        xA = np.zeros(self.maxIters-1)
        xB = np.zeros(self.maxIters-1)
        xU = np.zeros(self.maxIters-1)
        xL = np.zeros(self.maxIters-1)
        I  = np.zeros(self.maxIters)

        xL[0] = self.interval[0]
        xU[0] = self.interval[1]

        k = 0 # Iteration counter
        I[0] = xU[0] - xL[0]
        I[1] = I[0]/self.goldenRatio

        xA[0] = xU[0] - I[1]
        xB[0] = xL[0] + I[1]

        # Evaluate f(xA), f(xB)
        fA[0] = self.evaluate(xA[0])
        fB[0] = self.evaluate(xB[0])

        # print("\nStart iterating")
        while True:
            I[k+2] = I[k+1]/self.goldenRatio

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

            if (I[k] < self.epsilon) or (xA[k+1] > xB[k+1]):
                if fA[k+1] >= fB[k+1]:
                    self.xOpt = 0.5*(xB[k+1] + xU[k+1])
                elif fA[k+1] < fB[k+1]:
                    self.xOpt = 0.5*(xL[k+1] + xA[k+1])

                return self.xOpt
            else:
                k += 1


# Quadratic interpolation method
class QuadraticInterpolation(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = interval

    def optimize(self):
        self.fevals = 0
        x0 = np.inf
        x1 = self.interval[0]
        x3 = self.interval[1]

        x2 = (x1 + x3)/2
        f1 = self.evaluate(x1)
        f2 = self.evaluate(x2)
        f3 = self.evaluate(x3)
        # print("f1: ", f1)
        # print("f2: ", f2)
        # print("f3: ", f3)
        while True:
            arg1 = (x2**2 - x3**2)*f1 + (x3**2 - x1**2)*f2 + (x1**2 - x2**2)*f3
            arg2 = 2*((x2 - x3)*f1 + (x3 - x1)*f2 + (x1 - x2)*f3)
            xTest = arg1/arg2

            fTest = self.evaluate(xTest)
            # print(fTest)
            # print(xTest)
            # input()

            if np.abs(xTest - x0) < self.epsilon:
                self.xOpt = xTest
                return self.xOpt
            elif (x1 < xTest) and (xTest < x2):
                if fTest <= f2:
                    x3 = x2
                    f3 = f2
                    x2 = xTest
                    f2 = fTest
                else: # fTest > f2
                    x1 = x2
                    f1 = f2
                    x2 = xTest
                    f2 = fTest
            elif (x2 < xTest) and (xTest < x3):
                if fTest <= f2:
                    x1 = x2
                    f1 = f2
                    x2 = xTest
                    f2 = fTest
                else: # fTest > f2
                    x3 = xTest
                    f3 = fTest
            x0 = xTest


# Cubic interpolation method
class CubicInterpolation(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon  = xtol/4
        self.interval = interval

    def optimize(self):
        self.iter = 0
        self.fevals = 0
        grad_func = grad(self.evaluate)

        x = np.zeros(4)
        fList = np.zeros(4)

        x[0] = np.inf
        x[1] = self.interval[0]
        x[3] = self.interval[1]
        x[2] = (x[1] + x[3])/2

        fList[1] = self.evaluate(x[1])
        fList[2] = self.evaluate(x[2])
        fList[3] = self.evaluate(x[3])
        f1_grad  = grad_func(x[1])

        while self.iter < self.maxIters:
            beta  = (fList[2] - fList[1] + f1_grad*(x[1] - x[2]))/((x[1] - x[2])**2)
            gamma = (fList[3] - fList[1] + f1_grad*(x[1] - x[3]))/((x[1] - x[3])**2)
            theta = (2*(x[1]**2) - x[2]*(x[1] + x[2]))/(x[1] - x[2])
            psi   = (2*(x[1]**2) - x[3]*(x[1] + x[3]))/(x[1] - x[3])
            a3 = (beta - gamma)/(theta - psi)
            a2 = beta - theta*a3
            a1 = f1_grad - 2*a2*x[1] - 3*a3*(x[1]**2)


            # Select xTest
            minXValue = -a2/(3*a3)

            extremPointsPos = (1/(3*a3)) * (-a2 + np.sqrt(a2**2 - 3*a1*a3))
            extremPointsNeg = (1/(3*a3)) * (-a2 + np.sqrt(a2**2 + 3*a1*a3))


            if extremPointsPos > minXValue:
                xTest = extremPointsPos
            elif extremPointsNeg > minXValue:
                xTest = extremPointsNeg
            else:
                xTest = extremPointsPos
            fTest = self.evaluate(xTest)

            ## Debug
            # print("Iter: ", self.iter)
            # print("beta: ", beta)
            # print("gamma: ", gamma)
            # print("theta: ", theta)
            # print("psi: ", psi)
            # print("a3: ", a3)
            # print("a2: ", a2)
            # print("a1: ", a1)
            # print("extremPointsPos: ", extremPointsPos)
            # print("extremPointsNeg: ", extremPointsNeg)
            # print("xTest: ", xTest)
            # print("fTest: ", fTest)

            # Test stopping conditions
            if (np.abs(xTest - x[0]) < self.epsilon):
                self.xOpt = xTest
                return self.xOpt

            maxArg = np.argmax(fList[1:])+1
            x[0] = xTest
            x[maxArg] = xTest
            fList[maxArg] = fTest
            if maxArg == 1:
                f1_grad = grad_func(xTest)

            self.iter += 1

        print("Algorithm did not converge.")
        self.xOpt = xTest
        return self.xOpt


# Davies-Swann-Campey Algorithm
class DSCAlgorithm(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = interval

    def optimize(self):
        self.fevals = 0
        # Initialize variables
        self.iter = -1   # Make k = k-1 to account for zero-indexing
        x0   = np.zeros(self.maxIters)
        x1   = np.zeros(self.maxIters)
        x_m1 = np.zeros(self.maxIters) # x_(-1)
        increment =np.zeros(self.maxIters)
        f0   = np.zeros(self.maxIters)
        f1   = np.zeros(self.maxIters)
        f_m1 = np.zeros(self.maxIters) # f_(-1)
        f_m  = np.zeros(self.maxIters)

        scaleFactor = 0.1   # K
        increment[0] = 5*self.xtol  # Sigma_0
        x0[0] = (self.interval[0]+self.interval[1])/2

        while True:
            self.iter += 1
            x_m1[self.iter] = x0[self.iter] - increment[self.iter]
            x_m1[self.iter] = x0[self.iter] + increment[self.iter]

            f0[self.iter] = self.evaluate(x0[self.iter])
            f1[self.iter] = self.evaluate(x1[self.iter])

            if not((f_m1[self.iter] >= f0[self.iter]) and (f0[self.iter] <= f1[self.iter])):
                if f0[self.iter] > f1[self.iter]:
                    p = 1
                elif f_m1[self.iter] < f0[self.iter]:
                    p = -1
                # STEP 4
                xn = [x0[self.iter]]
                fn = [f0[self.iter]]
                n = 1
                while True:
                    xnNew = xn[n-1] + (2**(n-1))*p*increment[self.iter]
                    xn.append(xnNew)
                    fn.append(self.evaluate(xn[n]))

                    if (fn[n] > fn[n-1]):
                        break
                    n += 1
                # STEP 5
                xm = xn[n-1] + (2**(n-2))*p*increment[self.iter]
                fm = self.evaluate(xm)

                # STEP 6
                if fm >= fn[n-1]:
                    x0[self.iter+1] = xn[n-1] + ((2**(n-2))*p*increment[self.iter]*(fn[n-2] - fm))/(2*(fn[n-2] - 2*fn[n-1] + fm))
                else: # if fm[self.iter] < fNew
                    x0[self.iter+1] = xm + ((2**(n-2))*p*increment[self.iter]*(fn[n-1] - fn[n]))/(2*(fn[n-1] - 2*fm + fn[n]))

                if ((2**(n-2))*increment[self.iter]) <= self.xtol:
                    # STEP 8
                    self.xOpt = x0[self.iter+1]
                    return self.xOpt
                    # Return to STEP 2
            else:
                # STEP 7
                x0[self.iter+1] = x0[self.iter] + (increment[self.iter]*(f_m1[self.iter] - f1[self.iter]))/(2*(f_m1[self.iter] - 2*f0[self.iter] + f1[self.iter]))
                if increment[self.iter] < self.xtol:
                    # STEP 8
                    self.xOpt = x0[self.iter+1]
                    return self.xOpt

            increment[self.iter+1] = scaleFactor*increment[self.iter]


# Backtracking Line Search
class BacktrackingLineSearch(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3, alpha=0.5, beta=0.5,
                initialX=None, initialDir=None):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.interval = interval
        self.alpha    = alpha    # alpha in ]0.0, 0.5[
        self.beta     = beta     # beta  in ]0.0, 1.0[

        if initialX is None:
            self.x = np.random.uniform(self.interval[0], self.interval[1])
        else:
            self.x = initialX

        self.direction = initialDir

    def optimize(self):
        self.fevals = 0
        grad_func = grad(self.evaluate)
        # print("Initial X: ", self.x)
        self.iter = 1

        gradient = grad_func(self.x)

        # Search direction is minus gradient direction
        if self.direction is None:
            self.direction = -np.sign(gradient)

        self.dirList   = []
        self.alphaList = []
        while self.iter <= self.maxIters:
            # print("LS Iter: ", self.iter)
            t = 1
            iter2 = 1

            self.fx = self.evaluate(self.x)

            # Actual Backtracking Line Search
            while (self.evaluate(self.x + t*self.direction) > self.fx + self.alpha*t*(np.transpose(gradient) @ self.direction)) and (iter2 < self.maxIters):
                # print("Iter2: ", iter2)
                t = self.beta*t
                t = np.clip(t, 2*self.xtol, None)
                iter2 += 1
            self.x = self.x + t*self.direction
            gradient = grad_func(self.x)

            self.dirList.append(self.direction)
            self.alphaList.append(t)

            if np.isnan(self.x).any() or np.isnan(gradient).any():
                print("\nBacktracking diverged.")
                print("x: ", self.x)
                print("Grad(x): ", gradient)
                return self.x

            self.direction = -np.sign(gradient)
            # self.direction = -gradient


            # if np.ndim(gradient) > 1:
                # print("dim>1")
            if np.linalg.norm(gradient, ord=2) < self.xtol:
                self.xOpt = self.x
                return self.xOpt
            else:
                self.iter += 1
            # else:
            #     # print("dim = 1")
            #     if norm2(gradient) < self.xtol:
            #         self.xOpt = self.x
            #         return self.xOpt
            #     else:
            #         self.iter += 1
        print("Algorithm did not converge.")
        self.xOpt = self.x
        return self.xOpt

# Fletcher Inexact Line Search
class FletcherILS(LineSearch):
    def __init__(self, costFunc, interval, xtol=1e-8, maxIters=1e3, initialX=None, initialDir=None):
        super().__init__(costFunc, xtol)

        self.maxIters = maxIters
        self.epsilon = xtol/4
        self.interval = np.array(interval)

        if initialX is None:
            self.xk = np.random.uniform(self.interval[0, :], self.interval[1, :])
        else:
            self.xk = initialX

        self.dk = initialDir

        self.rho     = 0.1
        self.sigma   = 0.7
        self.tau     = 0.1
        self.chi     = 9
        self.alpha_L = 0
        self.alpha_U = np.inf

    def optimize(self):
        self.fevals = 0
        self.grad_func = grad(self.evaluate)
        self.hess_func = hessian(self.evaluate)

        if self.dk is None:
            self.dk = -self.grad_func(self.xk)

        # print("Initial X: ", self.xk)
        self.alphaList = []
        self.dirList = []
        self.iter = 0
        while self.iter <= self.maxIters:
            print("\nIter: ", self.iter)
            # Perform Line Search to compute new alpha_0
            self.alpha_0 = self.inexact_line_search()

            # Compute new xk (and limit it to search interval)
            self.xk = self.xk + self.alpha_0*self.dk

            self.alphaList.append(self.alpha_0)
            self.dirList.append(self.dk)

            gradient = self.grad_func(self.xk)

            if np.linalg.norm(gradient, ord=2) <= self.xtol:
                self.xOpt = self.xk
                return self.xOpt
            else:
                self.dk = -self.grad_func(self.xk)    # Compute new direction
                self.iter += 1

        print("Algorithm did not converge")
        self.xOpt = self.xk
        return self.xOpt

    def inexact_line_search(self):
        self.alpha_L = 0
        self.alpha_U = np.inf

        # Step1
        gk = self.grad_func(self.xk + self.alpha_L*self.dk)

        # Step 2
        fL = self.evaluate(self.xk + self.alpha_L*self.dk)
        fL_grad = np.dot(gk, self.dk)

        # Step 3
        g0 = self.grad_func(self.xk)
        H0 = self.hess_func(self.xk)
        self.alpha_0 = (np.transpose(g0)@ g0)/(np.transpose(g0) @ H0 @ g0)
        self.alpha_0 = np.clip(self.alpha_0, self.alpha_L, self.alpha_U)

        iter2 = 0
        while iter2 <= self.maxIters:
            print("Iter2 int: ", iter2)
            # Step 4
            self.f0 = self.evaluate(self.xk + self.alpha_0*self.dk)

            # Step 5 (Interpolation)
            if self.f0 > fL + self.rho*(self.alpha_0 - self.alpha_L)*fL_grad:
                print("Interpol")

                if self.alpha_0 < self.alpha_U:
                    self.alpha_U = self.alpha_0
                self.alpha_0_estim = self.alpha_L + ((self.alpha_0 - self.alpha_L)**2)*fL_grad/(2*(fL - self.f0 + (self.alpha_0 - self.alpha_L)*fL_grad))

                if self.alpha_0_estim < self.alpha_L + self.tau*(self.alpha_U - self.alpha_L):
                    self.alpha_0_estim = self.alpha_L + self.tau*(self.alpha_U - self.alpha_L)
                if self.alpha_0_estim > self.alpha_U - self.tau*(self.alpha_U - self.alpha_L):
                    self.alpha_0_estim = self.alpha_U - self.tau*(self.alpha_U - self.alpha_L)

                self.alpha_0 = self.alpha_0_estim
                iter2 += 1
                continue

            # Step 6
            f0_grad = np.dot(self.grad_func(self.xk + self.alpha_0*self.dk), self.dk)

            # Step 7 (Extrapolation)
            if f0_grad < self.sigma*fL_grad:
                print("Extrapol")

                deltaAlpha0 = ((self.alpha_0 - self.alpha_L)*f0_grad)/(fL_grad - f0_grad)
                if deltaAlpha0 < self.tau*(self.alpha_0 - self.alpha_L):
                    deltaAlpha0 = self.tau*(self.alpha_0 - self.alpha_L)
                if deltaAlpha0 > self.chi*(self.alpha_0 - self.alpha_L):
                    deltaAlpha0 = self.chi*(self.alpha_0 - self.alpha_L)

                if deltaAlpha0 < 1e-6:
                    deltaAlpha0 = self.alpha_0

                self.alpha_0_estim = self.alpha_0 + deltaAlpha0
                self.alpha_L = self.alpha_0
                self.alpha_0 = self.alpha_0_estim
                fL = self.f0
                fL_grad = f0_grad
                iter2 += 1
                continue
            # If neither Interpolation nor Extrapolation conditions are met,
            # convergence has been reached. Terminate Line Search.
            break
        # Step 8
        return self.alpha_0
