from tqdm             import tqdm
from autograd         import grad, hessian, jacobian
from copy             import copy
import autograd.numpy as np

from libs.operators        import positive_definite
from libs.line_search      import *

class QuasiNewton:
    def __init__(self, func, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
        self.costFunc   = func
        self.gradFunc   = grad(self.evaluate)
        self.hessFunc   = hessian(self.evaluate)
        self.maxIters   = int(maxIters)
        self.maxItersLS = maxItersLS
        self.interval   = interval
        self.fevals     = 0
        self.ftol       = ftol

        self.x_is_matrix= False

        if np.ndim(initialX) > 1:
            self.x_is_matrix = True
            initialX = np.squeeze(initialX)

        self.xLen = np.shape(initialX)[0]
        self.direction  = np.zeros((maxIters, self.xLen))
        self.x          = np.zeros((maxIters, self.xLen))
        self.x[0] = initialX

        self.gradient   = np.zeros((self.maxIters, self.xLen))
        self.S          = np.zeros((self.maxIters, self.xLen, self.xLen))

        self.epsilon1 = self.ftol
        self.k = 0
        self.m = 0
        self.rho = 0.1
        self.sigma = 0.7
        self.tau = 0.1
        self.chi = 0.75
        self.m_hat = 600
        self.epsilon2 = 1e-10

        self.S[0] = np.eye(self.xLen)


    def evaluate(self, x):
        if self.x_is_matrix:
            x = np.reshape(x, (self.xLen,1))
        result = self.costFunc(x)
        self.fevals += 1
        return result

    def line_search(self, x):
        def funcLS(alpha):
            return self.evaluate(x + alpha*self.direction[self.iter])

        lineSearch = FibonacciSearch(funcLS, self.interval, ftol=self.ftol, maxIters=self.maxItersLS)
        self.alpha[self.iter] = lineSearch.optimize()
        self.xLS = self.x[self.iter] + self.alpha[self.iter]*self.direction[self.iter]
        return self.xLS


    def get_S(self):
        pass


    # def get_direction(self, x):
    #     if self.k > 0:
    #         self.gradient[self.k] = self.gradFunc(self.x[self.k])
    #         self.S[self.k]        = self.get_S()
    #
    #     direction = -(self.S[self.k] @ self.gradient[self.k])
    #     return direction


    def get_alpha0(self):
        if np.abs(self.fLGrad) > self.epsilon2:
            self.alpha0 = -2*self.delta_f0/self.fLGrad
        else:
            self.alpha0 = 1

        if (self.alpha0 <= 0) or (self.alpha0 > 1):
            self.alpha0 = 1
        return self.alpha0


    def optimize(self):
        self.fevals = 0
        ## Step 1
        self.f0 = self.evaluate(self.x[0])
        self.gradient[0] = self.gradFunc(self.x[0])
        self.m += 2
        self.f_00 = copy(self.f0)
        self.delta_f0 = copy(self.f0)

        while self.k < self.maxIters-1:
            # print("Iter: ", self.k)
            self.escape = False
            ## Step 2
            self.direction[self.k] = -(self.S[self.k] @ self.gradient[self.k])
            self.alphaL = 0
            self.alphaU = 1e99
            self.fL = copy(self.f0)
            self.fLGrad = (self.gradFunc(self.x[self.k] + self.alphaL*self.direction[self.k]).T)@self.direction[self.k]
            self.alpha0 = self.get_alpha0()

            while self.escape == False:
                ## Step 3
                self.deltaK = self.alpha0*self.direction[self.k]
                self.f0 = self.evaluate(self.x[self.k] + self.deltaK)
                self.m += 1

                ## Step 4
                cond1 = self.f0 > self.fL + self.rho*(self.alpha0 - self.alphaL)*self.fLGrad
                cond2 = np.abs(self.fL - self.f0) > self.epsilon2
                if cond1 and cond2:
                    self.alphaU = self.alpha0
                    self.alpha0_hat = self.alphaL + ((self.alpha0 - self.alphaL)**2)*self.fLGrad/(2*(self.fL - self.f0 + (self.alpha0 - self.alphaL)*self.fLGrad))

                    self.alpha0L_hat = self.alphaL + self.tau*(self.alphaU - self.alphaL)
                    if self.alpha0_hat < self.alpha0L_hat:
                        self.alpha0_hat = self.alpha0L_hat

                    self.alpha0U_hat = self.alphaU - self.tau*(self.alphaU - self.alphaL)
                    if self.alpha0_hat > self.alpha0U_hat:
                        self.alpha0_hat = self.alpha0U_hat

                    self.alpha0 = self.alpha0_hat
                    # Go to Step 3
                else:
                    ## Step 5
                    self.f0Grad = (self.gradFunc(self.x[self.k] + self.alpha0*self.direction[self.k]).T)@self.direction[self.k]
                    self.m += 1

                    ## Step 6
                    cond1 = self.f0Grad < self.sigma*self.fLGrad
                    cond2 = np.abs(self.fL - self.f0) > self.epsilon2
                    if cond1 and cond2:
                        deltaAlpha0 = (self.alpha0 - self.alphaL)*self.f0Grad/(self.fLGrad - self.f0Grad)
                        if deltaAlpha0 <= 0:
                            self.alpha0_hat = 2*self.alpha0
                        else:
                            self.alpha0_hat = self.alpha0 + deltaAlpha0

                        self.alpha0U_hat = self.alpha0 + self.chi*(self.alphaU - self.alpha0)
                        if self.alpha0_hat > self.alpha0U_hat:
                            self.alpha0_hat = self.alpha0U_hat

                        self.alphaL = self.alpha0
                        self.alpha0 = self.alpha0_hat
                        self.fL = self.f0
                        self.fLGrad = self.f0Grad
                        # Go to Step 3
                    else:
                        ## Step 7
                        self.x[self.k+1] = self.x[self.k] + self.deltaK
                        self.delta_f0 = self.f_00 - self.f0
                        # Stopping condition
                        if np.linalg.norm(self.deltaK, ord=2) < self.epsilon1 and np.linalg.norm(self.delta_f0) < self.epsilon1:
                            print("Stopping conditions reached. Algorithm terminating.")
                            self.xOpt = self.x[self.k+1]
                            return self.xOpt, self.costFunc(self.xOpt), self.fevals
                        self.f_00 = self.f0

                        ## Step 8
                        self.gradient[self.k+1] = self.gradFunc(self.x[self.k+1])
                        self.gamma = self.gradient[self.k+1] - self.gradient[self.k]

                        self.D = self.deltaK.T@self.gamma
                        if self.D <= 0:
                            self.S[self.k+1] = np.eye(self.xLen)
                        else:
                            self.S[self.k+1] = self.get_S()
                        self.k += 1
                        self.escape == True
                        # Go to Step 2
        print("\nAlgorithm did not converge.")
        self.xOpt = self.x[self.k]
        return self.xOpt, self.costFunc(self.xOpt), self.fevals



class QuasiNewtonDFP(QuasiNewton):
    def get_S(self):
        arg1 = (self.deltaK @ np.transpose(self.deltaK))/(np.transpose(self.deltaK) @ self.gamma)
        arg2 = (self.S[self.k]@self.gamma)@(self.gamma.T@self.S[self.k])/(self.gamma.T@self.S[self.k]@self.gamma)

        self.S[self.k+1] = self.S[self.k] + arg1 - arg2
        return self.S[self.k+1]

class QuasiNewtonBFGS(QuasiNewton):
    def get_S(self):
        arg1 = (1 + ((self.gamma.T@self.S[self.k])@self.gamma)/(self.gamma.T@self.deltaK))*((self.deltaK@self.deltaK.T)/self.gamma.T@self.deltaK)
        arg2 = (self.deltaK@(self.gamma.T@self.S[self.k]) + self.S[self.k]@self.gamma@self.deltaK.T)/(self.gamma.T@self.deltaK)

        self.S[self.k+1] = self.S[self.k] + arg1 - arg2
        return self.S[self.k+1]
