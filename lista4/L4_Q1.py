import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func10
from libs.constrained_optim   import *
from libs.quasi_newton        import *
from libs.gradient_methods    import *
from libs.conjugate_direction import *


ftol       = 1e-8
maxIters   = 500
maxItersLS = 200
function   = func10
interval   = [-1e3, 1e3]
savePath = dirs.results+"L4_Q1.xls"

# Q 11.6
eqConstraintsFun   = [ lambda x: x[0] + 2*x[1] + x[2] + 2*x[3] - 3,
                    lambda x: x[0] + x[1] + 2*x[2] + 4*x[3] - 5,
]

eqConstraintsMat = {'A': np.array([[1, 2, 1, 2],
                                   [1, 1, 2, 4]]),
                    'b': np.array([[3],
                                    [5]])}

# NOTE: Scipy uses inequality constraints of form
#    f(x) >= 0
# While Boyd uses
#   f(x) <= 0
# As such, they are defined in Boyd's format and converted by
# get_scipy_constraints script
ineqConstraints = [ lambda x: x,
]

constraintList = get_scipy_constraints(eqConstraintsFun, ineqConstraints)
# constraintList = get_scipy_constraints(eqConstraintsFun, None)

initialX = [1,0,0,1]
initialX = feasibility(constraintList, initialX)
# print(initialX)
# input()

initialXList = [initialX]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
for initialX in initialXList:
    # sd_algorithm = ConjugateGradient(function, initialX, interval=interval, ftol=ftol,
    #                                  maxIters=maxIters, maxItersLS=maxItersLS)
    #
    # xOpt, fOpt, fevals = sd_algorithm.optimize()
    # #
    # # print("Initial X:", initialX)
    # print("\nUnconstrained Optimization")
    # print("f(x*): ", fOpt)
    # print("x*: ", xOpt)
    # print("FEvals: ", fevals)


    optimResult = spo.minimize(function, initialX, method='SLSQP', tol=ftol,
                                constraints=constraintList)
    xRef = optimResult.x
    fRef = optimResult.fun
    # fevalRef = optimResult.nfev
    # deltaF = np.abs(fOpt - fRef)
    # input()

    # print("Delta f(x) = ", deltaF)
    # print("Ref FEvals = ", fevalRef)

    F, x_hat = eq_constraint_elimination_composer(eqConstraintsMat)
    newFunction = compose_eq_cons_func(function, F, x_hat)

    newConstraintList = get_scipy_constraints(None, ineqConstraints)


    optimResult = spo.minimize(newFunction, np.array([0, 1]), method='SLSQP', tol=ftol)
                                # constraints=newConstraintList)
    xOpt = F @ optimResult.x + x_hat
    fOpt = function(xOpt)

    print("\nConstrained Optimization")
    # print("F:", F)
    # print("x_hat: ", x_hat)

    print("Ref x* = ", xRef)
    # print("x*: ", xOpt)
    print("Ref f(x*) = ", fRef)
    # print("f(x*): ", fOpt)

#     xList.append(xOpt)
#     fxList.append(fOpt)
#     fevalsList.append(fevals)
#     deltaFList.append(deltaF)
#
# xList = np.array(xList)
# resultsDf = pd.DataFrame(data={"Initial x": initialXList,
#                                 "x*_1"    : xList[:, 0],
#                                 "x*_2"    : xList[:, 1],
#                                 "f(x*)"   : fxList,
#                                 "FEvals"  : fevalsList,
#                                 "Delta F" : deltaFList})
#
# print(resultsDf)
# resultsDf.to_excel(savePath)
