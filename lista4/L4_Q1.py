import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func10
from libs.constrained_optim   import *
from libs.quasi_newton        import *
from libs.gradient_methods    import *


xtol       = 1e-8
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

ineqConstraints = [ lambda x: x,
]

constraintList = get_scipy_constraints(eqConstraintsFun, ineqConstraints)
# constraintList = get_scipy_constraints(eqConstraintsFun, None)

initialX = [0,0,0,0]
initialX = feasibility(constraintList, initialX)

initialXList = [initialX]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
for initialX in initialXList:
    # sd_algorithm = QuasiNewtonDFP(function, initialX, interval=interval, xtol=xtol,
    #                                  maxIters=maxIters, maxItersLS=maxItersLS)
    #
    # xOpt, fOpt, fevals = sd_algorithm.optimize()
    #
    # print("Initial X:", initialX)
    # print("f(x*): ", fOpt)
    # print("x*: ", xOpt)
    # print("FEvals: ", fevals)


    # optimResult = spo.minimize(function, initialX, method='SLSQP', tol=xtol,
    #                             constraints=constraintList)
    optimResult = spo.linprog(np.zeros(4), A_eq=eqConstraintsMat['A'], b_eq=eqConstraintsMat['b'])
    xRef = optimResult.x
    fRef = optimResult.fun
    # fevalRef = optimResult.nfev
    # deltaF = np.abs(fOpt - fRef)
    print(eqConstraintsMat['A'].shape)
    print(eqConstraintsMat['b'].shape)
    print(xRef.shape)

    check = eqConstraintsMat['A']@xRef
    print(check.shape)
    # print(np.isclose(check, eqConstraintsMat['b'], atol=xtol))
    print(check[0] == eqConstraintsMat['b'][0])
    print(check[1] == eqConstraintsMat['b'][1])
    input()

    # print("Delta f(x) = ", deltaF)
    print("Ref x* = ", xRef)
    print("Ref f(x*) = ", fRef)
    print("Ref FEvals = ", fevalRef)
    # input()
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
