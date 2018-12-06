import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
# from libs.functions           import func10
from libs.constrained_optim   import *
from libs.quasi_newton        import *
from libs.gradient_methods    import *
from libs.conjugate_direction import *


ftol       = 1e-8
maxIters   = 200
maxItersLS = 2000
interval   = [-1e8, 1e8]
savePath = dirs.results+"L4_Q2.xls"

# Cost function
def costFunction(x):
    return (x[0] - x[2])**2 + (x[1] - x[3])**2
    # return np.linalg.norm(x[:2]- x[2:], ord=2)**2

# Q2
# NOTE: Scipy uses inequality constraints of form
#    f(x) >= 0
# While Boyd uses
#   f(x) <= 0
# As such, they are defined in Boyd's format and converted by
# get_scipy_constraints script for use in scipy.optimize.minimize

def ineq1(x):
    # print("THIS IS INEQ1")
    # input()
    A = np.array([[0.25, 0],
                  [0, 1]], dtype=np.float32)
    b = np.array([[0.5],
                  [0]], dtype=np.float32)
    # print(A)
    # print("\n", b)
    # input()

    arg1 = -np.dot(x[:2], np.dot(A, x[:2]))
    arg2 = np.dot(x[:2], b)

    result = arg1 + arg2 + 3/4
    return -result

def ineq2(x):
    # print("THIS IS INEQ2")
    A = np.array([[5, 3],
                  [3, 5]], dtype=np.float32)
    b = np.array([[11/2],
                  [13/2]], dtype=np.float32)

    arg1 = -(1/8)*np.dot(x[2:], np.dot(A, x[2:]))
    arg2 = np.dot(x[2:], b)

    result = arg1 + arg2 - 35/2
    return -result

ineqConstraints = [ineq1, ineq2]

constraintList    = get_scipy_constraints(None, ineqConstraints, scipy=False)
constraintListSCP = get_scipy_constraints(None, ineqConstraints, scipy=True)
# exit()

initialX = [1,0, 2, 4]
initialX = feasibility(constraintListSCP, initialX, tolerance=ftol)

print("\nCheck initial point")
print("Initial X", initialX)
print("f_1(x) = {} <= 0".format(ineq1(initialX)))
print("f_2(x) = {} <= 0".format(ineq2(initialX)))
# for constraint in constraintList:
#     print("f_c(x) = {} <= 0".format(constraint['fun'](initialX)))
# input()


# optimizerList = [SteepestDescentBacktracking,
#                  NewtonRaphson,
#                  ConjugateGradient,
#                  QuasiNewtonDFP,
#                  ]
optimizerList = [SteepestDescentBacktracking]

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
for optimizer in optimizerList:
    # Get base results from Scipy
    optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
                                constraints=constraintListSCP)
    xRef      = optimResult.x
    fRef      = optimResult.fun
    fevalsRef = optimResult.nfev

    # Find optimum
    xOpt, fOpt, fevals = barrier_method(costFunction, constraintList, None, initialX,
                        optimizer=optimizer, interval=interval, ftol=ftol, maxIters=maxIters,
                         maxItersLS=maxItersLS, scipy=False)

    print("")
    print("x* ",    xRef)
    print("f(x*) ", fRef)
    print("fevals*: ", fevalsRef)
    print("")
    print("x ",    xOpt)
    print("f(x) ", fOpt)
    print("fevals: ", fevals)

    # print("\nCheck end point")
    # print("xOpt", xOpt)
    # print("f_1(x) = {} <= 0".format(ineq1(xOpt)))
    # print("f_2(x) = {} <= 0".format(ineq2(xOpt)))
    #
    # print("\nCheck1")
    # print("f_1(x)  = {} <= 0".format(ineq1(xOpt)))
    # print("f_2(x)  = {} <= 0".format(ineq2(xOpt)))
    # print("\nCheck2")
    # print("f_1(x)c = {} <= 0".format(constraintList[0]['fun'](xOpt)))
    # print("f_2(x)c = {} <= 0".format(constraintList[1]['fun'](xOpt)))
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
