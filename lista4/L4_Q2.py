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

# Q2
# Cost function
def costFunction(x):
    return (x[0] - x[2])**2 + (x[1] - x[3])**2
    # return np.linalg.norm(x[:2]- x[2:], ord=2)**2

# Constraints defined in format:
#    f(x) <= 0
def ineq1(x):
    A = np.array([[0.25, 0],
                  [0, 1]], dtype=np.float32)
    b = np.array([[0.5],
                  [0]], dtype=np.float32)

    arg1 = -np.dot(x[:2], np.dot(A, x[:2]))
    arg2 = np.dot(x[:2], b)

    result = arg1 + arg2 + 3/4
    return -result

def ineq2(x):
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

initialX = [1,0, 2, 4]
initialX = feasibility(constraintListSCP, initialX, tolerance=ftol)

print("\nCheck initial point")
print("Initial X", initialX)
print("f_1(x) = {} <= 0".format(ineq1(initialX)))
print("f_2(x) = {} <= 0".format(ineq2(initialX)))

optimizerList = [(SteepestDescentBacktracking,"Steepest Descent w/ Backtracking"),
                 (NewtonRaphson,              "Newton Raphson"),
                 (ConjugateGradient,          "Conjugate Gradient"),
                 (QuasiNewtonDFP,             "Quasi-Newton DFP"),
                ]

# Get reference results from Scipy
# Scipy optimization is broken due to get_scipy_constraints error.
# optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
# constraints=constraintListSCP)
# xRef      = optimResult.x
# fRef      = optimResult.fun
# fevalsRef = optimResult.nfev

# Scipy minimize SLSQP Results
fRef        = 2.916580917967838
fevalsRef   = 63

print("")
# print("x* ",    xRef)
print("f(x*) ", fRef)
print("fevals*: ", fevalsRef)

xList      = []
fxList     = []
fevalsList = []
deltaFList = []
optNameList= []
for optTuple in optimizerList:
    optimizer = optTuple[0]
    optName   = optTuple[1]

    # Find optimum
    xOpt, fOpt, fevals = barrier_method(costFunction, constraintList, None, initialX,
                        optimizer=optimizer, interval=interval, ftol=ftol, maxIters=maxIters,
                         maxItersLS=maxItersLS, scipy=False)

    deltaF = np.abs(fOpt - fRef)
    print("")
    print("x ",    xOpt)
    print("f(x) ", fOpt)
    print("fevals: ", fevals)
    print("DeltaF: ", deltaF)

    print("\nCheck end point")
    print("xOpt", xOpt)
    print("f_1(x) = {} <= 0".format(ineq1(xOpt)))
    print("f_2(x) = {} <= 0".format(ineq2(xOpt)))

    optNameList.append(optName)
    xList.append(xOpt)
    fxList.append(fOpt)
    fevalsList.append(fevals)
    deltaFList.append(deltaF)

xList = np.array(xList)
resultsDf = pd.DataFrame(data={"Optimizer": optNameList,
                                "\\(x^*_1\\)"    : xList[:, 0],
                                "\\(x^*_2\\)"    : xList[:, 1],
                                "\\(x^*_3\\)"    : xList[:, 2],
                                "\\(x^*_4\\)"    : xList[:, 3],
                                "\\(f(x^*)\\)"   : fxList,
                                "FEvals"  : fevalsList,
                                "\\(\\Delta F\\)" : deltaFList})

print(resultsDf)
resultsDf.to_excel(savePath)
