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
interval   = [-1e3, 1e3]
savePath = dirs.results+"L4_Q1.xls"

# Q 11.6
costFunction   = func10
eqConstraintsFun   = [ lambda x: x[0] + 2*x[1] + x[2] + 2*x[3] - 3,
                       lambda x: x[0] + x[1] + 2*x[2] + 4*x[3] - 5,
]

eqConstraintsMat = {'A': np.array([[1, 2, 1, 2],
                                   [1, 1, 2, 4]], dtype=np.float32),
                    'b': np.array([[3],
                                    [5]], dtype=np.float32)}

# NOTE: Scipy uses inequality constraints of form
#    f(x) >= 0
# While Boyd uses
#   f(x) <= 0
# As such, they are defined in Boyd's format and converted by
# get_scipy_constraints script
ineqConstraints = [ lambda x: -x,]

constraintList    = get_scipy_constraints(None, ineqConstraints, scipy=False)
initialX = np.array([1,0, 0, 1], dtype=np.float32)


# constraintListSCP = get_scipy_constraints(eqConstraintsFun, ineqConstraints, scipy=False)
# initialX = feasibility(constraintListSCP, initialX, tolerance=ftol)

print("\nCheck initial point")
print("Initial X", initialX)
print("f(x0): ", costFunction(initialX))
print("f_1(x) = {} <= 0".format(ineqConstraints[0](initialX)))
# print("f_2(x) = {} <= 0".format(L4_Q3_ineq2(initialX)))

optimizerList = [(SteepestDescentBacktracking,"Steepest Descent"),
                 (NewtonRaphson,              "Newton Raphson"),
                 (ConjugateGradient,          "Conjugate Gradient"),
                 (QuasiNewtonDFP,             "Quasi-Newton DFP"),
                ]

# Get base results from Scipy
# Scipy optimization is broken due to get_scipy_constraints error.
# optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
#     constraints=constraintListSCP)
# xRef      = optimResult.x
# fRef      = optimResult.fun
# fevalsRef = optimResult.nfev
# print("\nx* ",    xRef)

# Scipy minimize SLSQP Results
# Updated Q
fRef        = 1.6666666666666672
fevalsRef   = 18

print("")
print("f(x*) ", fRef)
print("fevals*: ", fevalsRef)
# exit()
xList      = []
fxList     = []
fevalsList = []
deltaFList = []
optNameList= []
for optTuple in optimizerList:
    optimizer = optTuple[0]
    optName   = optTuple[1]

    # Find optimum
    xOpt, fOpt, fevals = barrier_method(costFunction, constraintList, eqConstraintsMat, initialX,
                        optimizer=optimizer, interval=interval, ftol=ftol, maxIters=maxIters,
                         maxItersLS=maxItersLS, scipy=False)
    fOpt = np.squeeze(fOpt)
    deltaF = np.abs(fOpt - fRef)
    print("")
    print("x ",    xOpt)
    print("f(x) ", fOpt)
    print("fevals: ", fevals)
    print("DeltaF: ", deltaF)

    print("\nCheck end point")
    print("xOpt", xOpt)
    print("f_1(x) = {} <= 0".format(ineqConstraints[0](initialX)))

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
