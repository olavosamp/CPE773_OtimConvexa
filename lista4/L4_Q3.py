import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import L4_Q3_f0, L4_Q3_ineq, L4_Q3_eqMat
from libs.constrained_optim   import *
from libs.quasi_newton        import *
from libs.gradient_methods    import *
from libs.conjugate_direction import *


ftol       = 1e-8
maxIters   = 200
maxItersLS = 2000
interval   = [-1e8, 1e8]
savePath = dirs.results+"L4_Q3.xls"

# Q3
# Cost function
costFunction = L4_Q3_f0

# Constraints defined in format:
#    f(x) <= 0
ineqConstraints  = [L4_Q3_ineq]

constraintList    = get_scipy_constraints(None, ineqConstraints, scipy=False)
initialX = np.array([0,3, 0], dtype=np.float32)


# eqConstraintsSCP = [lambda x: x[0] + x[1] + x[2] -3]
# constraintListSCP = get_scipy_constraints(eqConstraintsSCP, ineqConstraints, scipy=False)
# initialX = feasibility(constraintListSCP, initialX, tolerance=ftol)
# print(initialX)
# print(costFunction(initialX))
# input()


print("\nCheck initial point")
print("Initial X", initialX)
print("f_1(x) = {} <= 0".format(L4_Q3_ineq(initialX)))
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
# Updated Q3
fRef        = -18.5
fevalsRef   = 31

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
    xOpt, fOpt, fevals = barrier_method(costFunction, constraintList, L4_Q3_eqMat, initialX,
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
    print("f_1(x) = {} <= 0".format(L4_Q3_ineq(initialX)))

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
                                "\\(f(x^*)\\)"   : fxList,
                                "FEvals"  : fevalsList,
                                "\\(\\Delta F\\)" : deltaFList})

print(resultsDf)
resultsDf.to_excel(savePath)
