import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.constrained_optim   import *
from libs.conjugate_direction import *


ftol       = 1e-8
maxIters   = 500
maxItersLS = 200
interval   = [-1e3, 1e3]

# Off-center paraboloid pointing towards +Z
def costFunction(x):
    return x[0]**2 + (x[1] - 1)**2 + 2

# Q 11.6
eqConstraintsFun   = [ lambda x: x[0] + x[1] -5
]

# eqConstraintsMat = {'A': np.array([[1, 2, 1, 2],
#                                    [1, 1, 2, 4]]),
#                     'b': np.array([[3],
#                                     [5]])}


# NOTE: Scipy uses inequality constraints of form
#    f(x) >= 0
# While Boyd uses
#   f(x) <= 0
# As such, they are defined in Boyd's format and converted by
# get_scipy_constraints script
ineqConstraints = [ lambda x: x[1] +1,
]

initialX = np.array([4, 4])

## I. Min f(x)
optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol)
                            #constraints=constraintList)
xRef = optimResult.x
fRef = optimResult.fun

algorithm = ConjugateGradient(costFunction, initialX, interval=interval, ftol=ftol,
                                 maxIters=maxIters, maxItersLS=maxItersLS)

xOpt, fOpt, fevals = algorithm.optimize()

print("\nProb I")
print("x* ",    xRef)
print("f(x*) ", fRef)
print("")
print("x ",    xOpt)
print("f(x) ", fOpt)


#
# ## II. min. f(x)
# #      s.t. x[1] + 1 <= 0
#
# constraintListII = get_scipy_constraints(None, ineqConstraints)
# optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
#                             constraints=constraintListII)
# xRef = optimResult.x
# fRef = optimResult.fun
#
# print("\nProb II")
# print("x* ",    xRef)
# print("f(x*) ", fRef)
#
#
# ## III. min. f(x)
# #      s.t. x[0] + x[1] -5 = 0
#
# constraintListIII = get_scipy_constraints(eqConstraintsFun, None)
# optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
#                             constraints=constraintListIII)
# xRef = optimResult.x
# fRef = optimResult.fun
#
# print("\nProb III")
# print("x* ",    xRef)
# print("f(x*) ", fRef)
#
# ## IV. min. f(x)
# #      s.t. x[1] + 1 <= 0
# #           x[0] + x[1] -5 = 0
#
# constraintListIV = get_scipy_constraints(eqConstraintsFun, ineqConstraints)
# optimResult = spo.minimize(costFunction, initialX, method='SLSQP', tol=ftol,
#                             constraints=constraintListIV)
# xRef = optimResult.x
# fRef = optimResult.fun
#
# print("\nProb IV")
# print("x* ",    xRef)
# print("f(x*) ", fRef)
