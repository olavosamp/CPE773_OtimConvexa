import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func10
from libs.constrained_optim   import *
from libs.quasi_newton        import *
from libs.gradient_methods    import *
from libs.conjugate_direction import *


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

ineqConstraints = [ lambda x: -x[0],
                    lambda x: -x[1],
                    lambda x: -x[2],
                    lambda x: -x[3],
]

constraintList = get_scipy_constraints(None, ineqConstraints)

# BUG: SO ESTÁ PEGANDO O ULTIMO ELEMENTO DE X
initialX = np.array([-10,-10,-10,-10])
# initialX = feasibility(constraintList, initialX)

logBarrier = compose_logarithmic_barrier(constraintList)
logCheck = modified_log(-ineqConstraints[0](initialX)) + modified_log(-ineqConstraints[1](initialX)) + modified_log(-ineqConstraints[2](initialX)) + modified_log(-ineqConstraints[3](initialX))

print(logBarrier(initialX))
print(logCheck)

# f1 = lambda x: x+1
# f2 = lambda x: x**2
# f3 = lambda x: x+1.5
# f4 = lambda x: x/2
#
# fList = [f1, f2, f3, f4]
#
# def accum(f1, f2):
#     return lambda x: f1(x) + f2(x)
#
# oldH = lambda x: 0
# for f in fList:
#     h = accum(f, oldH)
#     oldH = h
#
# print(h(2))
