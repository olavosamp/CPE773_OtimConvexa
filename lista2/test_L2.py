import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs           as dirs
from libs.functions        import func6
from libs.gradient_methods import *

ftol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func6
interval   = [-1e15, 1e15]
savePath = dirs.results+"L2_Q4.xls"


# Q 5.20
initialX = np.array([-2, -1, 1, 2])

# initialX = [+4, -4]
# initialX = [-4, +4]
# initialX = [-4, -4]
# initialXList = [[+4., +4.],
#                 [+4., -4.],
#                 [-4., +4.],
#                 [-4., -4.],]

# xList      = []
# fxList     = []
# fevalsList = []
# deltaFList = []
#
# for initialX in initialXList:
#     sd_algorithm = NewtonRaphson(function, initialX, interval=interval, ftol=ftol,
#                                  maxIters=maxIters, maxItersLS=maxItersLS)
#     xOpt, fOpt, fevals = sd_algorithm.optimize()
#
#     print("Initial X:", initialX)
#     print("f(x*): ", fOpt)
#     print("x*: ", xOpt)
#     print("FEvals: ", fevals)
#
#     optimResult = spo.minimize(function, initialX, method='BFGS', tol=ftol)
#     xRef = optimResult.x
#     fRef = optimResult.fun
#     deltaF = np.abs(fOpt - fRef)
#
#     print("Delta f(x) = ", deltaF)
#     print("Ref x* = ", xRef)
#     print("Ref f(x*) = ", fRef)
#     print("CondVal: ", sd_algorithm.condVal)
#
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

sd_algorithm = GaussNewton(function, initialX, interval=interval, ftol=ftol,
                             maxIters=maxIters, maxItersLS=maxItersLS)
result = sd_algorithm.optimize()
print("fx\n", sd_algorithm.fx)
print("jacobian\n", sd_algorithm.jacobian)
print("gradient\n", sd_algorithm.gradient)
print("hessian\n", sd_algorithm.hessian)

# (x + 10*y) = -200*(x - z)^3
# 5*(x + 10*y) = -(y - 2*w)
# 5*(w - z) = 4*(y - 2*w)^3
# (w - z) = -40*(x - z)^3
