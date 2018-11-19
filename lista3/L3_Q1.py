import pandas              as pd

import autograd.numpy      as np
# import scipy.optimize      as spo

import libs.dirs              as dirs
from libs.functions           import func7, func8
from libs.conjugate_direction import *

xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func7
interval   = [-10, 10]
savePath = dirs.results+"L3_Q1.xls"

xLen = 16
initialX = np.random.uniform(interval[0], interval[1], size=(xLen))

# Q 6.1
cd_alg = ConjugateGradient(function, initialX, interval=interval, xtol=xtol,
                            maxIters=maxIters, maxItersLS=maxItersLS)
xOpt, fOpt, fevals = cd_alg.optimize()

print("Initial X:", initialX)
print("f(x*): ", fOpt)
print("x*: ", xOpt)
print("FEvals: ", fevals)

xRef   = np.zeros(xLen)
fRef   = func7(xRef)
deltaF = np.abs(fOpt - fRef)

print("Ref x* = ", xRef)
print("Ref f(x*) = ", fRef)
print("Delta f(x) = ", deltaF)
