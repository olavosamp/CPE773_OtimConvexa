import pandas              as pd

import autograd.numpy      as np
import scipy.optimize      as spo

import libs.dirs           as dirs
from libs.functions        import func7, func8
from libs.gradient_methods import *

xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func7
interval   = [-1e15, 1e15]
savePath = dirs.results+"L3_Q1.xls"

x = np.ones(16)
print(func7(x))

x = np.ones(2)
print(func8(x))
