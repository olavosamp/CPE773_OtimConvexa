import autograd.numpy      as np
import scipy.optimize      as spo

from libs.functions        import func5
from libs.gradient_methods import *

xtol       = 1e-6
maxIters   = 200
maxItersLS = 200
function   = func5
interval   = [-1e15, 1e15]

# Q 5.7
initialX = [+4, +4]
# initialX = [+4, -4]
# initialX = [-4, +4]
# initialX = [-4, -4]

sd_backtrack = SteepestDescentBacktracking(function, initialX, interval=interval,
                                            xtol=xtol, maxIters=maxIters, maxItersLS=maxItersLS)
xOpt, fOpt, fevals = sd_backtrack.optimize()
print("Initial X:", initialX)
print("x*: ", xOpt)
print("f(x*): ", fOpt)
print("FEvals: ", fevals)

optimResult = spo.minimize(function, initialX, method='BFGS', tol=xtol)
xRef = optimResult.x
fRef = optimResult.fun

print("Ref x* = ", xRef)
print("Ref f(x*) = ", fRef)
print("Delta f(x) = ", np.abs(fOpt - fRef))
