import numpy as np

def quadratic(x):
    return x**2 - 4*x -4

def poly1(x):
    # f(x) = −5x5 + 4x4 − 12x3 + 11x2 − 2x + 1
    return -5*x**5 + 4*x**4 - 12*x**3 + 11*x**2 - 2*x + 1

def poly2(x):
    # f(x) = ln2(x − 2) + ln2(10 − x) − x0.2
    return np.log(x - 2)**2 + np.log(10 - x)**2 - x**0.2

def poly3(x):
    # f(x) = −3x sin 0.75x + e−2x
    return -3*x*np.sin(0.75*x) + np.exp(-2*x)
