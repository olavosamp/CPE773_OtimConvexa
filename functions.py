import autograd.numpy as np

def quadratic(x):
    return x**2 - 4*x -4

def poly1(x):
    # f(x) = −5x5 + 4x4 − 12x3 + 11x2 − 2x + 1
    return -5*x**5 + 4*x**4 - 12*x**3 + 11*x**2 - 2*x + 1

def func2(x):
    # f(x) = ln2(x − 2) + ln2(10 − x) − x0.2
    if x < 6.:
        x = 6.
    if x > 9.9:
        x = 9.9

    value = np.log(x - 2)**2 + np.log(10 - x)**2 - x**0.2
    # if np.isnan(value):
    #     return np.inf

    return value

def func3(x):
    # f(x) = −3x sin 0.75x + e−2x
    if x < 0.:
        x = 0.
    if x > 2*np.pi:
        x = 2*np.pi
    return -3*x*np.sin(0.75*x) + np.exp(-2*x)

def func4(x):
    xLen = np.shape(x)[0]
    if xLen < 2:
        raise ValueError("Input dimensions doesn't match. Expected 2, received {}.".format(xLen))
    return 0.7*(x[0]**4) - 8*(x[0]**2) + 6*(x[1]**2) + np.cos(x[0]*x[1]) - 8*x[0]
