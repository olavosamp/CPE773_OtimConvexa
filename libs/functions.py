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
    else:
        return 0.7*(np.power(x[0], 4)) - 8*(np.power(x[0], 2)) + 6*(np.power(x[1], 2)) + np.cos(x[0]*x[1]) - 8*x[0]

def func5(x):
    xLen = np.shape(x)[0]
    if xLen < 2:
        raise ValueError("Input dimensions doesn't match. Expected 2, received {}.".format(xLen))
    else:
        return np.power(np.power(x[0], 2) + np.power(x[1], 2) -1, 2) + np.power(x[0] + x[1] - 1, 2)

def func6(x):
    xLen = np.shape(x)[0]
    if xLen < 4:
        raise ValueError("Input dimensions doesn't match. Expected 4, received {}.".format(xLen))
    else:
        f1 = (x[0] + 10*x[1])
        f2 = np.sqrt(5)*(x[2] + x[3])
        f3 = (x[1] -2*x[2])**2
        f4 = 10*((x[0] - x[3])**2)
        fVec = np.array([f1, f2, f3, f4])
        return fVec

def func6_scalar(x):
    xLen = np.shape(x)[0]
    if xLen < 4:
        raise ValueError("Input dimensions doesn't match. Expected 4, received {}.".format(xLen))
    else:
        f1 = (x[0] + 10*x[1])**2
        f2 = 5*(x[2] + x[3])**2
        f3 = (x[1] -2*x[2])**4
        f4 = 100*((x[0] - x[3])**4)

        return f1+f2+f3+f4

def func7(x):
    Q1 = np.array([[12,8,7,6],
                   [8,12,8,7],
                   [7,8,12,8],
                   [6,7,8,12],
    ])
    Q2 = np.array([[3,2,1,0],
                   [2,3,2,1],
                   [1,2,3,2],
                   [0,1,2,3],
    ])
    Q3 = np.array([[2,1,0,0],
                   [1,2,1,0],
                   [0,1,2,1],
                   [0,0,1,2],
    ])
    Q4 = np.eye(4)

    Q = np.block([[Q1,Q2,Q3,Q4],
                  [Q2,Q1,Q2,Q3],
                  [Q3,Q2,Q1,Q2],
                  [Q4,Q3,Q2,Q1],
    ])

    b = -np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])

    xLen = np.shape(x)[0]
    qLen = np.shape(Q)[1]
    assert xLen == qLen, "Input should be a vector of length {}, received".format(qLen, xLen)
    a = 0.5*(np.transpose(x) @ Q @ x)
    c = + b @ x
    return  a + c


def func8(x):
    xLen = np.shape(x)[0]
    assert xLen == 2, "Input should be a vector of length {}, received".format(2, xLen)
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def func9(x):
    xLen = np.shape(x)[0]
    assert xLen == 2, "Input should be a vector of length {}, received".format(2, xLen)
    return 5*(x[0]**2) - 9*x[0]*x[1] + 4.075*(x[1]**2) + x[0]


def func10(x):
    xLen = np.shape(x)[0]
    assert xLen == 4, "Input should be a vector of length {}, received".format(4, xLen)
    return x[0] + 1.5*x[1] + x[2] + x[3]

def L4_Q5_f0(x):
    arg1 = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2
    arg2 = (x[2] - 1)**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2)
    arg3 = 19.8*(x[1] - 1)*(x[3] - 1)
    return arg1 + arg2 + arg3

def L4_Q5_ineq1(x):
    # x_i - 10 <= 0
    arg = x - 10
    return -arg

def L4_Q5_ineq2(x):
    # -x_i - 10 <= 0
    arg = -x - 10
    return -arg
