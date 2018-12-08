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

def L4_Q3_f0(x):
    A = np.array([[4,0,0],
                  [0,1,-1],
                  [0,-1,1]], dtype=np.float32)
    b = np.array([-8,-6,-6], dtype=np.float32)
    b.shape = (3,1)
    f0 = 0.5*np.dot(x.T, np.dot(A, x)) + np.dot(x.T, b)
    return f0

def L4_Q3_ineq(x):
    # -x <= 0
    return -x

L4_Q3_eqMat = {'A': np.array([[1, 1, 1]]),
               'b': np.array([[3]])
}


def L4_Q4_ineq(x):
    F0 = np.array([[0.50,0.55,0.33,0.238],
                   [0.55,0.18,-1.18,-0.40],
                   [0.33,-1.18,-0.94,1.46],
                   [2.38,-0.40,1.46,0.17],
                   ], dtype=np.float32)
    F1 = np.array([[5.19,1.54,1.56,-2.80],
                   [1.54,2.20,0.39,-2.50],
                   [1.56,0.39,4.43,1.77],
                   [-2.80,-2.50,1.77,4.06],
                   ], dtype=np.float32)
    F2 = np.array([[-1.11,0.00,-2.12,0.38],
                   [0.00,1.91,-0.25,-0.58],
                   [-2.12,-0.25,-1.49,1.45],
                   [0.38,-0.58,1.45,0.63],
                   ], dtype=np.float32)
    F3 = np.array([[2.69,-2.24,-0.21,-0.74],
                   [-2.24,1.77,1.16,-2.01],
                   [-0.21,1.16,-1.82,-2.79],
                   [-0.74,-2.01,-2.79,-2.22],
                   ], dtype=np.float32)
    F4 = np.array([[0.58,-2.19,1.69,1.28],
                   [-2.19,-0.05,-0.01,0.91],
                   [1.69,-0.01,2.56,2.14],
                   [1.28,0.91,2.14,-0.75],
                   ], dtype=np.float32)

    result = F0 + x[0]*F1 + x[1]*F2 + x[2]*F3 + x[3]*F2
    return result  # Scipy
    # return -result # Real optim

def L4_Q4_f0(x):
    c = np.array([1, 0, 2, -1], dtype=np.float32)
    return np.dot(c, x)

def L4_Q5_f0(x):
    arg1 = 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2
    arg2 = (x[2] - 1)**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2)
    arg3 = 19.8*(x[1] - 1)*(x[3] - 1)
    return arg1 + arg2 + arg3

def L4_Q5_ineq1(x):
    # x_i - 10 <= 0
    arg = x - 10
    return arg

def L4_Q5_ineq2(x):
    # -x_i - 10 <= 0
    arg = -x - 10
    return arg
