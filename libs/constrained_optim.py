import scipy                  as scp
import scipy.optimize         as spo
import autograd.numpy         as np
from copy                     import copy
from autograd                 import grad
from libs.utils               import modified_log
from libs.conjugate_direction import ConjugateGradient
from libs.gradient_methods    import *
from libs.quasi_newton        import *


def delete_constraints(constraintList, type):
    newConstraintList = []
    for constraint in constraintList:

        if constraint['type'] == type:
            newConstraintList.append(constraint)

    return newConstraintList


def retrieve_constraints(constraintList, type):
    '''
        Receives Scipy-format constraint list and returns constraint function list.
        Type can be
            'eq'
            'ineq'
    '''
    funcList = []
    for constraint in constraintList:
        if constraint['type'] == type:
            funcList.append(constraint['fun'])

    # Check if constraint list was empty
    if len(funcList ) == 0:
        print(funcList)
        raise ValueError("No constraints of type {} were found.".format(type))
    return funcList


def get_scipy_constraints(eqConstraintsFun, ineqConstraints, scipy=False):
    '''
        Args:
            eqConstraints, ineqConstraints:
               List of functions corresponding to the equality and inequality
                constraints, respectively.

                Note: If scipy=True, the script receives inequality constraints of form
                    f_i(x) <= 0
                and converts to
                    f_i(x) >= 0,
                to fit Scipy formatting.

        Returns:
            constraintList:
               List of dicts with the input constraints
    '''
    # initialX = [1,0, 2, 4]
    # print("Start")
    # for cons in ineqConstraints:
    #     print(cons)

    constraintList = []
    if ineqConstraints != None:
        for consFunc in ineqConstraints:
            if scipy == True:
                # STILL BROKEN
                func = lambda x: -consFunc(x)
                cons = dict([('type', 'ineq'),
                             ('fun', func)])
            else:
                func = consFunc
                cons = dict([('type', 'ineq'),
                             ('fun', consFunc)])

            constraintList.append(cons)

    if eqConstraintsFun != None:
        for consFunc in eqConstraintsFun:
            cons = dict([('type', 'eq'),
                         ('fun', consFunc)])
            constraintList.append(cons)
    assert len(constraintList) != 0, "No constraints were given. Please input valid equality and/or inequality constraints."

    # print("\nEnd")
    # print(constraintList)
    # input()
    # for constrCheck in constraintList:
    #     print(constrCheck['fun'](initialX))
    return constraintList


def feasibility(constraintList, initialX, tolerance=1e-8):
    '''
        Returns a feasible point that satisfies the constraints in constraintList.
    '''
    optimResult = spo.minimize(lambda x: 1, initialX, method='SLSQP', tol=tolerance,
                                constraints=constraintList)
    feasiblePoint = optimResult.x
    return feasiblePoint


def eq_constraint_elimination_composer(eqConstraintsMat):
    '''
        Eliminates equality constraints Ax = b by parametrization of x as Fz + x_hat.
        F is the nullspace of matrix A, such as AFz = 0, where z is any vector.
        Indeed, AF = 0. x_hat is any viable x that satisfies the constraint.

        eqConstraintsMat is a dict with matrixes A, b in keys 'A' and 'b', respectively.
    '''
    A = eqConstraintsMat['A']
    b = eqConstraintsMat['b']

    if A.ndim == 1:
        xLen = 1
    else:
        xLen = A.shape[1]
    bLen = b.shape[0]

    x_hat = scp.linalg.lstsq(A,b)[0]

    # Find a matrix F whose range is the nullspace of A
    F = scp.linalg.null_space(A)

    # Result check
    check = np.dot(A, x_hat)
    for i in range(bLen):
        if check[i] != b[i]:
            print("A . x_hat\n")
            print(check)
            print("b\n")
            print(b)
            raise ValueError("Error in equality constraint elimination: x_hat does not satisfy Ax = b.")

    if not(np.isclose(A @ F, 0., atol=1e-7).all()):
        print("A @ F\n")
        print(A @ F)
        raise ValueError("Error in equality constraint elimination: F miscalculated and AF != 0")

    return F, x_hat


def eq_constraint_elimination_func(func, F, x_hat):
    '''
        Create parametrized cost function, based on F and x_hat obtained by equality
        constraint elimination.

        Returns: parametrized function of z, f(z) = Fz + x_hat
    '''

    # print(F.shape)
    # print(x_hat.shape)
    # input()
    newFunc = lambda z: np.squeeze(func(np.dot(F, z) + x_hat)[0])
    # newFunc = lambda z: func(np.dot(F, z))
    return newFunc


def compose_logarithmic_barrier(constraintList):
    '''
        Create logarithmic barrier function, based on the given inequality constraints
        f_i(x) <= 0.

        Returns:
            Logarithmic barrier function

            f(x) = log(-f_1(x)) + log(-f_2(x)) + ...
    '''
    funcList = retrieve_constraints(constraintList, 'ineq')

    # Compose all constraint functions in one
    def logBarrier(x):
        resultList = list(map(lambda f: modified_log(-f(x)), funcList))
        result = np.sum(resultList)

        return np.sum(result)


    # print("\nDebug logBarrier")
    # xBad = np.array([99, 99, 2, 4], dtype=np.float32)
    # print(xBad)
    # print("f_1(xb)", funcList[0](xBad))
    # print("f_2(xb)", funcList[1](xBad))
    #
    # print("logBarrier manual check: ", modified_log(-funcList[1](xBad)) + modified_log(-funcList[0](xBad)))
    # print("logBarrier(xBad): ", logBarrier(xBad))
    # input()
    return logBarrier


def eq_constraint_elimination(func, eqConstraintsMat, optimizer, initialX,
                              interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
    fevals = 0
    xLen = initialX.shape[0]              # n
    bLen = eqConstraintsMat['b'].shape[0] # p

    F, x_hat  = eq_constraint_elimination_composer(eqConstraintsMat)
    paramFunc = eq_constraint_elimination_func(func, F, x_hat)

    # Choose any initial Z
    initialZ = np.zeros((xLen - bLen,1)) # Must be (n - p)x1

    # Scipy optimizer
    # optimizer  = spo.minimize(paramFunc, initialZ, method='BFGS', tol=ftol)
    # zOpt       = optimizer.x
    # zFevals    = optimizer.nfev

    # Debug
    # print("F: ", F.shape)
    # print("A:", eqConstraintsMat['A'].shape)
    # print("b:", eqConstraintsMat['b'].shape)
    # print("x_hat: ", x_hat.shape)
    # print("initialZ: ", initialZ.shape)
    # print("paramFunc(z0): ", paramFunc(initialZ))
    # print("F@z + x_hat", (np.dot(F, initialZ) + x_hat).shape)
    # print((np.dot(F, initialZ) + x_hat))
    # print("zOpt: ", zOpt.shape)
    # print("xOpt", xOpt.shape)
    # print("fOpt", fOpt.shape)
    # print("Restr check:\nAx* = b\n{} = {}".format(
    # np.squeeze(np.dot(eqConstraintsMat['A'], xOpt)), eqConstraintsMat['b']
    # ))
    # input()
    # Find z* to minimize parametrized cost function
    algorithm = optimizer(paramFunc, initialZ, interval=interval, ftol=ftol,
                                     maxIters=maxIters, maxItersLS=maxItersLS)

    zOpt, _, zFevals = algorithm.optimize()
    zOpt.shape = (xLen - bLen, 1)


    fevals += zFevals
    xOpt    = np.squeeze(np.dot(F,  zOpt) + x_hat)
    fOpt    = np.squeeze(func(xOpt))

    return xOpt, fOpt, fevals


def barrier_method(func, constraintList, eqConstraintsMat, initialX, optimizer=NewtonRaphson, interval=[-1e15, 1e15],
                    ftol=1e-6, maxIters=1e3, maxItersLS=2000, scipy=False):
    t_0     = 0.5
    mu      = 2
    epsilon = ftol
    x       = initialX
    xLen    = initialX[0]

    logBarrier    = compose_logarithmic_barrier(constraintList)
    funcList      = retrieve_constraints(constraintList, 'ineq')

    # eqConstraints = delete_constraints(constraintList, 'ineq')

    m = len(funcList)

    t = t_0
    fevals = 0
    iter   = 0
    # Barreira
    while iter < int(maxIters - 1):
        # Compose new centering function
        centerFunc = lambda x: func(x) - (1/t)*logBarrier(x)


        ## Centering Step
        if scipy == True:
            # Using Scipy optimizer for testing
            optimResult  = spo.minimize(centerFunc, x, method='BFGS', tol=ftol)
            x            = optimResult.x
            centerFevals = optimResult.nfev
        else:
            if eqConstraintsMat == None:
                optm = optimizer(centerFunc, x, interval=interval, ftol=ftol,
                        maxIters=maxIters, maxItersLS=maxItersLS)
                x, _, centerFevals = optm.optimize()
            else:
                x, _, centerFevals = eq_constraint_elimination(centerFunc, eqConstraintsMat, optimizer,
                       x, interval=interval, ftol=ftol, maxIters=maxIters, maxItersLS=maxItersLS)


        # print("\nIter: ", iter)
        # print("x: ", x)
        # print("t: ", t)
        # print("func(x): ", func(x))
        # # print("norms(gradient): ", np.linalg.norm(gradient, ord=2))
        # print("logBarrier(x): ", logBarrier(x))
        # print("centerFunc(x): ", centerFunc(x))
        # print("centerCheck  : ", func(x) - (1/t)*logBarrier(x))
        # print("Restrição: {} <= 0".format(funcList[0](x)))
        # print("Restrição: {} <= 0".format(funcList[1](x)))

        fevals += centerFevals
        # Verify stopping conditions
        if m/t < epsilon:
            print("Stopping conditions reached. Barrier method terminating.")
            xOpt = x
            fOpt = func(xOpt)
            return xOpt, fOpt, fevals
        else:
            t    = mu*t
            iter+= 1

    print("Algorithm did not converge.")
    xOpt = x
    fOpt = np.squeeze(func(xOpt))
    return xOpt, fOpt, fevals
