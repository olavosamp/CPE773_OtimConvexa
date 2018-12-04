import scipy                  as scp
import scipy.optimize         as spo
import autograd.numpy         as np
from libs.utils               import modified_log
from libs.conjugate_direction import ConjugateGradient

# def wrap_eq_constraints(fun, mat):
#     constraint = {'fun': fun,
#                   'A': mat['A'],
#                   'b': mat['b']}
#     return constraint

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


def get_scipy_constraints(eqConstraintsFun, ineqConstraints):
    '''
        Args:
            eqConstraints, ineqConstraints:
               List of functions corresponding to the equality and inequality
                constraints, respectively.

        Returns:
            constraintList:
               List of dicts with the input constraints
    '''
    constraintList = []
    if ineqConstraints != None:
        for consFunc in ineqConstraints:
            cons = {'type': 'ineq',
            'fun': consFunc,
            }
            constraintList.append(cons)

    if eqConstraintsFun != None:
        for consFunc in eqConstraintsFun:
            cons = {'type': 'eq',
            'fun': consFunc,
            }
            constraintList.append(cons)
    assert len(constraintList) != 0, "No constraints were given. Please input valid equality and/or inequality constraints."

    return constraintList


def feasibility(constraintList, initialX, tolerance=1e-8):
    '''
        Returns a feasible point that satisfies the constraints in constraintList.
    '''
    optimResult = spo.minimize(lambda x: 1, initialX, method='SLSQP', tol=tolerance,
                                constraints=constraintList)
    feasiblePoint = optimResult.x
    return feasiblePoint


def eq_constraint_elimination(func, eqConstraintsMat):
    '''
        Eliminates equality constraints Ax = b by parametrization of x as Fz + x_hat.
        F is the nullspace of matrix A, such as AFz = 0, where z is any vector.
        Indeed, AF = 0. x_hat is any viable x that satisfies the constraint.

        eqConstraintsMat is a dict with matrixes A, b in keys 'A' and 'b', respectively.
    '''
    A = eqConstraintsMat['A']
    b = eqConstraintsMat['b']

    xLen = A.shape[1]
    bLen = b.shape[0]

    optimResult = spo.linprog(np.zeros(xLen), A_eq=A, b_eq=b)
    x_hat = optimResult.x

    # Find a matrix F whose range is the nullspace of A
    F = scp.linalg.null_space(A)

    # Result check
    check = A @ x_hat
    for i in range(bLen):
        if check[i] != b[i]:
            raise ValueError("Error in equality constraint elimination: x_hat does not satisfy Ax = b.")

    if not(np.isclose(A @ F, 0).all()):
        raise ValueError("Error in equality constraint elimination: F miscalculated and AF != 0")

    return F, x_hat


def compose_eq_cons_func(func, F, x_hat):
    '''
        Create parametrized cost function, based on F and x_hat obtained by equality
        constraint elimination.

        Returns: parametrized function of z, f(z) = Fz + x_hat
    '''
    newFunc = lambda z: func(F @ z + x_hat)
    return newFunc


def compose_logarithmic_barrier(constraintList):
    '''
        Create logarithmic barrier function, based on the given inequality constraints
        f_i(x) <= 0.

        Returns:
            Logarithmic barrier function

            f(x) = log(-f_1(x)) + log(-f_2(x)) + ...
    '''
    funcList = retrieve_constraints(constraintList, 'eq')

    # Auxiliary accumulator function for composition
    def accum(f1, f2):
        return lambda x: f1(x) + f2(x)

    # Compose all constraint functions in one
    oldLogBarrier = lambda x: 0
    for func in funcList:
        newTerm = lambda w: modified_log(-func(w))

        logBarrier = accum(oldLogBarrier, newTerm)
        oldLogBarrier = logBarrier

    return logBarrier


def barrier_method(func, constraintList, initialX, interval=[-1e15, 1e15], ftol=1e-6, maxIters=1e3, maxItersLS=200):
    t_0 = 10
    mu  = 5
    epsilon = ftol

    logBarrier    = compose_logarithmic_barrier(constraintList)
    funcList      = retrieve_constraints(constraintList, 'eq')

    eqConstraints = delete_constraints(constraintList, 'ineq')


    m = len(funcList)
    t = t_0
    fevals = 0
    iter   = 0
    while iter < maxIters - 1:
        print("Outer iteration ", iter)
        # Compose new centering function
        centerFunc = lambda x: t*func(x) + logBarrier(x)

        # Centering Step
        # Using Scipy optimizer for testing
        optimizer = spo.minimize(centerFunc, x, method='Newton-CG', tol=ftol,
                                    constraints=eqConstraints)
        x       = optimizer.x
        fevals += optimizer.nfev

        # Verify stopping conditions
        if m/t < epsilon:
            print("Stopping condition reached. Algorithm terminating.")
            xOpt = x
            fOpt = func(xOpt)
            return xOpt, fOpt
        else:
            t    = mu*t
            iter+= 1

    print("Algorithm did not converge.")
    xOpt = x
    fOpt = func(xOpt)
    return xOpt, fOpt
