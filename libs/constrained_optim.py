import scipy               as scp
import scipy.optimize      as spo
import autograd.numpy      as np
from libs.utils            import modified_log

# def wrap_eq_constraints(fun, mat):
#     constraint = {'fun': fun,
#                   'A': mat['A'],
#                   'b': mat['b']}
#     return constraint


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
    funcList = []
    for constraint in constraintList:
        if constraint['type'] == 'ineq':
            funcList.append(constraint['fun'])

    # Check if constraint list was empty
    if len(funcList ) == 0:
        print(funcList)
        raise ValueError("No inequality constraints found.")

    # Auxiliary accumulator function for composition
    def accum(f1, f2):
        return lambda x: f1(x) + f2(x)

    # Compose all constraint functions in one
    oldLogBarrier = lambda x: 0
    for func in funcList:
        newTerm = lambda x: modified_log(-func(x))
        # newTerm = lambda x: func(x)**3
        logBarrier = accum(oldLogBarrier, newTerm)
        oldLogBarrier = logBarrier

    def logBarrier2(x):
        print(x)
        return logBarrier(x)
    return logBarrier2
