import scipy               as scp
import scipy.optimize      as spo
import autograd.numpy      as np

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

    # newFunc = lambda z: np.dot(F, z) + x_hat

    return F, x_hat


def compose_eq_cons_func(func, F, x_hat):
    newFunc = lambda z: np.dot(F, z) + x_hat
    return newFunc
