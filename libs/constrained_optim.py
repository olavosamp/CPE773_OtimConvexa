import scipy.optimize      as spo

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
    xLen = eqConstraintsMat['A'].shape[1]
    bLen = eqConstraintsMat['b'].shape[0]

    optimResult = spo.linprog(np.zeros(xLen), A_eq=eqConstraintsMat['A'], b_eq=eqConstraintsMat['b'])
    x_hat = optimResult.x

    # Result check
    check = eqConstraintsMat['A'] @ x_hat
    for i in range(bLen):
        if check[i] != eqConstraintsMat['b'][i]:
            raise ValueError("Error in reference point calculation: x_hat does not satisfy Ax = b.")
