import scipy.optimize      as spo

def get_constraints(eqConstraints, ineqConstraints):
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
    for consFunc in ineqConstraints:
        cons = {'type': 'ineq',
        'fun': consFunc,
        }
        constraintList.append(cons)

    for consFunc in eqConstraints:
        cons = {'type': 'eq',
        'fun': consFunc,
        }
        constraintList.append(cons)
    return constraintList

def feasibility(constraintList, initialX, tolerance=1e-8):
    optimResult = spo.minimize(lambda x: 1, initialX, method='SLSQP', tol=tolerance,
                                constraints=constraintList)
    feasiblePoint = optimResult.x                    
    return feasiblePoint
