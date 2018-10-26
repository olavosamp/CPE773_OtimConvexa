import numpy as np

def norm2(x):
    if np.ndim(x) > 1:
        raise ValueError("Norm of Matrix. Dimension of X greater than one")
    else:
        return np.sqrt(np.sum(np.power(x, 2)))
