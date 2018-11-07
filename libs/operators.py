import autograd.numpy as np

def norm2(x):
    # if np.ndim(x) > 1:
    #     raise ValueError("Expected dimension greater than 2, but received")
    # else:
    return np.sqrt(np.sum(np.power(x, 2)))

def positive_definite(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
