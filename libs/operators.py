import numpy as np

def norm2(x):
    # if np.ndim(x) > 1:
    #     raise ValueError("Expected dimension greater than 2, but received")
    # else:
    return np.sqrt(np.sum(np.power(x, 2)))
