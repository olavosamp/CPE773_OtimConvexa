import autograd.numpy as np

def modified_log(x):
    if x < 0:
        return np.inf
    else:
        return np.log(x)
