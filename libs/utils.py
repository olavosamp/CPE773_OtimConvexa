import autograd.numpy as np

# TODO: Check if
#   f_i < 0, or
#   f_i <= 0
# should be used.
def modified_log(x):
    if (x <= 0).any():   # Option between x <= 0 and x < 0
        return -np.inf
    else:
        return np.sum(np.log(x))
