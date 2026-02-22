import numpy as np


def log_eps(x, eps=1e-20):
    return np.log(np.maximum(x, eps))


def xlogx(x, eps0=1e-20, eps1=1e-20):
    # x in (0,1)
    return x * log_eps(np.minimum(x, 1 - eps1), eps0)
