"""
Numerical helpers for variational inference (log/clamp, entropy terms).
"""
import numpy as np


def log_eps(x, eps: float = 1e-20) -> np.ndarray:
    """log(max(x, eps)) — avoids log(0) in ELBO and prior updates."""
    return np.log(np.maximum(x, eps))


def xlogx(x, eps0: float = 1e-20, eps1: float = 1e-20) -> np.ndarray:
    """x * log(x) for x in (0, 1), with clamping near 0 and 1."""
    return x * log_eps(np.minimum(x, 1 - eps1), eps0)
