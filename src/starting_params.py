"""
Starting parameter helpers shared by FA_model and W prior modules.
"""
import numpy as np


def starting_params_z(
    starting_params: dict, N: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize variational parameters for the Z node."""
    if "z_mu" in starting_params:
        z_mean = np.array(starting_params["z_mu"], copy=True)
    else:
        z_mean = np.random.normal(size=(N, K))

    if "z_var" in starting_params:
        z_var = np.array(starting_params["z_var"], copy=True)
    else:
        z_var = np.ones((N, K))

    return z_mean, z_var


def starting_params_hat_w_m(
    starting_params: dict, m: int, D: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize variational parameters for the hat_w node (ARD / ARD_SS)."""
    starting_params_m = starting_params.get(m, {})

    if "w_mu" in starting_params_m:
        w_mean = np.array(starting_params_m["w_mu"], copy=True)
    else:
        w_mean = np.random.normal(size=(D, K))

    if "w_var" in starting_params_m:
        w_var = np.array(starting_params_m["w_var"], copy=True)
    else:
        w_var = np.ones((D, K))

    return w_mean, w_var


def starting_params_s_m(starting_params: dict, m: int, D: int, K: int) -> np.ndarray:
    """Initialize spike-and-slab lambda parameters for the s_m node."""
    starting_params_m = starting_params.get(m, {})

    if "s_lambda" in starting_params_m:
        return np.array(starting_params_m["s_lambda"], copy=True)

    # lambda=10 -> P(s=1) > 0.999
    return 10.0 * np.ones((D, K))
