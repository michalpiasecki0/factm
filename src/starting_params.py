"""
Starting parameter initialization functions.

This module contains functions for initializing starting parameters
that are used by both FA_model and prior modules, avoiding circular imports.
"""
import numpy as np


def starting_params_z(starting_params, N, K):
    """Initialize starting parameters for Z node."""
    if "z_mu" in starting_params.keys():
        z_mean = 1 * starting_params["z_mu"]
    else:
        z_mean = np.random.normal(size=(N, K))

    if "z_var" in starting_params.keys():
        z_var = starting_params["z_var"]
    else:
        z_var = np.ones((N, K))

    return z_mean, z_var


def starting_params_hat_w_m(starting_params, key_M, D, K):
    """Initialize starting parameters for hat_w_m node."""
    starting_params_m = starting_params[key_M]

    if "w_mu" in starting_params_m.keys():
        w_mean = 1 * starting_params_m["w_mu"]
    else:
        w_mean = np.random.normal(size=(D, K))

    if "w_var" in starting_params_m.keys():
        w_var = 1 * starting_params_m["w_var"]
    else:
        w_var = np.ones((D, K))

    return w_mean, w_var


def starting_params_s_m(starting_params, key_M, D, K):
    """Initialize starting parameters for s_m node."""
    starting_params_m = starting_params[key_M]

    if "s_lambda" in starting_params_m.keys():
        s_lambda = 1.0 * starting_params_m["s_lambda"]
    else:
        # start with p(s=1) > 0.999
        s_lambda = 10.0 * np.ones((D, K))

    return s_lambda
