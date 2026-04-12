"""
FA model node classes.

This module contains node classes that are used by both FA_model and
likelihood/prior modules, avoiding circular imports.
"""
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import digamma, gammaln

if TYPE_CHECKING:
    from .FA_model import FAParams

from .utils import log_eps

EPS = 1e-20


class nodeFA_tau_m:
    """
    Class to define tau node (dim d, for one m)
    """

    def __init__(
        self, a0, b0, m, params: "FAParams", D: int = None, is_ctm: bool = False
    ):
        self.params = params
        self.m = m
        self.D = D if D is not None else (params.D[m] if m is not None else 0)
        self.is_ctm = is_ctm

        if not self.is_ctm:
            self.a0 = a0
            self.b0 = b0

            self.E_resid_squared_half = 0

            self.elbo = 0

        if self.is_ctm:
            self.Sigma0_inv = np.eye(self.D)

    def MB(self, y_m_node, w_m_node, z_node):
        self.w_m_node = w_m_node
        self.y_m_node = y_m_node
        self.z_node = z_node

        # we need MB for this, so it is here and not in init
        if not self.is_ctm:
            self.vi_a = (
                self.a0
                + (
                    self.params.N
                    - np.sum(np.ma.getmaskarray(self.y_m_node.data), axis=0)
                )
                / 2
            )
            self.vi_b = self.vi_a.copy()
            self.update_all_params()
            self.update_params()

    def update(self):
        self.update_params_w_z()
        self.vi_b = self.b0 + np.ma.sum(self.E_resid_squared_half, axis=0)

        self.update_params()

    def svi_update(self, indices: np.ndarray, rho: float):
        """
        Stochastic update for tau using a minibatch of samples.
        """
        if self.is_ctm:
            return
        if indices is None:
            indices = np.arange(self.params.N)
        self.update_params_w_z()
        batch_sum = np.ma.sum(self.E_resid_squared_half[indices], axis=0)
        vi_b_target = self.b0 + batch_sum
        self.vi_b = (1.0 - rho) * self.vi_b + rho * vi_b_target
        self.update_params()

    def update_all_params(self):
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        # self.update_params()

        self.kl_const = -self.D * (self.log_gamma_a0) + self.D * (
            self.a0 * log_eps(self.b0)
        )
        self.entropy_cons = (
            np.sum(self.vi_a)
            + np.sum(self.log_gamma_vi_a)
            + np.sum((1 - self.vi_a) * self.digamma_vi_a)
        )

    def update_params(self):
        mask = np.ma.getmaskarray(self.y_m_node.data)
        self.E_tau_1D = self.vi_a / self.vi_b
        self.E_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_tau_1D),
            mask=mask,
        )

        self.E_log_tau_1D = -log_eps(self.vi_b) + self.digamma_vi_a
        self.E_log_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_log_tau_1D),
            mask=mask,
        )

    def update_params_w_z(self):
        # d x n, sum over n

        # <(y_nd - \sum_k z_nk w_kd)**2>/2
        third_term_of_tau = (
            np.ma.array(
                self.w_m_node.E_w_z_squared, mask=np.ma.getmaskarray(self.y_m_node.data)
            )
            / 2
        )
        first_term_of_tau = self.y_m_node.data**2 / 2
        second_term_of_tau = -self.y_m_node.data * self.w_m_node.E_w_z

        self.E_resid_squared_half = (
            first_term_of_tau + second_term_of_tau + third_term_of_tau
        )

    def ELBO(self):
        kl = (
            self.kl_const
            + (self.a0 - 1) * np.sum(self.E_log_tau_1D)
            - self.b0 * np.sum(self.E_tau_1D)
        )
        entropy = self.entropy_cons - np.sum(log_eps(self.vi_b))
        self.elbo = kl + entropy


class nodeFA_y_m:
    def __init__(self, data_n, m, params: "FAParams", D: int = None):
        self.params = params
        self.m = m
        self.D = D if D is not None else (params.D[m] if m is not None else 0)

        self.data = data_n
        self.data_original = data_n

        self.elbo = 0

    def MB(self, w_m_node, tau_m_node):
        self.w_m_node = w_m_node
        self.tau_m_node = tau_m_node

    def ELBO(self):
        elbo = (
            -(self.params.N * self.D - np.sum(np.ma.getmaskarray(self.data)))
            * log_eps(2 * np.pi)
            / 2
            + np.ma.sum(self.tau_m_node.E_log_tau) / 2
            - np.ma.sum(self.tau_m_node.E_tau * self.tau_m_node.E_resid_squared_half)
        )
        self.elbo = elbo
