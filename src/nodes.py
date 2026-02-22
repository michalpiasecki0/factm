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

from .enums import Likelihood
from .utils import log_eps

EPS = 1e-20


class nodeFA_tau_m:
    """
    Class to define tau node (dim d, for one m)
    """

    def __init__(self, a0, b0, m, params: "FAParams", likelihood=None):
        self.params = params
        self.likelihood = likelihood

        self.m = m

        if not (self.params.likelihoods[self.m] == Likelihood.CTM):
            self.a0 = a0
            self.b0 = b0

            self.E_resid_squared_half = 0

            self.elbo = 0

        if self.params.likelihoods[self.m] == Likelihood.CTM:
            self.Sigma0_inv = np.eye(self.params.D[self.m])

    def MB(self, y_m_node, w_m_node, z_node):
        self.w_m_node = w_m_node
        self.y_m_node = y_m_node
        self.z_node = z_node

        # we need MB for this, so it is here and not in init
        if not (self.params.likelihoods[self.m] == Likelihood.CTM):
            self.vi_a = (
                self.a0 + (self.params.N - np.sum(self.y_m_node.data.mask, axis=0)) / 2
            )
            self.vi_b = self.vi_a.copy()
            self.update_all_params()
            self.update_params()

    def update(self):
        self.update_params_w_z()
        self.vi_b = self.b0 + np.ma.sum(self.E_resid_squared_half, axis=0)

        self.update_params()

    def update_all_params(self):
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        # self.update_params()

        self.kl_const = -self.params.D[self.m] * (self.log_gamma_a0) + self.params.D[
            self.m
        ] * (self.a0 * log_eps(self.b0))
        self.entropy_cons = (
            np.sum(self.vi_a)
            + np.sum(self.log_gamma_vi_a)
            + np.sum((1 - self.vi_a) * self.digamma_vi_a)
        )

    def update_params(self):
        self.E_tau_1D = self.vi_a / self.vi_b
        self.E_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_tau_1D),
            mask=self.y_m_node.data.mask,
        )

        self.E_log_tau_1D = -log_eps(self.vi_b) + self.digamma_vi_a
        self.E_log_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_log_tau_1D),
            mask=self.y_m_node.data.mask,
        )

    def update_params_w_z(self):
        # d x n, sum over n

        # <(y_nd - \sum_k z_nk w_kd)**2>/2
        third_term_of_tau = (
            np.ma.array(self.w_m_node.E_w_z_squared, mask=self.y_m_node.data.mask) / 2
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
    def __init__(self, data_n, m, params: "FAParams"):
        self.params = params

        self.m = m

        # there is no difference between data and data_original
        # for likelihood == "normal"
        # however, for likelihood == 'Bernoulli", data_original is 0/1 and data is
        # an approximation used for updates
        self.data = data_n
        self.data_original = data_n

        self.elbo = 0

    def MB(self, w_m_node, tau_m_node):
        self.w_m_node = w_m_node
        self.tau_m_node = tau_m_node

    def ELBO(self):
        elbo = (
            -(self.params.N * self.params.D[self.m] - np.sum(self.data.mask))
            * log_eps(2 * np.pi)
            / 2
            + np.ma.sum(self.tau_m_node.E_log_tau) / 2
            - np.ma.sum(self.tau_m_node.E_tau * self.tau_m_node.E_resid_squared_half)
        )
        self.elbo = elbo
