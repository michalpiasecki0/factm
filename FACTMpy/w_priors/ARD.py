"""
Automatic Relevance Determination (ARD) prior for weights.
"""
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import digamma, gammaln

if TYPE_CHECKING:
    from FACTMpy.FA_model import FAParams

from FACTMpy.utils import log_eps

EPS = 1e-20


class nodeFA_hat_w_m:
    """
    Class to update variational params of W_hat node
    (sparsity: ARD) (d times k, for one m)
    """

    def __init__(self, vi_mu, vi_var, m, params: "FAParams"):
        self.params = params

        self.m = m

        self.vi_mu = vi_mu
        self.vi_var = vi_var

    def MB(self):
        pass

    def update_k(self, k, nominator, denominator):
        self.vi_mu[:, k] = nominator / denominator
        self.vi_var[:, k] = 1 / denominator


class nodeFA_alpha_m:
    """
    Class to define alpha node (k, for one m)
    """

    def __init__(self, a0, b0, m, params: "FAParams"):
        self.params = params

        self.m = m

        self.a0 = a0
        self.b0 = b0

        # start from E_alpha = 1
        # update of vi_a:
        self.vi_a = a0 + self.params.D[m] * np.ones(self.params.K) / 2
        self.vi_b = self.vi_a.copy()

        self.update_all_params()

        self.elbo = 0

    def MB(self, hat_w_m_node, s_m_node, w_m_node):
        self.hat_w_m_node = hat_w_m_node
        self.s_m_node = s_m_node
        self.w_m_node = w_m_node

    def update_k(self, k):
        self.vi_b[k] = self.b0 + np.sum(self.w_m_node.E_hat_w_squared[:, k]) / 2

        # alpha small -> w_hat big TBD - check
        if self.vi_a[k] / self.vi_b[k] < EPS:
            self.vi_b[k] = self.vi_a[k] / EPS
        # alpha big -> w_hat small and potentially not significant
        if self.vi_b[k] / (self.vi_a[k] - 1) < EPS:
            self.vi_b[k] = (self.vi_a[k] - 1) * EPS

        self.update_params()

    def update_params(self):
        self.E_alpha = self.vi_a / self.vi_b
        self.E_inv_alpha = self.vi_b / (self.vi_a - 1)
        self.E_log_alpha = -log_eps(self.vi_b) + self.digamma_vi_a

    def update_all_params(self):
        # including params that are const.
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        self.update_params()

        self.kl_const = -self.params.K * (self.log_gamma_a0) + self.params.K * (
            self.a0 * log_eps(self.b0)
        )
        self.entropy_const = (
            np.sum(self.vi_a)
            + np.sum(self.log_gamma_vi_a)
            + np.sum((1 - self.vi_a) * self.digamma_vi_a)
        )

    def ELBO(self):
        kl = (
            self.kl_const
            + (self.a0 - 1) * np.sum(self.E_log_alpha)
            - self.b0 * np.sum(self.E_alpha)
        )
        entropy = self.entropy_const - np.sum(log_eps(self.vi_b))
        self.elbo = kl + entropy
