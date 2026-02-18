"""
Automatic Relevance Determination with Spike and Slab (ARD_SS) prior for weights.
"""
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import betaln, digamma

if TYPE_CHECKING:
    from FACTMpy.FA_model import FAParams

from FACTMpy.utils import log_eps

EPS = 1e-20


class nodeFA_s_m:
    """
    Class to update variational params of S node
    (sparsity: SS - spike and slab) (d times k, for one m)
    """

    def __init__(self, vi_lambda, m, params: "FAParams"):
        self.params = params

        self.m = m

        self.vi_lambda = vi_lambda
        self.vi_gamma = 1 / (1 + np.exp(-vi_lambda))

    def MB(self):
        pass

    def update_k(self, k, nominator, denominator, E_alpha, E_log_LR_theta):
        lambda_k = (
            E_log_LR_theta
            + log_eps(E_alpha) / 2
            + log_eps(denominator) / 2
            + (nominator**2) / (2 * denominator)
        )

        # max value for lambda -> if it is reached, then P(S=1) = 1
        # if np.any(lambda_k > -np.log(EPS)):
        lambda_k[lambda_k > -np.log(EPS)] = -np.log(EPS)
        # min value for lambda -> if it is reached, then P(S=1) = 1/D_m
        # if np.any(lambda_k < np.log(self.D[self.m]-1)):
        lambda_k[lambda_k < np.log(self.params.D[self.m] - 1)] = np.log(
            self.params.D[self.m] - 1
        )

        self.vi_lambda[:, k] = lambda_k
        self.vi_gamma[:, k] = 1 / (1 + np.exp(-lambda_k))


class nodeFA_theta_m:
    """
    Class to define theta node (k, for one m)
    """

    def __init__(self, a0, b0, vi_a, vi_b, m, params: "FAParams"):
        self.params = params

        self.m = m

        self.a0 = a0
        self.b0 = b0

        self.vi_a = vi_a
        self.vi_b = vi_b

        self.update_params()

        self.elbo = 0

    def MB(self, s_m_node):
        self.s_m_node = s_m_node

    def update_k(self, k):
        sum_sdk = np.sum(self.s_m_node.vi_gamma[:, k])
        self.vi_a[k] = self.a0 + sum_sdk
        self.vi_b[k] = self.b0 - sum_sdk + self.params.D[self.m]

        self.update_params()

    def update_params(self):
        self.E_log_theta = digamma(self.vi_a) - digamma(self.vi_a + self.vi_b)
        self.E_log_1minustheta = digamma(self.vi_b) - digamma(self.vi_a + self.vi_b)
        self.E_log_LR = self.E_log_theta - self.E_log_1minustheta

    def ELBO(self):
        kl = (
            np.sum((self.a0 - 1) * self.E_log_theta)
            + np.sum((self.b0 - 1) * self.E_log_1minustheta)
            - np.sum(betaln(self.a0, self.b0))
        )
        entropy = (
            -np.sum((self.vi_a - 1) * self.E_log_theta)
            - np.sum((self.vi_b - 1) * self.E_log_1minustheta)
            + np.sum(betaln(self.vi_a, self.vi_b))
        )
        self.elbo = kl + entropy
