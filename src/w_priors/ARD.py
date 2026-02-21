"""
Automatic Relevance Determination (ARD) prior for weights.
"""
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import digamma, gammaln

from .base import WPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams, nodeFA_w_m

from ..enums import Likelihood
from ..starting_params import starting_params_hat_w_m
from ..utils import log_eps

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


class ARDWPrior(WPriorBase):
    """ARD prior for weights."""

    def create_nodes(
        self, m: int, params: "FAParams", starting_params: dict, D: int, K: int
    ) -> dict:
        """Create nodes for ARD prior."""

        key_tmp = "M" + str(m)
        w_mu, w_var = starting_params_hat_w_m(starting_params, key_tmp, D, K)
        node_hat_w_m = nodeFA_hat_w_m(w_mu, w_var, m, params=params)
        node_alpha_m = nodeFA_alpha_m(1e-14, 1e-14, m, params=params)

        return {
            "w_m_node_not_sparse": None,
            "hat_w_m_node": node_hat_w_m,
            "s_m_node": None,
            "alpha_m_node": node_alpha_m,
            "theta_m_node": None,
        }

    def update_w_k(
        self, w_node: "nodeFA_w_m", k: int, z_node: Any, y_node: Any, tau_node: Any
    ):
        """Update W node for factor k for ARD prior."""
        if w_node.params.likelihoods[w_node.m] != Likelihood.CTM:
            # For FA likelihoods (Normal/Bernoulli)
            nominator_second_term_tmp = np.dot(
                w_node.E_w, z_node.E_z.T * z_node.E_z[:, k]
            ).T
            nominator_second_term = nominator_second_term_tmp - np.outer(
                z_node.E_z[:, k] ** 2, w_node.E_w[:, k]
            )
            nominator = np.ma.sum(
                tau_node.E_tau
                * ((y_node.data.T * z_node.E_z[:, k]).T - nominator_second_term),
                axis=0,
            )

            denominator = (
                np.ma.dot(z_node.E_z_squared[:, k], tau_node.E_tau)
                + w_node.alpha_m_node.E_alpha[k]
            )

            w_node.hat_w_m_node.update_k(k, nominator, denominator)
            w_node.update_params()
            w_node.update_params_z()
            w_node.alpha_m_node.update_k(k)
        else:
            # For CTM likelihood
            E_zE_zk = np.dot(z_node.E_z.T, z_node.E_z[:, k])
            E_zE_zk[k] = 0
            EwE_zE_zk = np.dot(w_node.E_w, E_zE_zk)
            term1 = np.dot(tau_node.Sigma0_inv, EwE_zE_zk)

            Ez_squaredk = np.sum(z_node.E_z_squared[:, k])
            Sigma_inv_nodiag = tau_node.Sigma0_inv - np.diag(
                np.diag(tau_node.Sigma0_inv)
            )
            term2 = np.dot(Sigma_inv_nodiag, w_node.E_w[:, k]) * Ez_squaredk

            term3 = np.dot(
                z_node.E_z[:, k],
                np.ma.dot(y_node.data, tau_node.Sigma0_inv),
            )

            nominator = term3 - term1 - term2

            denominator = np.diag(
                w_node.alpha_m_node.E_alpha[k] * np.eye(w_node.params.D[w_node.m])
                + np.sum(z_node.E_z_squared[:, k]) * (tau_node.Sigma0_inv)
            )
            w_node.hat_w_m_node.update_k(k, nominator, denominator)
            w_node.update_params()
            w_node.update_params_z()
            w_node.alpha_m_node.update_k(k)

    def update_params(self, w_node: "nodeFA_w_m"):
        """Update E_w and E_w_squared for ARD prior."""
        w_node.E_w = w_node.hat_w_m_node.vi_mu
        w_node.E_w_squared = w_node.hat_w_m_node.vi_var + w_node.hat_w_m_node.vi_mu**2
        w_node.E_hat_w_squared = w_node.E_w_squared

    def compute_elbo(self, w_node: "nodeFA_w_m") -> float:
        """Compute ELBO for ARD prior."""
        return (
            w_node.params.D[w_node.m] * w_node.params.K / 2
            + w_node.params.D[w_node.m] * np.sum(w_node.alpha_m_node.E_log_alpha) / 2
            - np.sum(w_node.alpha_m_node.E_alpha * w_node.E_w_squared) / 2
            + np.sum(log_eps(w_node.hat_w_m_node.vi_var)) / 2
        )

    def get_additional_nodes_to_update(self) -> list:
        """ARD prior has alpha node to update."""
        return ["alpha"]
