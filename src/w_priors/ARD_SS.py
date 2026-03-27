"""
Automatic Relevance Determination with Spike and Slab (ARD_SS) prior for weights.
"""
# Import ARD classes (ARD_SS uses ARD components)
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import betaln, digamma

from .ARD import nodeFA_alpha_m, nodeFA_hat_w_m
from .base import WPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams, nodeFA_w_m

from ..starting_params import starting_params_hat_w_m, starting_params_s_m
from ..utils import log_eps, xlogx

EPS = 1e-20


class nodeFA_s_m:
    """
    Class to update variational params of S node
    (sparsity: SS - spike and slab) (d times k, for one m)
    """

    def __init__(self, vi_lambda, D: int, params: "FAParams"):
        self.params = params
        self.D = D
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
        lambda_k[lambda_k < np.log(self.D - 1)] = np.log(self.D - 1)

        self.vi_lambda[:, k] = lambda_k
        self.vi_gamma[:, k] = 1 / (1 + np.exp(-lambda_k))


class nodeFA_theta_m:
    """
    Class to define theta node (k, for one m)
    """

    def __init__(self, a0, b0, vi_a, vi_b, D: int, params: "FAParams"):
        self.params = params
        self.D = D

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
        self.vi_b[k] = self.b0 - sum_sdk + self.D

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


class ARD_SSWPrior(WPriorBase):
    """ARD with Spike and Slab prior for weights."""

    def create_nodes(
        self, m: int, params: "FAParams", starting_params: dict, D: int, K: int
    ) -> dict:
        """Create nodes for ARD_SS prior."""

        w_mu, w_var = starting_params_hat_w_m(starting_params, m, D, K)
        node_hat_w_m = nodeFA_hat_w_m(w_mu, w_var, D, params=params)
        node_alpha_m = nodeFA_alpha_m(1e-14, 1e-14, D, params=params)

        s_lambda = starting_params_s_m(starting_params, m, D, K)
        node_s = nodeFA_s_m(s_lambda, D, params=params)
        node_theta_m = nodeFA_theta_m(
            1, 1, 99 * np.ones(K), np.ones(K), D, params=params
        )

        return {
            "w_m_node_not_sparse": None,
            "hat_w_m_node": node_hat_w_m,
            "s_m_node": node_s,
            "alpha_m_node": node_alpha_m,
            "theta_m_node": node_theta_m,
        }

    def update_w_k(
        self,
        w_node: "nodeFA_w_m",
        k: int,
        z_node: Any,
        y_node: Any,
        tau_node: Any,
        indices: Any = None,
        data_scale: float = 1.0,
    ):
        """Update W node for factor k for ARD_SS prior."""
        if not w_node.is_ctm:
            # For FA likelihoods (Normal/Bernoulli)
            if indices is None:
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
            else:
                # Minibatch-only computation (Algorithm 3 global update).
                z_sub = z_node.E_z[indices]  # (S,K)
                z_k = z_sub[:, k]  # (S,)
                z_sq_k = z_node.E_z_squared[indices, k]  # (S,)
                tau_sub = tau_node.E_tau[indices]  # (S,D)
                y_sub = y_node.data[indices]  # (S,D)

                nominator_second_term_tmp = np.dot(w_node.E_w, z_sub.T * z_k).T  # (S,D)
                nominator_second_term = nominator_second_term_tmp - np.outer(
                    z_k**2, w_node.E_w[:, k]
                )  # (S,D)

                nominator_data = np.ma.sum(
                    tau_sub * ((y_sub.T * z_k).T - nominator_second_term),
                    axis=0,
                )
                nominator = data_scale * nominator_data

                denominator = (
                    data_scale * np.ma.dot(z_sq_k, tau_sub)
                    + w_node.alpha_m_node.E_alpha[k]
                )

            w_node.hat_w_m_node.update_k(k, nominator, denominator)

            # Update s node for ARD_SS
            E_log_LR_theta = w_node.theta_m_node.E_log_LR[k].copy()
            w_node.s_m_node.update_k(
                k,
                nominator,
                denominator,
                w_node.alpha_m_node.E_alpha[k],
                E_log_LR_theta,
            )

            w_node.update_params()
            w_node.update_params_z()
            w_node.alpha_m_node.update_k(k)
            w_node.theta_m_node.update_k(k)
        else:
            # For CTM likelihood - ARD_SS not typically used with CTM
            # But implement for completeness
            raise NotImplementedError("ARD_SS with CTM not yet implemented")

    def update_params(self, w_node: "nodeFA_w_m"):
        """Update E_w and E_w_squared for ARD_SS prior."""
        w_node.E_w = w_node.s_m_node.vi_gamma * w_node.hat_w_m_node.vi_mu
        w_node.E_w_squared = w_node.s_m_node.vi_gamma * (
            w_node.hat_w_m_node.vi_var + w_node.hat_w_m_node.vi_mu**2
        )

        w_node.E_hat_w_squared = w_node.s_m_node.vi_gamma * (
            w_node.hat_w_m_node.vi_var + w_node.hat_w_m_node.vi_mu**2
        ) + (1 - w_node.s_m_node.vi_gamma) * (
            np.outer(
                np.ones(w_node.D),
                w_node.alpha_m_node.E_inv_alpha,
            )
        )

    def compute_elbo(self, w_node: "nodeFA_w_m") -> float:
        """Compute ELBO for ARD_SS prior."""
        elbo_s = (
            np.sum(np.dot(w_node.theta_m_node.E_log_theta, w_node.s_m_node.vi_gamma.T))
            + np.sum(
                np.dot(
                    w_node.theta_m_node.E_log_1minustheta,
                    (1 - w_node.s_m_node.vi_gamma).T,
                )
            )
            - np.sum(xlogx(w_node.s_m_node.vi_gamma))
            - np.sum(xlogx(1 - w_node.s_m_node.vi_gamma))
        )
        elbo_w_hat = (
            w_node.D * np.sum(w_node.alpha_m_node.E_log_alpha) / 2
            - (
                np.sum(
                    np.dot(
                        (1 - w_node.s_m_node.vi_gamma),
                        w_node.alpha_m_node.E_log_alpha,
                    )
                )
                / 2
            )
            - np.sum(np.dot(w_node.E_hat_w_squared, w_node.alpha_m_node.E_alpha)) / 2
            + np.sum(w_node.s_m_node.vi_gamma * log_eps(w_node.hat_w_m_node.vi_var)) / 2
        )
        return elbo_s + elbo_w_hat

    def get_additional_nodes_to_update(self) -> list:
        """ARD_SS prior has alpha and theta nodes to update."""
        return ["alpha", "theta"]
