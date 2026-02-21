"""
No sparsity prior for weights (W_prior = None).
"""
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import WPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams, nodeFA_w_m

from ..model_config import Likelihood
from ..utils import log_eps


class nodeFA_w_m_not_sparse:
    """
    Class to update variational params of W (no sparsity) (d times k, for one m)
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


class NoneWPrior(WPriorBase):
    """No sparsity prior for weights."""

    def create_nodes(
        self, m: int, params: "FAParams", starting_params: dict, D: int, K: int
    ) -> dict:
        """Create nodes for None prior."""
        from ..starting_params import starting_params_hat_w_m

        key_tmp = "M" + str(m)
        w_mu, w_var = starting_params_hat_w_m(starting_params, key_tmp, D, K)
        node_w_m_not_sparse = nodeFA_w_m_not_sparse(w_mu, w_var, m, params=params)

        return {
            "w_m_node_not_sparse": node_w_m_not_sparse,
            "hat_w_m_node": None,
            "s_m_node": None,
            "alpha_m_node": None,
            "theta_m_node": None,
        }

    def update_w_k(
        self, w_node: "nodeFA_w_m", k: int, z_node: Any, y_node: Any, tau_node: Any
    ):
        """Update W node for factor k for None prior."""
        # Check if this is a CTM likelihood or FA likelihood
        is_ctm = w_node.params.likelihoods[w_node.m] == Likelihood.CTM

        if not is_ctm:
            # For Normal/Bernoulli likelihoods
            # sum_j!=k <z_nk><z_nj><w_jd>
            nominator_second_term = np.dot(w_node.E_w, z_node.E_z.T) - np.outer(
                w_node.E_w[:, k], z_node.E_z[:, k]
            )
            nominator_resid = y_node.data - nominator_second_term.T
            nominator = np.ma.dot(z_node.E_z[:, k], tau_node.E_tau * nominator_resid)

            denominator = np.ma.dot(z_node.E_z_squared[:, k], tau_node.E_tau) + 1

            w_node.w_m_node_not_sparse.update_k(k, nominator, denominator)
            w_node.update_params()
            w_node.update_params_z()
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
                np.eye(w_node.params.D[w_node.m])
                + np.sum(z_node.E_z_squared[:, k]) * (tau_node.Sigma0_inv)
            )
            w_node.w_m_node_not_sparse.update_k(k, nominator, denominator)
            w_node.update_params()
            w_node.update_params_z()

    def update_params(self, w_node: "nodeFA_w_m"):
        """Update E_w and E_w_squared for None prior."""
        w_node.E_w = w_node.w_m_node_not_sparse.vi_mu
        w_node.E_w_squared = (
            w_node.w_m_node_not_sparse.vi_var + w_node.w_m_node_not_sparse.vi_mu**2
        )

    def compute_elbo(self, w_node: "nodeFA_w_m") -> float:
        """Compute ELBO for None prior."""
        return (
            w_node.params.D[w_node.m] * w_node.params.K / 2
            - np.sum(w_node.E_w_squared) / 2
            + np.sum(log_eps(w_node.w_m_node_not_sparse.vi_var)) / 2
        )

    def get_additional_nodes_to_update(self) -> list:
        """None prior has no additional nodes."""
        return []
