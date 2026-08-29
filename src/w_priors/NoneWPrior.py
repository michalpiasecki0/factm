"""
No sparsity prior for weights (W_prior = None).
"""
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import WPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams, nodeFA_w_m

from ..starting_params import starting_params_hat_w_m
from ..utils import log_eps


def _none_fa_w_k_nominator_denominator(
    w_node: "nodeFA_w_m",
    z_node: Any,
    y_node: Any,
    tau_node: Any,
    k: int,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if indices is None:
        indices = np.arange(z_node.params.N)

    z_k = z_node.E_z[indices, k]
    z2_k = z_node.E_z_squared[indices, k]

    nominator_second_term = np.dot(w_node.E_w, z_node.E_z[indices].T) - np.outer(
        w_node.E_w[:, k], z_k
    )
    nominator_resid = y_node.data[indices] - nominator_second_term.T
    nominator = np.ma.dot(z_k, tau_node.E_tau[indices] * nominator_resid)
    denominator = np.ma.dot(z2_k, tau_node.E_tau[indices]) + 1.0
    return nominator, denominator


def _none_ctm_w_k_nominator_denominator(
    w_node: "nodeFA_w_m",
    z_node: Any,
    y_node: Any,
    tau_node: Any,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    E_zE_zk = np.dot(z_node.E_z.T, z_node.E_z[:, k])
    E_zE_zk[k] = 0
    EwE_zE_zk = np.dot(w_node.E_w, E_zE_zk)
    term1 = np.dot(tau_node.Sigma0_inv, EwE_zE_zk)

    Ez_squaredk = np.sum(z_node.E_z_squared[:, k])
    Sigma_inv_nodiag = tau_node.Sigma0_inv - np.diag(np.diag(tau_node.Sigma0_inv))
    term2 = np.dot(Sigma_inv_nodiag, w_node.E_w[:, k]) * Ez_squaredk

    term3 = np.dot(
        z_node.E_z[:, k],
        np.ma.dot(y_node.data, tau_node.Sigma0_inv),
    )

    nominator = term3 - term1 - term2
    denominator = np.diag(
        np.eye(w_node.D) + np.sum(z_node.E_z_squared[:, k]) * tau_node.Sigma0_inv
    )
    return nominator, denominator


class nodeFA_w_m_not_sparse:
    """
    Class to update variational params of W (no sparsity) (d times k, for one m)
    """

    def __init__(self, vi_mu, vi_var, D: int, params: "FAParams"):
        self.params = params
        self.D = D
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

        w_mu, w_var = starting_params_hat_w_m(starting_params, m, D, K)
        node_w_m_not_sparse = nodeFA_w_m_not_sparse(w_mu, w_var, D, params=params)

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
        if w_node.is_ctm:
            nominator, denominator = _none_ctm_w_k_nominator_denominator(
                w_node, z_node, y_node, tau_node, k
            )
        else:
            nominator, denominator = _none_fa_w_k_nominator_denominator(
                w_node, z_node, y_node, tau_node, k
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
            w_node.D * w_node.params.K / 2
            - np.sum(w_node.E_w_squared) / 2
            + np.sum(log_eps(w_node.w_m_node_not_sparse.vi_var)) / 2
        )

    def get_additional_nodes_to_update(self) -> list[str]:
        """None prior has no additional nodes."""
        return []

    def svi_target_w_k(
        self,
        w_node: "nodeFA_w_m",
        k: int,
        z_node: Any,
        y_node: Any,
        tau_node: Any,
        indices: np.ndarray | None,
    ) -> dict:
        if indices is None:
            indices = np.arange(z_node.params.N)
        if w_node.is_ctm:
            raise NotImplementedError("SVI globals for CTM-injected W not supported.")

        nominator, denominator = _none_fa_w_k_nominator_denominator(
            w_node, z_node, y_node, tau_node, k, indices
        )
        return {"mu": nominator / denominator, "var": 1.0 / denominator}
