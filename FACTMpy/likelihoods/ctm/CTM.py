"""
CTM likelihood implementation.
"""
from typing import TYPE_CHECKING

import numpy as np
from likelihoods.base import LikelihoodBase

if TYPE_CHECKING:
    from FACTMpy.FA_model import (
        FAParams,
        nodeFA_tau_m,
        nodeFA_w_m,
        nodeFA_y_m,
        nodeFA_z,
    )

# Re-export CTM model classes for convenience
from FACTMpy.CTM_model import (
    CTM,
    CTMParams,
    nodeCTM_beta,
    nodeCTM_eta,
    nodeCTM_mu0,
    nodeCTM_Sigma0,
    nodeCTM_w_z,
    nodeCTM_xi,
    nodeCTM_y,
)

__all__ = [
    "CTM",
    "CTMLikelihood",
    "CTMParams",
    "nodeCTM_beta",
    "nodeCTM_eta",
    "nodeCTM_mu0",
    "nodeCTM_Sigma0",
    "nodeCTM_w_z",
    "nodeCTM_xi",
    "nodeCTM_y",
]


class CTMLikelihood(LikelihoodBase):
    """CTM likelihood - handled separately from FA views."""

    def create_y_node(self, data_m, m, params: "FAParams") -> "nodeFA_y_m":
        """Create a y node for CTM likelihood (placeholder for FA integration)."""
        from FACTMpy.FA_model import nodeFA_y_m

        # CTM views don't use the standard y node in FA
        return nodeFA_y_m(None, m, params=params)

    def create_tau_node(self, a0, b0, m, params: "FAParams") -> "nodeFA_tau_m":
        """Create a tau node for CTM likelihood."""
        from FACTMpy.FA_model import nodeFA_tau_m

        return nodeFA_tau_m(a0, b0, m, params, likelihood=self)

    def update_w_k_for_likelihood(
        self, w_node: "nodeFA_w_m", k: int, z_node: "nodeFA_z", tau_node: "nodeFA_tau_m"
    ):
        """Update W node for factor k for CTM likelihood."""
        # CTM uses different update logic - delegate to prior
        w_node.w_prior.update_w_k(w_node, k, z_node, w_node.y_m_node, tau_node)

    def update_z_k_for_likelihood(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_node: "nodeFA_w_m",
        tau_node: "nodeFA_tau_m",
        y_node: "nodeFA_y_m",
    ):
        """Update Z node for factor k for CTM likelihood."""
        # E[w' Sigma0^{-1} w]
        E_quadratic_form_first_term = np.dot(
            np.dot(w_node.E_w[:, k], tau_node.Sigma0_inv),
            w_node.E_w[:, k],
        )
        E_quadratic_form_second_term = np.sum(
            np.diag(
                np.dot(
                    tau_node.Sigma0_inv,
                    np.diag(w_node.E_w_squared[:, k] - w_node.E_w[:, k] ** 2),
                )
            )
        )

        # VI var
        vi_var_new = E_quadratic_form_first_term + E_quadratic_form_second_term

        # VI mu
        resid = y_node.data - np.dot(w_node.E_w, z_node.E_z.T).T
        partial_resid = resid + np.outer(w_node.E_w[:, k], z_node.E_z[:, k]).T
        first_term = np.dot(w_node.E_w[:, k], tau_node.Sigma0_inv)
        vi_mu_new = np.sum(first_term * partial_resid, axis=1)

        return vi_mu_new, vi_var_new

    def should_update_tau_after_w(self) -> bool:
        """CTM likelihood does not require tau updates after W."""
        return False
