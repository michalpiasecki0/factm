"""
Normal likelihood implementation for Factor Analysis.
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


class NormalLikelihood(LikelihoodBase):
    """Normal likelihood for FA views."""

    def create_y_node(self, data_m, m, params: "FAParams") -> "nodeFA_y_m":
        """Create a y node for Normal likelihood."""
        from FACTMpy.FA_model import nodeFA_y_m

        data_m = np.array(data_m)
        data_m = np.ma.array(data_m, mask=np.isnan(data_m))
        feature_mean_m = np.ma.mean(data_m, axis=0)
        node_y_m = nodeFA_y_m(data_m - feature_mean_m, m, params=params)
        node_y_m.data_mean = feature_mean_m
        return node_y_m

    def create_tau_node(self, a0, b0, m, params: "FAParams") -> "nodeFA_tau_m":
        """Create a tau node for Normal likelihood."""
        from FACTMpy.FA_model import nodeFA_tau_m

        return nodeFA_tau_m(a0, b0, m, params, likelihood=self)

    def update_w_k_for_likelihood(
        self, w_node: "nodeFA_w_m", k: int, z_node: "nodeFA_z", tau_node: "nodeFA_tau_m"
    ):
        """Update W node for factor k for Normal likelihood."""
        # This delegates to the prior's update method, which will call back
        # The prior knows how to update W for Normal likelihood
        w_node.w_prior.update_w_k(w_node, k, z_node, w_node.y_m_node, tau_node)

    def update_z_k_for_likelihood(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_node: "nodeFA_w_m",
        tau_node: "nodeFA_tau_m",
        y_node: "nodeFA_y_m",
    ):
        """Update Z node for factor k for Normal likelihood."""
        # VI var
        vi_var_new = np.ma.dot(tau_node.E_tau, w_node.E_w_squared[:, k])

        # VI mu
        resid = y_node.data - np.dot(w_node.E_w, z_node.E_z.T).T
        partial_resid = resid + np.outer(w_node.E_w[:, k], z_node.E_z[:, k]).T

        vi_mu_new = np.ma.sum(
            tau_node.E_tau * w_node.E_w[:, k] * partial_resid,
            axis=1,
        )

        return vi_mu_new, vi_var_new

    def should_update_tau_after_w(self) -> bool:
        """Normal likelihood requires tau updates after W."""
        return True
