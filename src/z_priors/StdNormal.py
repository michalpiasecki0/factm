"""
Standard Normal prior for latent factors Z.
"""
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import ZPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams

from ..utils import log_eps
from .Informed import InformedZPrior


def _compute_z_k_variational_params(
    E_z: np.ndarray,
    indices: np.ndarray,
    k: int,
    w_nodes: list[Any],
    tau_nodes: list[Any],
    y_nodes: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    n_idx = len(indices)
    vi_mu_new = np.zeros(n_idx)
    vi_var_new = np.zeros(n_idx)

    for m in range(len(w_nodes)):
        E_w_k = w_nodes[m].E_w[:, k]
        E_z_k = E_z[indices, k]

        resid = y_nodes[m].data[indices] - np.dot(E_z[indices], w_nodes[m].E_w.T)
        partial_resid = resid + np.outer(E_z_k, E_w_k)

        if not tau_nodes[m].is_ctm:
            vi_var_new += np.ma.dot(
                tau_nodes[m].E_tau[indices],
                w_nodes[m].E_w_squared[:, k],
            )
            vi_mu_new += np.ma.sum(
                tau_nodes[m].E_tau[indices] * E_w_k * partial_resid,
                axis=1,
            )
        else:
            E_quadratic_form_first_term = np.dot(
                np.dot(E_w_k, tau_nodes[m].Sigma0_inv), E_w_k
            )
            E_quadratic_form_second_term = np.sum(
                np.diag(tau_nodes[m].Sigma0_inv)
                * (w_nodes[m].E_w_squared[:, k] - E_w_k**2)
            )
            vi_var_new += E_quadratic_form_first_term + E_quadratic_form_second_term

            first_term = np.dot(E_w_k, tau_nodes[m].Sigma0_inv)
            vi_mu_new += np.sum(first_term * partial_resid, axis=1)

    vi_var_new = 1 / (vi_var_new + 1)
    vi_mu_new = vi_mu_new * vi_var_new
    return vi_mu_new, vi_var_new


class nodeFA_z:
    """
    Class to define Z node (n times k) with standard normal prior.
    """

    def __init__(self, vi_mu, vi_var, params: "FAParams", z_priors=None):
        self.params = params
        self.z_priors = z_priors

        self.vi_mu = vi_mu
        self.vi_var = vi_var

        self.update_params()

        self.elbo = 0

    def MB(self, y_list, w_list, tau_list):
        self.y_node = y_list
        self.w_node = w_list
        self.tau_node = tau_list

    def update(self, indices: np.ndarray | None = None):
        for k in range(self.params.K):
            if self.z_priors is not None and self.z_priors[k] is not None:
                should_update = self.z_priors[k].should_update_for_factor(
                    k, self.z_priors
                )
            elif self.params.Z_priors[k] == "informed":
                should_update = False
            else:
                should_update = True

            if not should_update:
                continue

            if self.z_priors is not None and self.z_priors[k] is not None:
                self.z_priors[k].update_z_k(
                    self, k, self.w_node, self.tau_node, self.y_node, indices
                )
            else:
                idx = indices if indices is not None else np.arange(self.params.N)
                vi_mu_new, vi_var_new = _compute_z_k_variational_params(
                    self.E_z, idx, k, self.w_node, self.tau_node, self.y_node
                )
                self.vi_mu[idx, k] = vi_mu_new
                self.vi_var[idx, k] = vi_var_new
                self.update_params()

            for m in range(len(self.w_node)):
                self.w_node[m].update_params_z()
                if not self.tau_node[m].is_ctm:
                    self.tau_node[m].update_params_w_z()

    def update_informed(self, set_k):
        """Reserved for a separate update path on informed factors."""
        pass

    def update_params(self):
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu**2

    def ELBO(self):
        elbo = 0.0
        for k in range(self.params.K):
            prior_k = None
            if self.z_priors is not None:
                prior_k = self.z_priors[k]

            prior_elbo = None
            if prior_k is not None and hasattr(prior_k, "compute_elbo_k"):
                prior_elbo = prior_k.compute_elbo_k(self, k)

            if prior_elbo is not None:
                elbo += prior_elbo
            else:
                elbo += (
                    self.params.N / 2
                    - np.sum(self.E_z_squared[:, k]) / 2
                    + np.sum(log_eps(self.vi_var[:, k])) / 2
                )
        self.elbo = elbo


class StdNormalZPrior(ZPriorBase):
    """Standard Normal prior for latent factors Z."""

    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list[Any],
        tau_nodes: list[Any],
        y_nodes: list[Any],
        indices: np.ndarray | None = None,
    ) -> None:
        """Update Z node for factor k."""
        if indices is None:
            indices = np.arange(z_node.params.N)

        vi_mu_new, vi_var_new = _compute_z_k_variational_params(
            z_node.E_z, indices, k, w_nodes, tau_nodes, y_nodes
        )
        z_node.vi_mu[indices, k] = vi_mu_new
        z_node.vi_var[indices, k] = vi_var_new
        z_node.update_params()

    def should_update_for_factor(self, k: int, z_priors: list[Any]) -> bool:
        """Standard normal prior updates all non-informed factors."""
        if isinstance(z_priors[k], str):
            return z_priors[k] != "informed"
        return not isinstance(z_priors[k], InformedZPrior)
