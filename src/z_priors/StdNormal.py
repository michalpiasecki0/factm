"""
Standard Normal prior for latent factors Z.
"""
from typing import TYPE_CHECKING

import numpy as np

from .base import ZPriorBase

if TYPE_CHECKING:
    from ..FA_model import FAParams

from ..utils import log_eps
from .Informed import InformedZPrior


class nodeFA_z:
    """
    Class to define Z node (n times k) with standard normal prior.
    """

    def __init__(self, vi_mu, vi_var, params: "FAParams", z_priors=None):
        self.params = params
        self.z_priors = z_priors  # List of ZPriorBase objects (one per factor)

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
            should_update = True
            if self.z_priors is not None and self.z_priors[k] is not None:
                should_update = self.z_priors[k].should_update_for_factor(
                    k, self.z_priors
                )
            else:
                # Fallback for backward compatibility
                if self.params.Z_priors[k] == "informed":
                    should_update = False

            if should_update:
                if self.z_priors is not None and self.z_priors[k] is not None:
                    self.z_priors[k].update_z_k(
                        self, k, self.w_node, self.tau_node, self.y_node, indices
                    )
                else:
                    self.update_k(k, indices=indices)

                for m in range(len(self.w_node)):
                    self.w_node[m].update_params_z()
                    if not self.tau_node[m].is_ctm:
                        self.tau_node[m].update_params_w_z()

    def update_informed(self, set_k):
        pass  # TBD

    def update_k(self, k: int, indices: np.ndarray | None = None):
        if indices is None:
            indices = np.arange(self.params.N)

        n_idx = len(indices)
        vi_mu_new = np.zeros(n_idx)
        vi_var_new = np.zeros(n_idx)

        for m in range(len(self.w_node)):
            E_w_k = self.w_node[m].E_w[:, k]
            E_z_k = self.E_z[indices, k]

            resid = self.y_node[m].data[indices] - np.dot(
                self.E_z[indices], self.w_node[m].E_w.T
            )
            partial_resid = resid + np.outer(E_z_k, E_w_k)
            print(self.tau_node[m].E_tau.shape)
            if not self.tau_node[m].is_ctm:
                vi_var_new += np.ma.dot(
                    self.tau_node[m].E_tau[indices], self.w_node[m].E_w_squared[:, k]
                )
                vi_mu_new += np.ma.sum(
                    self.tau_node[m].E_tau[indices] * E_w_k * partial_resid, axis=1
                )
            else:
                E_quadratic_form_first_term = np.dot(
                    np.dot(E_w_k, self.tau_node[m].Sigma0_inv), E_w_k
                )
                E_quadratic_form_second_term = np.sum(
                    np.diag(self.tau_node[m].Sigma0_inv)
                    * (self.w_node[m].E_w_squared[:, k] - E_w_k**2)
                )
                vi_var_new += E_quadratic_form_first_term + E_quadratic_form_second_term

                first_term = np.dot(E_w_k, self.tau_node[m].Sigma0_inv)
                vi_mu_new += np.sum(first_term * partial_resid, axis=1)

        vi_var_new = 1 / (vi_var_new + 1)
        vi_mu_new = vi_mu_new * vi_var_new

        self.vi_mu[indices, k] = vi_mu_new
        self.vi_var[indices, k] = vi_var_new
        self.update_params()

    def update_params(self):
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu**2

    def ELBO(self):
        self.elbo = (
            self.params.N * self.params.K / 2
            - np.sum(self.E_z_squared) / 2
            + np.sum(log_eps(self.vi_var)) / 2
        )


class StdNormalZPrior(ZPriorBase):
    """Standard Normal prior for latent factors Z."""

    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list,
        tau_nodes: list,
        y_nodes: list,
        indices: np.ndarray | None = None,
    ):
        """Update Z node for factor k."""
        if indices is None:
            indices = np.arange(z_node.params.N)

        n_idx = len(indices)
        vi_mu_new = np.zeros(n_idx)
        vi_var_new = np.zeros(n_idx)

        for m in range(len(w_nodes)):
            E_w_k = w_nodes[m].E_w[:, k]  # (M,)
            E_z_k = z_node.E_z[indices, k]  # (n_idx,)

            # partial_resid only for indexed rows: (n_idx, M)
            resid = y_nodes[m].data[indices] - np.dot(
                z_node.E_z[indices], w_nodes[m].E_w.T
            )
            partial_resid = resid + np.outer(E_z_k, E_w_k)

            if not tau_nodes[m].is_ctm:
                vi_var_new += np.ma.dot(
                    tau_nodes[m].E_tau[indices],
                    w_nodes[m].E_w_squared[:, k],  # (n_idx, M) @ (M,) -> (n_idx,)
                )
                vi_mu_new += np.ma.sum(
                    tau_nodes[m].E_tau[indices] * E_w_k * partial_resid,
                    axis=1,  # all (n_idx, M)
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

                first_term = np.dot(E_w_k, tau_nodes[m].Sigma0_inv)  # (M,)
                vi_mu_new += np.sum(first_term * partial_resid, axis=1)

        vi_var_new = 1 / (vi_var_new + 1)
        vi_mu_new = vi_mu_new * vi_var_new

        z_node.vi_mu[indices, k] = vi_mu_new
        z_node.vi_var[indices, k] = vi_var_new
        z_node.update_params()

    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Standard normal prior updates all non-informed factors."""
        if isinstance(z_priors[k], str):
            return z_priors[k] != "informed"
        return not isinstance(z_priors[k], InformedZPrior)
