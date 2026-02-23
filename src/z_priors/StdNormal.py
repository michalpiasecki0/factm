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

    def update(self):
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
                        self, k, self.w_node, self.tau_node, self.y_node
                    )
                else:
                    self.update_k(k)

                for m in range(len(self.w_node)):
                    self.w_node[m].update_params_z()
                    if not self.tau_node[m].is_ctm:
                        self.tau_node[m].update_params_w_z()

    def update_informed(self, set_k):
        pass  # TBD

    def update_k(self, k):
        vi_mu_new = np.zeros(self.params.N)
        vi_var_new = np.zeros(self.params.N)

        for m in range(len(self.w_node)):
            if not self.tau_node[m].is_ctm:
                # VI var
                vi_var_new += np.ma.dot(
                    self.tau_node[m].E_tau, self.w_node[m].E_w_squared[:, k]
                )

                # VI mu
                resid = self.y_node[m].data - np.dot(self.w_node[m].E_w, self.E_z.T).T
                partial_resid = (
                    resid + np.outer(self.w_node[m].E_w[:, k], self.E_z[:, k]).T
                )
                vi_mu_new += np.ma.sum(
                    self.tau_node[m].E_tau * self.w_node[m].E_w[:, k] * partial_resid,
                    axis=1,
                )

            else:
                # CTM view
                E_quadratic_form_first_term = np.dot(
                    np.dot(self.w_node[m].E_w[:, k], self.tau_node[m].Sigma0_inv),
                    self.w_node[m].E_w[:, k],
                )
                E_quadratic_form_second_term = np.sum(
                    np.diag(
                        np.dot(
                            self.tau_node[m].Sigma0_inv,
                            np.diag(
                                self.w_node[m].E_w_squared[:, k]
                                - self.w_node[m].E_w[:, k] ** 2
                            ),
                        )
                    )
                )
                vi_var_new += E_quadratic_form_first_term + E_quadratic_form_second_term

                resid = self.y_node[m].data - np.dot(self.w_node[m].E_w, self.E_z.T).T
                partial_resid = (
                    resid + np.outer(self.w_node[m].E_w[:, k], self.E_z[:, k]).T
                )
                first_term = np.dot(
                    self.w_node[m].E_w[:, k], self.tau_node[m].Sigma0_inv
                )
                vi_mu_new += np.sum(first_term * partial_resid, axis=1)

        vi_var_new = 1 / (vi_var_new + 1)
        vi_mu_new = vi_mu_new * vi_var_new

        self.vi_mu[:, k] = vi_mu_new
        self.vi_var[:, k] = vi_var_new
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
        self, z_node: "nodeFA_z", k: int, w_nodes: list, tau_nodes: list, y_nodes: list
    ):
        """Update Z node for factor k."""
        vi_mu_new = np.zeros(z_node.params.N)
        vi_var_new = np.zeros(z_node.params.N)

        for m in range(len(w_nodes)):
            if not tau_nodes[m].is_ctm:
                # VI var
                vi_var_new += np.ma.dot(
                    tau_nodes[m].E_tau, w_nodes[m].E_w_squared[:, k]
                )

                # VI mu
                resid = y_nodes[m].data - np.dot(w_nodes[m].E_w, z_node.E_z.T).T
                partial_resid = (
                    resid + np.outer(w_nodes[m].E_w[:, k], z_node.E_z[:, k]).T
                )
                vi_mu_new += np.ma.sum(
                    tau_nodes[m].E_tau * w_nodes[m].E_w[:, k] * partial_resid,
                    axis=1,
                )

            else:
                # CTM view
                E_quadratic_form_first_term = np.dot(
                    np.dot(w_nodes[m].E_w[:, k], tau_nodes[m].Sigma0_inv),
                    w_nodes[m].E_w[:, k],
                )
                E_quadratic_form_second_term = np.sum(
                    np.diag(
                        np.dot(
                            tau_nodes[m].Sigma0_inv,
                            np.diag(
                                w_nodes[m].E_w_squared[:, k] - w_nodes[m].E_w[:, k] ** 2
                            ),
                        )
                    )
                )
                vi_var_new += E_quadratic_form_first_term + E_quadratic_form_second_term

                resid = y_nodes[m].data - np.dot(w_nodes[m].E_w, z_node.E_z.T).T
                partial_resid = (
                    resid + np.outer(w_nodes[m].E_w[:, k], z_node.E_z[:, k]).T
                )
                first_term = np.dot(w_nodes[m].E_w[:, k], tau_nodes[m].Sigma0_inv)
                vi_mu_new += np.sum(first_term * partial_resid, axis=1)

        vi_var_new = 1 / (vi_var_new + 1)
        vi_mu_new = vi_mu_new * vi_var_new

        z_node.vi_mu[:, k] = vi_mu_new
        z_node.vi_var[:, k] = vi_var_new
        z_node.update_params()

    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Standard normal prior updates all non-informed factors."""
        if isinstance(z_priors[k], str):
            return z_priors[k] != "informed"
        return not isinstance(z_priors[k], InformedZPrior)
