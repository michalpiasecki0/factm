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

        # Incorporate cohort-specific normal prior if provided
        if hasattr(self.params, "cohort_labels") and self.params.cohort_labels is not None:
            cohort_idx = self.params.cohort_labels[indices]
            prior_mu = self.params.cohort_mu[cohort_idx, k]
            prior_sigma2 = self.params.cohort_sigma2[cohort_idx, k]
            prior_precision = 1.0 / prior_sigma2

            vi_var_new = 1.0 / (vi_var_new + prior_precision)
            vi_mu_new = (vi_mu_new + prior_precision * prior_mu) * vi_var_new
        else:
            # standard normal prior (mu=0, sigma2=1)
            vi_var_new = 1.0 / (vi_var_new + 1)
            vi_mu_new = vi_mu_new * vi_var_new

        self.vi_mu[indices, k] = vi_mu_new
        self.vi_var[indices, k] = vi_var_new
        self.update_params()

    def update_params(self):
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu**2

    def ELBO(self):
        # Compute entropy H[q(Z)]
        # H = 0.5 * sum(log(vi_var)) + 0.5 * N*K * (1 + log(2*pi))
        H = 0.5 * np.sum(np.log(self.vi_var)) + 0.5 * self.params.N * self.params.K * (
            1.0 + np.log(2.0 * np.pi)
        )

        # Compute expected log-prior E_q[log p(Z)]
        params = self.params
        if getattr(params, "cohort_labels", None) is not None:
            labels = params.cohort_labels
            C = int(labels.max() + 1)
            E_log_p = 0.0
            for c in range(C):
                idx = np.where(labels == c)[0]
                if idx.size == 0:
                    continue
                mu_c = params.cohort_mu[c]  # (K,)
                sigma2_c = params.cohort_sigma2[c]  # (K,)
                Ez = self.E_z[idx]  # (n_c, K)
                Ez2 = self.E_z_squared[idx]
                # sum over n and k
                term1 = -0.5 * np.sum((Ez2 - 2.0 * Ez * mu_c + mu_c ** 2) / sigma2_c)
                term2 = -0.5 * idx.size * np.sum(np.log(2.0 * np.pi * sigma2_c))
                E_log_p += term1 + term2
        else:
            # Standard normal prior: p(z)=N(0,1)
            E_log_p = -0.5 * np.sum(self.E_z_squared) - 0.5 * self.params.N * self.params.K * np.log(
                2.0 * np.pi
            )

        # ELBO contribution from Z node
        elbo = H + E_log_p

        self.elbo = elbo


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

        # Incorporate cohort-specific normal prior if provided
        params = z_node.params
        if getattr(params, "cohort_labels", None) is not None:
            cohort_idx = params.cohort_labels[indices]
            prior_mu = params.cohort_mu[cohort_idx, k]
            prior_sigma2 = params.cohort_sigma2[cohort_idx, k]
            prior_precision = 1.0 / prior_sigma2

            vi_var_new = 1.0 / (vi_var_new + prior_precision)
            vi_mu_new = (vi_mu_new + prior_precision * prior_mu) * vi_var_new
        else:
            # standard normal prior (mu=0, sigma2=1)
            vi_var_new = 1.0 / (vi_var_new + 1)
            vi_mu_new = vi_mu_new * vi_var_new

        z_node.vi_mu[indices, k] = vi_mu_new
        z_node.vi_var[indices, k] = vi_var_new
        z_node.update_params()

    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Standard normal prior updates all non-informed factors."""
        if isinstance(z_priors[k], str):
            return z_priors[k] != "informed"
        return not isinstance(z_priors[k], InformedZPrior)
