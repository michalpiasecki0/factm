"""
Standard Normal prior for latent factors Z.
"""
from typing import TYPE_CHECKING

import numpy as np
from model_config import Likelihood

if TYPE_CHECKING:
    from FACTMpy.FA_model import FAParams

from FACTMpy.utils import log_eps


class nodeFA_z:
    """
    Class to define Z node (n times k) with standard normal prior.
    """

    def __init__(self, vi_mu, vi_var, params: "FAParams"):
        self.params = params

        self.vi_mu = vi_mu
        self.vi_var = vi_var

        self.update_params()

        self.elbo = 0

    def MB(self, y_list, w_list, tau_list):
        self.y_node = y_list
        self.w_node = w_list
        self.tau_node = tau_list

    def update(self):
        # update uninformed (standard normal)
        for k in range(self.params.K):
            if self.params.Z_priors[k] != "informed":
                self.update_k(k)

                for m in range(self.params.M):
                    self.w_node[m].update_params_z()
                    # Only CTM doesn't need tau updates - all FA likelihoods do
                    if self.params.likelihoods[m] != Likelihood.CTM:
                        self.tau_node[m].update_params_w_z()

    def update_informed(self, set_k):
        pass  # TBD

    def update_k(self, k):
        vi_mu_new = np.zeros(self.params.N)
        vi_var_new = np.zeros(self.params.N)

        for m in range(self.params.M):
            # Only distinguish CTM from FA - Normal and Bernoulli are treated the same
            if self.params.likelihoods[m] != Likelihood.CTM:
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

            if self.params.likelihoods[m] in [Likelihood.CTM]:
                # E[w' Sigma0^{-1} w]
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

                # VI var
                vi_var_new += E_quadratic_form_first_term + E_quadratic_form_second_term

                # VI mu
                resid = self.y_node[m].data - np.dot(self.w_node[m].E_w, self.E_z.T).T
                partial_resid = (
                    resid + np.outer(self.w_node[m].E_w[:, k], self.E_z[:, k]).T
                )
                first_term = np.dot(
                    self.w_node[m].E_w[:, k], self.tau_node[m].Sigma0_inv
                )
                vi_mu_new += np.sum(first_term * partial_resid, axis=1)

        # VI var:
        # prior variance of Z equals 1
        vi_var_new = 1 / (vi_var_new + 1)

        # VI mu
        vi_mu_new = vi_mu_new * vi_var_new

        # update mu, var and other internal params
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
