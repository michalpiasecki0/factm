"""
Cohort-aware latent-factor prior for FACTM.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import digamma, gammaln

from ..utils import log_eps
from .base import ZPriorBase

if TYPE_CHECKING:
    from ..FA_model import nodeFA_z

EPS = 1e-20


class Cohort:
    """Helper carrying cohort labels, per-sample cohort ids and cohort counts."""

    def __init__(self, labels: np.ndarray):
        labels_arr = np.asarray(labels)
        if labels_arr.ndim != 1:
            raise ValueError("Cohort labels must be a 1D array-like.")
        if labels_arr.size == 0:
            raise ValueError("Cohort labels cannot be empty.")

        unique_labels, cohort_indices = np.unique(labels_arr, return_inverse=True)
        self.labels = unique_labels
        self.cohort_indices = cohort_indices.astype(int)
        self.C = int(len(unique_labels))
        self.N = int(labels_arr.shape[0])
        self.counts = np.bincount(self.cohort_indices, minlength=self.C).astype(float)


class CohortZPrior(ZPriorBase):
    """Cohort-aware prior for one latent factor."""

    def __init__(
        self,
        cohorts: np.ndarray,
        pi: float,
        a_lambda: float,
        b_lambda: float,
        a0_tau: float,
        b0_tau: float,
    ):
        self.cohort = Cohort(cohorts)

        self.pi = float(np.clip(pi, EPS, 1.0 - EPS))
        self.logit_pi = np.log(self.pi) - np.log(1.0 - self.pi)

        self.a_lambda = float(a_lambda)
        self.b_lambda = float(b_lambda)
        self.a0_tau = float(a0_tau)
        self.b0_tau = float(b0_tau)

        self.delta_mu = np.zeros(self.cohort.C)
        self.delta_var = np.ones(self.cohort.C)
        self.gamma_prob = np.full(self.cohort.C, self.pi)

        self.tau_a = self.a0_tau + self.cohort.counts / 2.0
        self.tau_b = np.full(self.cohort.C, self.b0_tau + 1.0)

        self.lambda_a = self.a_lambda + self.cohort.C / 2.0
        self.lambda_b = self.b_lambda + 0.5 * np.sum(self.delta_var)

        self._update_expectations()

    def _update_expectations(self) -> None:
        self.E_delta = self.delta_mu
        self.E_delta_squared = self.delta_var + self.delta_mu**2

        self.E_gamma = np.clip(self.gamma_prob, EPS, 1.0 - EPS)

        self.E_tau = self.tau_a / np.maximum(self.tau_b, EPS)
        self.E_log_tau = digamma(self.tau_a) - log_eps(self.tau_b)

        self.E_lambda = self.lambda_a / max(self.lambda_b, EPS)
        self.E_log_lambda = digamma(self.lambda_a) - np.log(max(self.lambda_b, EPS))

    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list,
        tau_nodes: list,
        y_nodes: list,
        indices: np.ndarray | None = None,
    ):
        if indices is None:
            indices = np.arange(z_node.params.N)

        n_idx = len(indices)
        vi_mu_num = np.zeros(n_idx)

        cohort_idx = self.cohort.cohort_indices[indices]
        prior_precision = self.E_tau[cohort_idx]
        prior_mean_term = prior_precision * self.E_gamma[cohort_idx] * self.E_delta[
            cohort_idx
        ]
        vi_mu_num += prior_mean_term
        vi_var_inv = prior_precision.copy()

        for m in range(len(w_nodes)):
            E_w_k = w_nodes[m].E_w[:, k]
            E_z_k = z_node.E_z[indices, k]

            resid = y_nodes[m].data[indices] - np.dot(
                z_node.E_z[indices], w_nodes[m].E_w.T
            )
            partial_resid = resid + np.outer(E_z_k, E_w_k)

            if not tau_nodes[m].is_ctm:
                vi_var_inv += np.ma.dot(
                    tau_nodes[m].E_tau[indices], w_nodes[m].E_w_squared[:, k]
                )
                vi_mu_num += np.ma.sum(
                    tau_nodes[m].E_tau[indices] * E_w_k * partial_resid, axis=1
                )
            else:
                E_quadratic_form_first_term = np.dot(
                    np.dot(E_w_k, tau_nodes[m].Sigma0_inv), E_w_k
                )
                E_quadratic_form_second_term = np.sum(
                    np.diag(tau_nodes[m].Sigma0_inv)
                    * (w_nodes[m].E_w_squared[:, k] - E_w_k**2)
                )
                vi_var_inv += (
                    E_quadratic_form_first_term + E_quadratic_form_second_term
                )

                first_term = np.dot(E_w_k, tau_nodes[m].Sigma0_inv)
                vi_mu_num += np.sum(first_term * partial_resid, axis=1)

        vi_var_new = 1.0 / np.maximum(vi_var_inv, EPS)
        vi_mu_new = vi_mu_num * vi_var_new

        z_node.vi_mu[indices, k] = vi_mu_new
        z_node.vi_var[indices, k] = vi_var_new
        z_node.update_params()

        self._update_cohort_params(z_node, k)

    def _update_cohort_params(self, z_node: "nodeFA_z", k: int) -> None:
        cohort_idx = self.cohort.cohort_indices
        counts = self.cohort.counts

        sum_z = np.bincount(
            cohort_idx,
            weights=np.asarray(z_node.E_z[:, k], dtype=float),
            minlength=self.cohort.C,
        )
        sum_z2 = np.bincount(
            cohort_idx,
            weights=np.asarray(z_node.E_z_squared[:, k], dtype=float),
            minlength=self.cohort.C,
        )

        delta_precision = self.E_lambda + self.E_tau * (self.E_gamma) * counts
        self.delta_var = 1.0 / np.maximum(delta_precision, EPS)
        self.delta_mu = self.delta_var * self.E_tau * (self.E_gamma) * sum_z
        self._update_expectations()

        gamma_u = self.logit_pi - 0.5 * self.E_tau * (
            counts * self.E_delta_squared - 2.0 * self.E_delta * sum_z
        )
        gamma_u = np.clip(gamma_u, -60.0, 60.0)
        self.gamma_prob = np.clip(1.0 / (1.0 + np.exp(-gamma_u)), EPS, 1.0 - EPS)
        self._update_expectations()

        self.tau_a = self.a0_tau + counts / 2.0
        tau_b = self.b0_tau + 0.5 * (
            sum_z2
            - 2.0 * self.E_gamma * self.E_delta * sum_z
            + counts * self.E_gamma * self.E_delta_squared
        )
        self.tau_b = np.maximum(tau_b, EPS)
        self._update_expectations()

        self.lambda_a = self.a_lambda + self.cohort.C / 2.0
        self.lambda_b = self.b_lambda + 0.5 * float(np.sum(self.E_delta_squared))
        self.lambda_b = max(self.lambda_b, EPS)
        self._update_expectations()

    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        return True

    def compute_elbo_k(self, z_node: "nodeFA_z", k: int) -> float:
        cohort_idx = self.cohort.cohort_indices
        counts = self.cohort.counts

        sum_z = np.bincount(
            cohort_idx,
            weights=np.asarray(z_node.E_z[:, k], dtype=float),
            minlength=self.cohort.C,
        )
        sum_z2 = np.bincount(
            cohort_idx,
            weights=np.asarray(z_node.E_z_squared[:, k], dtype=float),
            minlength=self.cohort.C,
        )
        sum_log_var = np.bincount(
            cohort_idx,
            weights=np.asarray(log_eps(z_node.vi_var[:, k]), dtype=float),
            minlength=self.cohort.C,
        )

        log_2pi = np.log(2.0 * np.pi)

        resid_sum = (
            sum_z2
            - 2.0 * self.E_gamma * self.E_delta * sum_z
            + counts * self.E_gamma * self.E_delta_squared
        )
        log_p_z = 0.5 * np.sum(
            counts * (self.E_log_tau - log_2pi) - self.E_tau * resid_sum
        )
        entropy_z = 0.5 * np.sum(counts * (1.0 + log_2pi) + sum_log_var)

        log_p_delta = 0.5 * np.sum(
            self.E_log_lambda - log_2pi - self.E_lambda * self.E_delta_squared
        )
        entropy_delta = 0.5 * np.sum(1.0 + log_2pi + log_eps(self.delta_var))

        log_p_gamma = np.sum(
            self.E_gamma * np.log(self.pi)
            + (1.0 - self.E_gamma) * np.log(1.0 - self.pi)
        )
        entropy_gamma = -np.sum(
            self.E_gamma * log_eps(self.E_gamma)
            + (1.0 - self.E_gamma) * log_eps(1.0 - self.E_gamma)
        )

        log_p_tau = np.sum(
            (self.a0_tau - 1.0) * self.E_log_tau
            - self.b0_tau * self.E_tau
            + self.a0_tau * np.log(self.b0_tau)
            - gammaln(self.a0_tau)
        )
        entropy_tau = np.sum(
            self.tau_a
            - log_eps(self.tau_b)
            + gammaln(self.tau_a)
            + (1.0 - self.tau_a) * digamma(self.tau_a)
        )

        log_p_lambda = (
            (self.a_lambda - 1.0) * self.E_log_lambda
            - self.b_lambda * self.E_lambda
            + self.a_lambda * np.log(self.b_lambda)
            - gammaln(self.a_lambda)
        )
        entropy_lambda = (
            self.lambda_a
            - np.log(self.lambda_b)
            + gammaln(self.lambda_a)
            + (1.0 - self.lambda_a) * digamma(self.lambda_a)
        )

        return float(
            log_p_z
            + entropy_z
            + log_p_delta
            + entropy_delta
            + log_p_gamma
            + entropy_gamma
            + log_p_tau
            + entropy_tau
            + log_p_lambda
            + entropy_lambda
        )
