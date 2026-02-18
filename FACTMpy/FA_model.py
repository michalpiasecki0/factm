"""
This module provides Factor Analysis model with all the nodes.
"""
import importlib
from dataclasses import dataclass
from typing import List

import numpy as np
from model_config import Likelihood, WPrior
from scipy.special import digamma, gammaln
from utils import log_eps, xlogx
from w_priors.ARD import nodeFA_alpha_m, nodeFA_hat_w_m
from w_priors.ARD_SS import nodeFA_s_m, nodeFA_theta_m
from z_priors.StdNormal import nodeFA_z

# Import None prior using importlib since "None" is a Python keyword
_none_prior = importlib.import_module("w_priors.None")
nodeFA_w_m_not_sparse = _none_prior.nodeFA_w_m_not_sparse

EPS = 1e-20


@dataclass
class FAParams:
    """Shared parameters for Factor Analysis model."""

    N: int  # number of observations (samples)
    K: int  # number of hidden factors
    D: List[int]  # number of features in each view
    M: int  # number of views
    likelihoods: List[str]
    Z_priors: List[str]
    W_priors: List[str]


# nodeFA_z is now imported from z_priors.StdNormal
# nodeFA_w_m_not_sparse is now imported from w_priors.None


# nodeFA_w_m_not_sparse is now imported from w_priors.None
# nodeFA_hat_w_m is now imported from w_priors.ARD
# nodeFA_s_m is now imported from w_priors.ARD_SS


class nodeFA_w_m:
    """
    Class to define W node (d times k, for one m)
    Gathers together the nodes specifying sparsity structure
    e.g. W_hat and S or W_tilde and P (TBD)
    """

    def __init__(self, m, params: FAParams):
        self.params = params

        self.E_w = np.zeros((self.params.D[m], self.params.K))
        self.E_w_squared = np.ones((self.params.D[m], self.params.K))

        self.m = m

        self.E_w_z = np.zeros((self.params.N, self.params.D[m]))
        self.E_w_z_squared = np.zeros((self.params.N, self.params.D[m]))

        self.elbo = 0

    def MB(
        self,
        z_node,
        y_m_node,
        tau_m_node,
        w_m_node_not_sparse=None,
        hat_w_m_node=None,
        s_m_node=None,
        alpha_m_node=None,
        theta_m_node=None,
        tilde_w_m_node=None,
        p_m_node=None,
    ):
        # regardless of W type
        self.z_node = z_node
        self.y_m_node = y_m_node
        self.tau_m_node = tau_m_node

        # Sparsity: None
        self.w_m_node_not_sparse = w_m_node_not_sparse

        # ARD/spike and slab
        self.hat_w_m_node = hat_w_m_node
        self.s_m_node = s_m_node
        self.alpha_m_node = alpha_m_node
        self.theta_m_node = theta_m_node

        # sparse pathways
        self.tilde_w_m_node = tilde_w_m_node
        self.p_m_node = p_m_node

        self.update_params()
        self.update_params_z()

    def update(self):
        for k in range(self.params.K):
            self.update_k(k)

        # Only CTM doesn't need tau updates - all FA likelihoods (Normal/Bernoulli) do
        if self.params.likelihoods[self.m] != Likelihood.CTM:
            self.tau_m_node.update_params_w_z()

    def update_k(self, k):
        if self.params.likelihoods[self.m] in [Likelihood.NORMAL, Likelihood.BERNOULLI]:
            # here we have these options: None/ARD/ARD+SS/Pathways

            if self.params.W_priors[self.m] == WPrior.NONE:
                # sum_j!=k <z_nk><z_nj><w_jd>
                nominator_second_term = np.dot(self.E_w, self.z_node.E_z.T) - np.outer(
                    self.E_w[:, k], self.z_node.E_z[:, k]
                )
                nominator_resid = self.y_m_node.data - nominator_second_term.T
                nominator = np.ma.dot(
                    self.z_node.E_z[:, k], self.tau_m_node.E_tau * nominator_resid
                )

                denominator = (
                    np.ma.dot(self.z_node.E_z_squared[:, k], self.tau_m_node.E_tau) + 1
                )

                self.w_m_node_not_sparse.update_k(k, nominator, denominator)
                self.update_params()
                self.update_params_z()

            if (
                self.params.W_priors[self.m] == WPrior.ARD
                or self.params.W_priors[self.m] == WPrior.ARD_SS
            ):
                # sum_j!=k <z_nk><z_nj><w_jd>
                nominator_second_term_tmp = np.dot(
                    self.E_w, self.z_node.E_z.T * self.z_node.E_z[:, k]
                ).T
                nominator_second_term = nominator_second_term_tmp - np.outer(
                    self.z_node.E_z[:, k] ** 2, self.E_w[:, k]
                )
                nominator = np.ma.sum(
                    self.tau_m_node.E_tau
                    * (
                        (self.y_m_node.data.T * self.z_node.E_z[:, k]).T
                        - nominator_second_term
                    ),
                    axis=0,
                )

                denominator = (
                    np.ma.dot(self.z_node.E_z_squared[:, k], self.tau_m_node.E_tau)
                    + self.alpha_m_node.E_alpha[k]
                )

                self.hat_w_m_node.update_k(k, nominator, denominator)

                if self.params.W_priors[self.m] == WPrior.ARD_SS:  # TBD: check
                    E_log_LR_theta = self.theta_m_node.E_log_LR[k].copy()
                    self.s_m_node.update_k(
                        k,
                        nominator,
                        denominator,
                        self.alpha_m_node.E_alpha[k],
                        E_log_LR_theta,
                    )

                self.update_params()
                self.update_params_z()
                self.alpha_m_node.update_k(k)
                if self.params.W_priors[self.m] == WPrior.ARD_SS:
                    self.theta_m_node.update_k(k)

            if self.params.W_priors[self.m] == WPrior.PATHWAYS:
                pass

        if self.params.likelihoods[self.m] in [Likelihood.CTM]:
            # here we have these options: None/ARD

            E_zE_zk = np.dot(self.z_node.E_z.T, self.z_node.E_z[:, k])
            E_zE_zk[k] = 0
            EwE_zE_zk = np.dot(self.E_w, E_zE_zk)
            term1 = np.dot(self.tau_m_node.Sigma0_inv, EwE_zE_zk)

            Ez_squaredk = np.sum(self.z_node.E_z_squared[:, k])
            Sigma_inv_nodiag = self.tau_m_node.Sigma0_inv - np.diag(
                np.diag(self.tau_m_node.Sigma0_inv)
            )
            term2 = np.dot(Sigma_inv_nodiag, self.E_w[:, k]) * Ez_squaredk

            term3 = np.dot(
                self.z_node.E_z[:, k],
                np.ma.dot(self.y_m_node.data, self.tau_m_node.Sigma0_inv),
            )

            nominator = term3 - term1 - term2

            if self.params.W_priors[self.m] == WPrior.NONE:
                denominator = np.diag(
                    np.eye(self.params.D[self.m])
                    + np.sum(self.z_node.E_z_squared[:, k])
                    * (self.tau_m_node.Sigma0_inv)
                )
                self.w_m_node_not_sparse.update_k(k, nominator, denominator)
                self.update_params()
                self.update_params_z()

            if self.params.W_priors[self.m] == WPrior.ARD:
                denominator = np.diag(
                    self.alpha_m_node.E_alpha[k] * np.eye(self.params.D[self.m])
                    + np.sum(self.z_node.E_z_squared[:, k])
                    * (self.tau_m_node.Sigma0_inv)
                )
                self.hat_w_m_node.update_k(k, nominator, denominator)
                self.update_params()
                self.update_params_z()
                self.alpha_m_node.update_k(k)

    def update_params(self):
        if self.params.W_priors[self.m] == WPrior.NONE:
            self.E_w = self.w_m_node_not_sparse.vi_mu
            self.E_w_squared = (
                self.w_m_node_not_sparse.vi_var + self.w_m_node_not_sparse.vi_mu**2
            )

        if self.params.W_priors[self.m] == WPrior.ARD:
            self.E_w = self.hat_w_m_node.vi_mu
            self.E_w_squared = self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2

            self.E_hat_w_squared = self.E_w_squared

        if self.params.W_priors[self.m] == WPrior.ARD_SS:
            self.E_w = self.s_m_node.vi_gamma * self.hat_w_m_node.vi_mu
            self.E_w_squared = self.s_m_node.vi_gamma * (
                self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2
            )

            self.E_hat_w_squared = self.s_m_node.vi_gamma * (
                self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2
            ) + (1 - self.s_m_node.vi_gamma) * (
                np.outer(
                    np.ones(self.params.D[self.m]),
                    self.alpha_m_node.E_inv_alpha,
                )
            )

        if self.params.W_priors[self.m] == WPrior.PATHWAYS:
            pass

    def update_params_z(self):
        self.E_w_z = np.dot(self.E_w, self.z_node.E_z.T).T

        # sum of squares sum_k (E W_kd Z_nk)**2
        term_tmp = np.dot(self.E_w, self.z_node.E_z.T)
        first_term = (term_tmp) ** 2
        # sum oif second moments sum_k (E W_kd**2 Z_nk**2)
        second_term = np.dot(self.E_w_squared, self.z_node.E_z_squared.T)
        # sum of squares (just the k=k' cases) sum_k (E W_kd Z_nk)**2
        third_term = np.dot(self.E_w**2, self.z_node.E_z.T**2)
        self.E_w_z_squared = (first_term + second_term - third_term).T

    def ELBO(self):
        if self.params.W_priors[self.m] == WPrior.NONE:
            elbo = (
                self.params.D[self.m] * self.params.K / 2
                - np.sum(self.E_w_squared) / 2
                + np.sum(log_eps(self.w_m_node_not_sparse.vi_var)) / 2
            )

        if self.params.W_priors[self.m] == WPrior.ARD:
            elbo = (
                self.params.D[self.m] * self.params.K / 2
                + self.params.D[self.m] * np.sum(self.alpha_m_node.E_log_alpha) / 2
                - np.sum(self.alpha_m_node.E_alpha * self.E_w_squared) / 2
                + np.sum(log_eps(self.hat_w_m_node.vi_var)) / 2
            )

        if self.params.W_priors[self.m] == WPrior.ARD_SS:
            elbo_s = (
                np.sum(np.dot(self.theta_m_node.E_log_theta, self.s_m_node.vi_gamma.T))
                + np.sum(
                    np.dot(
                        self.theta_m_node.E_log_1minustheta,
                        (1 - self.s_m_node.vi_gamma).T,
                    )
                )
                - np.sum(xlogx(self.s_m_node.vi_gamma))
                - np.sum(xlogx(1 - self.s_m_node.vi_gamma))
            )
            elbo_w_hat = (
                self.params.D[self.m] * np.sum(self.alpha_m_node.E_log_alpha) / 2
                - (
                    np.sum(
                        np.dot(
                            (1 - self.s_m_node.vi_gamma),
                            self.alpha_m_node.E_log_alpha,
                        )
                    )
                    / 2
                )
                - np.sum(np.dot(self.E_hat_w_squared, self.alpha_m_node.E_alpha)) / 2
                + np.sum(self.s_m_node.vi_gamma * log_eps(self.hat_w_m_node.vi_var)) / 2
            )
            elbo = elbo_s + elbo_w_hat

        self.elbo = elbo


# nodeFA_alpha_m is now imported from w_priors.ARD
# nodeFA_theta_m is now imported from w_priors.ARD_SS


class nodeFA_tau_m:
    """
    Class to define tau node (dim d, for one m)
    """

    def __init__(self, a0, b0, m, params: FAParams):
        self.params = params

        self.m = m

        if not (self.params.likelihoods[self.m] == Likelihood.CTM):
            self.a0 = a0
            self.b0 = b0

            self.E_resid_squared_half = 0

            self.elbo = 0

        if self.params.likelihoods[self.m] == Likelihood.CTM:
            self.Sigma0_inv = np.eye(self.params.D[self.m])

    def MB(self, y_m_node, w_m_node, z_node):
        self.w_m_node = w_m_node
        self.y_m_node = y_m_node
        self.z_node = z_node

        # we need MB for this, so it is here and not in init
        if not (self.params.likelihoods[self.m] == Likelihood.CTM):
            self.vi_a = (
                self.a0 + (self.params.N - np.sum(self.y_m_node.data.mask, axis=0)) / 2
            )
            self.vi_b = self.vi_a.copy()
            self.update_all_params()
            self.update_params()

    def update(self):
        self.update_params_w_z()
        self.vi_b = self.b0 + np.ma.sum(self.E_resid_squared_half, axis=0)

        self.update_params()

    def update_all_params(self):
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        # self.update_params()

        self.kl_const = -self.params.D[self.m] * (self.log_gamma_a0) + self.params.D[
            self.m
        ] * (self.a0 * log_eps(self.b0))
        self.entropy_cons = (
            np.sum(self.vi_a)
            + np.sum(self.log_gamma_vi_a)
            + np.sum((1 - self.vi_a) * self.digamma_vi_a)
        )

    def update_params(self):
        self.E_tau_1D = self.vi_a / self.vi_b
        self.E_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_tau_1D),
            mask=self.y_m_node.data.mask,
        )

        self.E_log_tau_1D = -log_eps(self.vi_b) + self.digamma_vi_a
        self.E_log_tau = np.ma.array(
            np.outer(np.ones(self.params.N), self.E_log_tau_1D),
            mask=self.y_m_node.data.mask,
        )

    def update_params_w_z(self):
        # d x n, sum over n

        # <(y_nd - \sum_k z_nk w_kd)**2>/2
        third_term_of_tau = (
            np.ma.array(self.w_m_node.E_w_z_squared, mask=self.y_m_node.data.mask) / 2
        )
        first_term_of_tau = self.y_m_node.data**2 / 2
        second_term_of_tau = -self.y_m_node.data * self.w_m_node.E_w_z

        self.E_resid_squared_half = (
            first_term_of_tau + second_term_of_tau + third_term_of_tau
        )

    def ELBO(self):
        kl = (
            self.kl_const
            + (self.a0 - 1) * np.sum(self.E_log_tau_1D)
            - self.b0 * np.sum(self.E_tau_1D)
        )
        entropy = self.entropy_cons - np.sum(log_eps(self.vi_b))
        self.elbo = kl + entropy


class nodeFA_y_m:
    def __init__(self, data_n, m, params: FAParams):
        self.params = params

        self.m = m

        # there is no difference between data and data_original
        # for likelihood == "normal"
        # however, for likelihood == 'Bernoulli", data_original is 0/1 and data is
        # an approximation used for updates
        self.data = data_n
        self.data_original = data_n

        self.elbo = 0

    def MB(self, w_m_node, tau_m_node):
        self.w_m_node = w_m_node
        self.tau_m_node = tau_m_node

    def ELBO(self):
        elbo = (
            -(self.params.N * self.params.D[self.m] - np.sum(self.data.mask))
            * log_eps(2 * np.pi)
            / 2
            + np.ma.sum(self.tau_m_node.E_log_tau) / 2
            - np.ma.sum(self.tau_m_node.E_tau * self.tau_m_node.E_resid_squared_half)
        )
        self.elbo = elbo


def starting_params_z(starting_params, N, K):
    if "z_mu" in starting_params.keys():
        z_mean = 1 * starting_params["z_mu"]
    else:
        z_mean = np.random.normal(size=(N, K))

    if "z_var" in starting_params.keys():
        z_var = starting_params["z_var"]
    else:
        z_var = np.ones((N, K))

    return z_mean, z_var


def starting_params_hat_w_m(starting_params, key_M, D, K):
    starting_params_m = starting_params[key_M]

    if "w_mu" in starting_params_m.keys():
        w_mean = 1 * starting_params_m["w_mu"]
    else:
        w_mean = np.random.normal(size=(D, K))

    if "w_var" in starting_params_m.keys():
        w_var = 1 * starting_params_m["w_var"]
    else:
        w_var = np.ones((D, K))

    return w_mean, w_var


def starting_params_s_m(starting_params, key_M, D, K):
    starting_params_m = starting_params[key_M]

    if "s_lambda" in starting_params_m.keys():
        s_lambda = 1.0 * starting_params_m["s_lambda"]
    else:
        # start with p(s=1) > 0.999
        s_lambda = 10.0 * np.ones((D, K))

    return s_lambda


class FA:
    """
    Class to define Factor Analysis model.
    """

    def __init__(
        self,
        data,
        N,
        M,
        K,
        D,
        likelihoods,
        Z_priors,
        W_priors,
        starting_params=None,
        view_configs=None,
        *args,
        **kwargs,
    ):
        self.N = N  # number of observations (samples)
        self.M = M  # number of views
        self.K = K  # number of hidden factors
        self.D = D  # a list containing number of features in each view

        self.likelihoods = likelihoods
        self.Z_priors = Z_priors
        self.W_priors = W_priors
        self.view_configs = view_configs  # ViewConfig objects for each view

        # Create shared parameters object
        self.params = FAParams(
            N=self.N,
            K=self.K,
            D=self.D,
            M=self.M,
            likelihoods=likelihoods,
            Z_priors=Z_priors,
            W_priors=W_priors,
        )

        # starting options
        if starting_params is None:
            starting_params = dict()
        for m in range(M):
            if "M" + str(m) not in starting_params.keys():
                starting_params.update({"M" + str(m): dict()})

        # CREATING NODES:
        z_mean, z_var = starting_params_z(starting_params, self.N, self.K)
        self.node_z = nodeFA_z(vi_mu=z_mean, vi_var=z_var, params=self.params)

        self.nodelist_y = []

        self.nodelist_w_not_sparse = []

        self.nodelist_hat_w = []
        self.nodelist_alpha = []

        self.nodelist_s = []
        self.nodelist_theta = []

        # TBD
        # self.nodelist_w_pathways = []

        self.nodelist_w = []

        self.nodelist_tau = []

        for m in range(self.M):
            key_tmp = "M" + str(m)
            data_m = data[key_tmp]

            # Create y node using view_config if available, otherwise check if CTM
            if self.view_configs is not None:
                # Use ViewConfig to create y node (handles Normal/Bernoulli/CTM)
                node_y_m = self.view_configs[m].create_y_node(data_m, m, self.params)
            else:
                # Fallback for backward compatibility - only distinguish CTM from FA
                from FACTMpy.FA_model import nodeFA_y_m

                if self.likelihoods[m] == Likelihood.CTM:
                    node_y_m = nodeFA_y_m(None, m, params=self.params)
                else:
                    # For FA likelihoods (Normal/Bernoulli), treat them the same
                    # Both Normal and Bernoulli use the same FA structure
                    import numpy as np

                    data_m = np.array(data_m)
                    data_m = np.ma.array(data_m, mask=np.isnan(data_m))
                    # Default to Normal behavior (centering) for backward compatibility
                    feature_mean_m = np.ma.mean(data_m, axis=0)
                    node_y_m = nodeFA_y_m(
                        data_m - feature_mean_m, m, params=self.params
                    )
                    node_y_m.data_mean = feature_mean_m
            self.nodelist_y.append(node_y_m)

            if self.W_priors[m] == WPrior.NONE:
                w_mu, w_var = starting_params_hat_w_m(starting_params, key_tmp, D[m], K)
                node_w_m_not_sparse = nodeFA_w_m_not_sparse(
                    w_mu, w_var, m, params=self.params
                )
                self.nodelist_w_not_sparse.append(node_w_m_not_sparse)
            else:
                self.nodelist_w_not_sparse.append(None)
            if (self.W_priors[m] == WPrior.ARD) or (self.W_priors[m] == WPrior.ARD_SS):
                w_mu, w_var = starting_params_hat_w_m(starting_params, key_tmp, D[m], K)
                node_hat_w_m = nodeFA_hat_w_m(w_mu, w_var, m, params=self.params)
                self.nodelist_hat_w.append(node_hat_w_m)
                node_alpha_m = nodeFA_alpha_m(1e-14, 1e-14, m, params=self.params)
                self.nodelist_alpha.append(node_alpha_m)
            else:
                self.nodelist_hat_w.append(None)
                self.nodelist_alpha.append(None)
            if self.W_priors[m] == WPrior.ARD_SS:
                s_lambda = starting_params_s_m(starting_params, key_tmp, D[m], K)
                node_s = nodeFA_s_m(s_lambda, m, params=self.params)
                self.nodelist_s.append(node_s)
                node_theta_m = nodeFA_theta_m(
                    1, 1, 99 * np.ones(K), np.ones(K), m, params=self.params
                )
                self.nodelist_theta.append(node_theta_m)
            else:
                self.nodelist_s.append(None)
                self.nodelist_theta.append(None)
            node_w_m = nodeFA_w_m(m, params=self.params)
            self.nodelist_w.append(node_w_m)

            node_tau_m = nodeFA_tau_m(0.001, 0.001, m, params=self.params)
            self.nodelist_tau.append(node_tau_m)

        self.elbo = 0

    def MB(self):
        self.node_z.MB(self.nodelist_y, self.nodelist_w, self.nodelist_tau)

        for m in range(self.M):
            self.nodelist_y[m].MB(self.nodelist_w[m], self.nodelist_tau[m])

            self.nodelist_w[m].MB(
                self.node_z,
                self.nodelist_y[m],
                self.nodelist_tau[m],
                self.nodelist_w_not_sparse[m],
                self.nodelist_hat_w[m],
                self.nodelist_s[m],
                self.nodelist_alpha[m],
                self.nodelist_theta[m],
                None,
                None,
            )

            if self.W_priors[m] in [WPrior.ARD, WPrior.ARD_SS]:
                self.nodelist_alpha[m].MB(
                    self.nodelist_hat_w[m], self.nodelist_s[m], self.nodelist_w[m]
                )
            if self.W_priors[m] == WPrior.ARD_SS:
                self.nodelist_theta[m].MB(self.nodelist_s[m])

            self.nodelist_tau[m].MB(self.nodelist_y[m], self.nodelist_w[m], self.node_z)

    def update(self):
        # update Z
        self.node_z.update()

        # update W
        # and all the nodes defying sparsity
        #  - it depends on tau params, but not tau_w_z
        for m in range(self.M):
            self.nodelist_w[m].update()

        # update tau by m - only for FA likelihoods, not CTM
        for m in range(self.M):
            if self.likelihoods[m] != Likelihood.CTM:
                self.nodelist_tau[m].update()

    def ELBO(self):
        # compute elbo
        self.node_z.ELBO()
        for m in range(self.M):
            self.nodelist_w[m].ELBO()
            if self.W_priors[m] in [WPrior.ARD_SS]:
                self.nodelist_theta[m].ELBO()
            if self.W_priors[m] in [WPrior.ARD, WPrior.ARD_SS]:
                self.nodelist_alpha[m].ELBO()
            if self.likelihoods[m] != Likelihood.CTM:
                self.nodelist_tau[m].ELBO()

                self.nodelist_y[m].ELBO()

        # update self.elbo
        elbo = 0
        elbo += self.node_z.elbo
        for m in range(self.M):
            elbo += self.nodelist_w[m].elbo
            if self.W_priors[m] in [WPrior.ARD_SS]:
                elbo += self.nodelist_theta[m].elbo
            if self.W_priors[m] in [WPrior.ARD, WPrior.ARD_SS]:
                elbo += self.nodelist_alpha[m].elbo
            if self.likelihoods[m] != Likelihood.CTM:
                elbo += self.nodelist_tau[m].elbo
            elbo += self.nodelist_y[m].elbo

        self.elbo = elbo

    def get_elbo(self):
        return self.elbo

    def variance_explained_per_factor(self):
        var_exp_nominator = np.zeros(self.K)
        var_exp_denominator = np.zeros(self.K)

        for k in range(self.K):
            for m in range(self.M):
                var_exp_nominator[k] += np.ma.sum(
                    (
                        self.nodelist_y[m].data
                        - np.outer(self.node_z.E_z[:, k], self.nodelist_w[m].E_w[:, k])
                    )
                    ** 2
                )
                var_exp_denominator[k] += np.ma.sum(self.nodelist_y[m].data ** 2)

        return 1 - var_exp_nominator / var_exp_denominator

    def variance_explained_per_view(self):
        var_exp_nominator = np.zeros(self.M)
        var_exp_denominator = np.zeros(self.M)

        for m in range(self.M):
            var_exp_nominator[m] += np.ma.sum(
                (
                    self.nodelist_y[m].data
                    - np.dot(self.node_z.E_z, self.nodelist_w[m].E_w.T)
                )
                ** 2
            )
            var_exp_denominator[m] += np.ma.sum(self.nodelist_y[m].data ** 2)

        return 1 - var_exp_nominator / var_exp_denominator

    def variance_explained_per_factor_view(self):
        var_exp_nominator = np.zeros((self.K, self.M))
        var_exp_denominator = np.zeros((self.K, self.M))

        for k in range(self.K):
            for m in range(self.M):
                var_exp_nominator[k, m] = np.ma.sum(
                    (
                        self.nodelist_y[m].data
                        - np.outer(self.node_z.E_z[:, k], self.nodelist_w[m].E_w[:, k])
                    )
                    ** 2
                )
                var_exp_denominator[k, m] = np.ma.sum(self.nodelist_y[m].data ** 2)

        return 1 - var_exp_nominator / var_exp_denominator
