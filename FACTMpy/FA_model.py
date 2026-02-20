"""
This module provides Factor Analysis model with all the nodes.
"""
import importlib
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.special import digamma, gammaln

from FACTMpy.model_config import Likelihood
from FACTMpy.utils import log_eps
from FACTMpy.z_priors.StdNormal import nodeFA_z

# Import None prior using importlib since "None" is a Python keyword
_none_prior = importlib.import_module("FACTMpy.w_priors.None")
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


# nodeFA_z is now imported from FACTMpy.z_priors.StdNormal
# nodeFA_w_m_not_sparse is now imported from FACTMpy.w_priors.None


# nodeFA_w_m_not_sparse is now imported from FACTMpy.w_priors.None
# nodeFA_hat_w_m is now imported from FACTMpy.w_priors.ARD
# nodeFA_s_m is now imported from FACTMpy.w_priors.ARD_SS


class nodeFA_w_m:
    """
    Class to define W node (d times k, for one m)
    Gathers together the nodes specifying sparsity structure
    e.g. W_hat and S or W_tilde and P (TBD)
    """

    def __init__(self, m, params: FAParams, w_prior):
        self.params = params
        self.w_prior = w_prior  # WPriorBase object (required)

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
        # Use prior object to update - w_prior should always be available
        self.w_prior.update_w_k(self, k, self.z_node, self.y_m_node, self.tau_m_node)

    def update_params(self):
        # Use prior object to update params - w_prior should always be available
        self.w_prior.update_params(self)

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
        # Use prior object to compute ELBO - w_prior should always be available
        self.elbo = self.w_prior.compute_elbo(self)


# nodeFA_alpha_m is now imported from FACTMpy.w_priors.ARD
# nodeFA_theta_m is now imported from FACTMpy.w_priors.ARD_SS


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
        # Store model_config if available (for creating Z priors)
        self.model_config = kwargs.get("model_config", None)

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
        # Create Z prior objects if model_config available
        z_prior_objs = None
        if self.model_config is not None:
            z_prior_objs = self.model_config.create_z_priors()
        self.node_z = nodeFA_z(
            vi_mu=z_mean, vi_var=z_var, params=self.params, z_priors=z_prior_objs
        )

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

            # Create W prior object - always use view_configs when available
            if self.view_configs is not None:
                w_prior_obj = self.view_configs[m].create_w_prior()
                # Create nodes using prior object
                prior_nodes = w_prior_obj.create_nodes(
                    m, self.params, starting_params, D[m], K
                )
                self.nodelist_w_not_sparse.append(prior_nodes["w_m_node_not_sparse"])
                self.nodelist_hat_w.append(prior_nodes["hat_w_m_node"])
                self.nodelist_s.append(prior_nodes["s_m_node"])
                self.nodelist_alpha.append(prior_nodes["alpha_m_node"])
                self.nodelist_theta.append(prior_nodes["theta_m_node"])
            else:
                # Fallback for backward compatibility - create prior objects from enums
                from FACTMpy.w_priors.factory import create_w_prior

                w_prior_obj = create_w_prior(self.W_priors[m])
                # Create nodes using prior object
                prior_nodes = w_prior_obj.create_nodes(
                    m, self.params, starting_params, D[m], K
                )
                self.nodelist_w_not_sparse.append(prior_nodes["w_m_node_not_sparse"])
                self.nodelist_hat_w.append(prior_nodes["hat_w_m_node"])
                self.nodelist_s.append(prior_nodes["s_m_node"])
                self.nodelist_alpha.append(prior_nodes["alpha_m_node"])
                self.nodelist_theta.append(prior_nodes["theta_m_node"])

            # Create w_m node with prior object (always available now)
            node_w_m = nodeFA_w_m(m, params=self.params, w_prior=w_prior_obj)
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

            # Use prior object to determine which nodes need MB
            # w_prior should always be available now (created in __init__)
            additional_nodes = self.nodelist_w[
                m
            ].w_prior.get_additional_nodes_to_update()
            if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
                self.nodelist_alpha[m].MB(
                    self.nodelist_hat_w[m], self.nodelist_s[m], self.nodelist_w[m]
                )
            if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
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
            # Use prior object to determine which nodes need ELBO
            # w_prior should always be available now (created in __init__)
            additional_nodes = self.nodelist_w[
                m
            ].w_prior.get_additional_nodes_to_update()
            if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
                self.nodelist_theta[m].ELBO()
            if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
                self.nodelist_alpha[m].ELBO()
            if self.likelihoods[m] != Likelihood.CTM:
                self.nodelist_tau[m].ELBO()
                self.nodelist_y[m].ELBO()

        # update self.elbo
        elbo = 0
        elbo += self.node_z.elbo
        for m in range(self.M):
            elbo += self.nodelist_w[m].elbo
            # Use prior object to determine which nodes contribute to ELBO
            # w_prior should always be available now (created in __init__)
            additional_nodes = self.nodelist_w[
                m
            ].w_prior.get_additional_nodes_to_update()
            if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
                elbo += self.nodelist_theta[m].elbo
            if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
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
