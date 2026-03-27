"""
Factor Analysis model.

FA now operates exclusively on simple (Normal / Bernoulli) views.
CTM (structured) views are owned by FACTM, which passes only the
derived eta signal to FA via injected y-node data.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .enums import Likelihood
from .model_config import ModelConfig, SimpleViewConfig
from .nodes import nodeFA_tau_m, nodeFA_y_m
from .starting_params import starting_params_z
from .views import Views
from .z_priors.StdNormal import nodeFA_z

EPS = 1e-20


@dataclass
class FAParams:
    """Shared parameters for Factor Analysis model."""

    N: int  # number of observations (samples)
    K: int  # number of hidden factors
    D: List[int]  # number of features per simple view
    M: int  # number of simple views
    likelihoods: List[Likelihood]
    Z_priors: list
    W_priors: List[str]


class nodeFA_w_m:
    """
    W node (D_m × K) for one simple view m.

    Gathers together the nodes that specify the sparsity structure,
    e.g. W_hat / S (ARD + spike-and-slab) or plain W (no sparsity).
    """

    def __init__(
        self, m: int | None, params: FAParams, w_prior, D: int, is_ctm: bool = False
    ):
        self.params = params
        self.w_prior = w_prior

        self.m = m  # integer index into node lists; None for injected CTM nodes
        self.D = D  # feature dimension for this view
        self.is_ctm = is_ctm

        self.E_w = np.zeros((self.D, self.params.K))
        self.E_w_squared = np.ones((self.D, self.params.K))

        self.E_w_z = np.zeros((self.params.N, self.D))
        self.E_w_z_squared = np.zeros((self.params.N, self.D))

        # Sub-nodes — populated by MB(); initialised to None so that
        # update_params() can be called before MB() if needed.
        self.z_node = None
        self.y_m_node = None
        self.tau_m_node = None
        self.w_m_node_not_sparse = None
        self.hat_w_m_node = None
        self.s_m_node = None
        self.alpha_m_node = None
        self.theta_m_node = None

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
    ):
        self.z_node = z_node
        self.y_m_node = y_m_node
        self.tau_m_node = tau_m_node

        self.w_m_node_not_sparse = w_m_node_not_sparse
        self.hat_w_m_node = hat_w_m_node
        self.s_m_node = s_m_node
        self.alpha_m_node = alpha_m_node
        self.theta_m_node = theta_m_node

        self.update_params()
        self.update_params_z()

    def update(self, indices: np.ndarray | None = None, data_scale: float = 1.0):
        for k in range(self.params.K):
            self.update_k(k, indices=indices, data_scale=data_scale)
        if not self.is_ctm:
            self.tau_m_node.update_params_w_z()

    def update_k(
        self, k: int, indices: np.ndarray | None = None, data_scale: float = 1.0
    ):
        self.w_prior.update_w_k(
            self,
            k,
            self.z_node,
            self.y_m_node,
            self.tau_m_node,
            indices=indices,
            data_scale=data_scale,
        )

    def update_params(self):
        self.w_prior.update_params(self)

    def update_params_z(self):
        self.E_w_z = np.dot(self.E_w, self.z_node.E_z.T).T

        term_tmp = np.dot(self.E_w, self.z_node.E_z.T)
        first_term = term_tmp**2
        second_term = np.dot(self.E_w_squared, self.z_node.E_z_squared.T)
        third_term = np.dot(self.E_w**2, self.z_node.E_z.T**2)
        self.E_w_z_squared = (first_term + second_term - third_term).T

    def ELBO(self):
        self.elbo = self.w_prior.compute_elbo(self)


class FA:
    """
    Factor Analysis model over simple (Normal / Bernoulli) views only.

    Parameters
    ----------
    views:        Views container (only .simple is used)
    K:            number of latent factors
    model_config: ModelConfig — only simple_view_configs are consumed here
    starting_params: optional dict for initialisation
    """

    def __init__(
        self,
        views: Views,
        K: int,
        model_config: ModelConfig,
        starting_params=None,
    ):
        self.views = views
        self.K = K
        self.model_config = model_config

        self.N = views.N
        self.M = views.num_simple  # number of simple views
        self.D = [sv.D for sv in views.simple]

        self.simple_view_configs: List[
            SimpleViewConfig
        ] = model_config.simple_view_configs

        # Create shared FAParams (used by sub-nodes)
        self.params = FAParams(
            N=self.N,
            K=self.K,
            D=self.D,
            M=self.M,
            likelihoods=[c.likelihood for c in self.simple_view_configs],
            Z_priors=model_config.z_priors,
            W_priors=[c.w_prior for c in self.simple_view_configs],
        )

        # Starting params
        if starting_params is None:
            starting_params = {}
        for m in range(self.M):
            starting_params.setdefault(m, {})

        z_mean, z_var = starting_params_z(starting_params, self.N, self.K)
        z_prior_objs = model_config.create_z_priors()
        self.node_z = nodeFA_z(
            vi_mu=z_mean,
            vi_var=z_var,
            params=self.params,
            z_priors=z_prior_objs,
        )

        # Per-view node lists (indexed by simple-view index, no "M0" strings)
        self.nodelist_y: List[nodeFA_y_m] = []
        self.nodelist_w: List[nodeFA_w_m] = []
        self.nodelist_tau: List[nodeFA_tau_m] = []

        # Sparsity helper nodes (parallel lists, entries may be None)
        self.nodelist_w_not_sparse = []
        self.nodelist_hat_w = []
        self.nodelist_s = []
        self.nodelist_alpha = []
        self.nodelist_theta = []

        for m, (simple_view, simple_view_cfg) in enumerate(
            zip(views.simple, self.simple_view_configs, strict=True)
        ):
            # y node
            node_y_m = simple_view_cfg.create_y_node(simple_view.data, m, self.params)
            self.nodelist_y.append(node_y_m)

            # W prior + sparsity nodes
            w_prior_obj = simple_view_cfg.create_w_prior()
            prior_nodes = w_prior_obj.create_nodes(
                m, self.params, starting_params, self.D[m], K
            )
            self.nodelist_w_not_sparse.append(prior_nodes["w_m_node_not_sparse"])
            self.nodelist_hat_w.append(prior_nodes["hat_w_m_node"])
            self.nodelist_s.append(prior_nodes["s_m_node"])
            self.nodelist_alpha.append(prior_nodes["alpha_m_node"])
            self.nodelist_theta.append(prior_nodes["theta_m_node"])

            node_w_m = nodeFA_w_m(
                m, params=self.params, w_prior=w_prior_obj, D=self.D[m], is_ctm=False
            )
            node_w_m.w_m_node_not_sparse = prior_nodes["w_m_node_not_sparse"]
            node_w_m.hat_w_m_node = prior_nodes["hat_w_m_node"]
            node_w_m.s_m_node = prior_nodes["s_m_node"]
            node_w_m.alpha_m_node = prior_nodes["alpha_m_node"]
            node_w_m.theta_m_node = prior_nodes["theta_m_node"]
            self.nodelist_w.append(node_w_m)

            node_tau_m = nodeFA_tau_m(
                0.001, 0.001, m, params=self.params, D=self.D[m], is_ctm=False
            )
            self.nodelist_tau.append(node_tau_m)

        self.elbo = 0

    # ------------------------------------------------------------------
    # Extra y / tau nodes injected by FACTM for structured views
    # ------------------------------------------------------------------

    def inject_structured_view(
        self,
        y_node: nodeFA_y_m,
        tau_node: nodeFA_tau_m,
        w_prior,
        starting_params: dict,
        D_m: int,
    ) -> int:
        """
        Append a proxy y/tau/w node pair representing one structured (CTM) view.

        Called by FACTM once per CTM view so that FA's Z update can see
        the eta signal from each CTM.  Returns the index into the node lists
        so that FACTM can later update the injected nodes directly.
        """
        m = len(self.nodelist_y)  # next index in the extended list

        self.nodelist_y.append(y_node)

        # Extend FAParams D and likelihoods to cover the injected view
        self.params.D.append(D_m)
        self.params.likelihoods.append(Likelihood.CTM)
        self.params.M += 1

        # W node for the CTM view (FA uses it to update Z)
        prior_nodes = w_prior.create_nodes(m, self.params, starting_params, D_m, self.K)
        self.nodelist_w_not_sparse.append(prior_nodes["w_m_node_not_sparse"])
        self.nodelist_hat_w.append(prior_nodes["hat_w_m_node"])
        self.nodelist_s.append(prior_nodes["s_m_node"])
        self.nodelist_alpha.append(prior_nodes["alpha_m_node"])
        self.nodelist_theta.append(prior_nodes["theta_m_node"])

        node_w_m = nodeFA_w_m(
            m, params=self.params, w_prior=w_prior, D=D_m, is_ctm=True
        )
        node_w_m.w_m_node_not_sparse = prior_nodes["w_m_node_not_sparse"]
        node_w_m.hat_w_m_node = prior_nodes["hat_w_m_node"]
        node_w_m.s_m_node = prior_nodes["s_m_node"]
        node_w_m.alpha_m_node = prior_nodes["alpha_m_node"]
        node_w_m.theta_m_node = prior_nodes["theta_m_node"]
        self.nodelist_w.append(node_w_m)

        self.nodelist_tau.append(tau_node)

        return m  # caller stores this index to update the injected node later

    # ------------------------------------------------------------------
    # MB / update / ELBO  (unchanged logic, just no Likelihood.CTM guards
    # needed for simple-view-only loops; injected CTM nodes are handled
    # via their index returned from inject_structured_view)
    # ------------------------------------------------------------------

    def MB(self):
        self.node_z.MB(self.nodelist_y, self.nodelist_w, self.nodelist_tau)

        for m in range(len(self.nodelist_y)):
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
            )

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

    def update(
        self,
        indices: np.ndarray | None = None,
        data_scale: float = 1.0,
        svi_rho: float = 1.0,
    ):
        # Local update for Z (factors) on a minibatch.
        self.node_z.update(indices=indices)

        # Algorithm 3: rho mixing for global variational params.
        # Only blend global parameters (W/tau and sparsity/noise hyperparameters),
        # never blend local parameters (Z).
        do_mix = abs(svi_rho - 1.0) > 1e-12
        rho = float(svi_rho)
        one_minus_rho = 1.0 - rho

        if do_mix:
            w_old = []
            for w_node in self.nodelist_w:
                entry = {}
                if w_node.hat_w_m_node is not None:
                    entry["hat_w_mu"] = w_node.hat_w_m_node.vi_mu.copy()
                    entry["hat_w_var"] = w_node.hat_w_m_node.vi_var.copy()
                if w_node.w_m_node_not_sparse is not None:
                    entry["w_mu"] = w_node.w_m_node_not_sparse.vi_mu.copy()
                    entry["w_var"] = w_node.w_m_node_not_sparse.vi_var.copy()
                if w_node.s_m_node is not None:
                    entry["s_lambda"] = w_node.s_m_node.vi_lambda.copy()
                if w_node.alpha_m_node is not None:
                    entry["alpha_b"] = w_node.alpha_m_node.vi_b.copy()
                if w_node.theta_m_node is not None:
                    entry["theta_a"] = w_node.theta_m_node.vi_a.copy()
                    entry["theta_b"] = w_node.theta_m_node.vi_b.copy()
                w_old.append(entry)

            tau_old = [self.nodelist_tau[m].vi_b.copy() for m in range(self.M)]

        # Candidate global updates (minibatch only; scaled by data_scale).
        for m in range(len(self.nodelist_w)):
            self.nodelist_w[m].update(indices=indices, data_scale=data_scale)

        # Tau update only for simple views (not CTM-injected ones)
        for m in range(self.M):
            self.nodelist_tau[m].update(indices=indices, data_scale=data_scale)

        if not do_mix:
            return

        # Apply rho mixing and recompute derived expectations to remain consistent.
        for w_node, entry in zip(self.nodelist_w, w_old, strict=True):
            if w_node.hat_w_m_node is not None:
                w_node.hat_w_m_node.vi_mu[:] = (
                    one_minus_rho * entry["hat_w_mu"] + rho * w_node.hat_w_m_node.vi_mu
                )
                w_node.hat_w_m_node.vi_var[:] = (
                    one_minus_rho * entry["hat_w_var"]
                    + rho * w_node.hat_w_m_node.vi_var
                )

            if w_node.w_m_node_not_sparse is not None:
                w_node.w_m_node_not_sparse.vi_mu[:] = (
                    one_minus_rho * entry["w_mu"]
                    + rho * w_node.w_m_node_not_sparse.vi_mu
                )
                w_node.w_m_node_not_sparse.vi_var[:] = (
                    one_minus_rho * entry["w_var"]
                    + rho * w_node.w_m_node_not_sparse.vi_var
                )

            if w_node.s_m_node is not None:
                w_node.s_m_node.vi_lambda[:] = (
                    one_minus_rho * entry["s_lambda"] + rho * w_node.s_m_node.vi_lambda
                )
                # nodeFA_s_m stores vi_gamma derived from vi_lambda.
                w_node.s_m_node.vi_gamma = 1.0 / (
                    1.0 + np.exp(-w_node.s_m_node.vi_lambda)
                )

            if w_node.alpha_m_node is not None:
                w_node.alpha_m_node.vi_b[:] = (
                    one_minus_rho * entry["alpha_b"] + rho * w_node.alpha_m_node.vi_b
                )
                w_node.alpha_m_node.update_params()

            if w_node.theta_m_node is not None:
                w_node.theta_m_node.vi_a[:] = (
                    one_minus_rho * entry["theta_a"] + rho * w_node.theta_m_node.vi_a
                )
                w_node.theta_m_node.vi_b[:] = (
                    one_minus_rho * entry["theta_b"] + rho * w_node.theta_m_node.vi_b
                )
                w_node.theta_m_node.update_params()

            w_node.update_params()
            w_node.update_params_z()

        # Blend tau (only simple views) and recompute residual expectations.
        for m in range(self.M):
            tau_node = self.nodelist_tau[m]
            tau_node.vi_b[:] = one_minus_rho * tau_old[m] + rho * tau_node.vi_b
            tau_node.update_params()
            tau_node.update_params_w_z()

    def ELBO(self):
        self.node_z.ELBO()

        for m in range(len(self.nodelist_w)):
            self.nodelist_w[m].ELBO()
            additional_nodes = self.nodelist_w[
                m
            ].w_prior.get_additional_nodes_to_update()
            if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
                self.nodelist_theta[m].ELBO()
            if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
                self.nodelist_alpha[m].ELBO()

        # ELBO contributions for simple views only
        for m in range(self.M):
            self.nodelist_tau[m].ELBO()
            self.nodelist_y[m].ELBO()

        elbo = self.node_z.elbo
        for m in range(len(self.nodelist_w)):
            elbo += self.nodelist_w[m].elbo
            additional_nodes = self.nodelist_w[
                m
            ].w_prior.get_additional_nodes_to_update()
            if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
                elbo += self.nodelist_theta[m].elbo
            if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
                elbo += self.nodelist_alpha[m].elbo

        for m in range(self.M):
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
