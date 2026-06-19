"""
Factor Analysis model.

FA now operates exclusively on simple (Normal / Bernoulli) views.
CTM (structured) views are owned by FACTM, which passes only the
derived eta signal to FA via injected y-node data.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .model_config import ModelConfig, SimpleViewConfig
from .nodes import nodeFA_tau_m, nodeFA_y_m
from .starting_params import starting_params_z
from .views import Views
from .z_priors.StdNormal import nodeFA_z


@dataclass
class FAParams:
    """Shared parameters for Factor Analysis model."""

    N: int
    K: int
    D: List[int]
    Z_priors: list


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

    def update(self):
        for k in range(self.params.K):
            self.update_k(k)
        if not self.is_ctm:
            self.tau_m_node.update_params_w_z()

    def update_k(self, k):
        self.w_prior.update_w_k(self, k, self.z_node, self.y_m_node, self.tau_m_node)

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

    @staticmethod
    def _ema_inplace(curr: np.ndarray, target: np.ndarray, rho: float) -> np.ndarray:
        return (1.0 - rho) * curr + rho * target

    def _svi_update_w_params_for_k(self, k: int, targets: dict, rho: float) -> None:
        if self.w_m_node_not_sparse is not None:
            self.w_m_node_not_sparse.vi_mu[:, k] = self._ema_inplace(
                self.w_m_node_not_sparse.vi_mu[:, k], targets["mu"], rho
            )
            self.w_m_node_not_sparse.vi_var[:, k] = self._ema_inplace(
                self.w_m_node_not_sparse.vi_var[:, k], targets["var"], rho
            )
            return

        if "mu" in targets:
            self.hat_w_m_node.vi_mu[:, k] = self._ema_inplace(
                self.hat_w_m_node.vi_mu[:, k], targets["mu"], rho
            )
            self.hat_w_m_node.vi_var[:, k] = self._ema_inplace(
                self.hat_w_m_node.vi_var[:, k], targets["var"], rho
            )
            return

        # Spike-and-slab-style targets
        self.hat_w_m_node.vi_mu[:, k] = self._ema_inplace(
            self.hat_w_m_node.vi_mu[:, k], targets["mu_hat"], rho
        )
        self.hat_w_m_node.vi_var[:, k] = self._ema_inplace(
            self.hat_w_m_node.vi_var[:, k], targets["var_hat"], rho
        )

        self.s_m_node.vi_lambda[:, k] = self._ema_inplace(
            self.s_m_node.vi_lambda[:, k], targets["lambda"], rho
        )
        self.s_m_node.vi_gamma[:, k] = 1.0 / (
            1.0 + np.exp(-self.s_m_node.vi_lambda[:, k])
        )

    def _svi_update_alpha_for_k(self, k: int, rho: float) -> None:
        if self.alpha_m_node is None:
            return
        vi_b_target = self.alpha_m_node.b0 + np.sum(self.E_hat_w_squared[:, k]) / 2
        self.alpha_m_node.vi_b[k] = self._ema_inplace(
            self.alpha_m_node.vi_b[k], vi_b_target, rho
        )
        self.alpha_m_node.update_params()

    def _svi_update_theta_for_k(self, k: int, rho: float) -> None:
        if self.theta_m_node is None or self.s_m_node is None:
            return
        sum_sdk = np.sum(self.s_m_node.vi_gamma[:, k])
        vi_a_target = self.theta_m_node.a0 + sum_sdk
        vi_b_target = self.theta_m_node.b0 - sum_sdk + self.D
        self.theta_m_node.vi_a[k] = self._ema_inplace(
            self.theta_m_node.vi_a[k], vi_a_target, rho
        )
        self.theta_m_node.vi_b[k] = self._ema_inplace(
            self.theta_m_node.vi_b[k], vi_b_target, rho
        )
        self.theta_m_node.update_params()

    def svi_update(self, indices: np.ndarray, rho: float):
        """
        Stochastic mean-field update for global W (+ sparsity nodes).
        """
        if self.is_ctm:
            # Keep CTM-injected W deterministic (not treated as global SVI here).
            self.update()
            return

        for k in range(self.params.K):
            targets = self.w_prior.svi_target_w_k(
                self, k, self.z_node, self.y_m_node, self.tau_m_node, indices
            )

            self._svi_update_w_params_for_k(k, targets=targets, rho=rho)
            self.update_params()

            self._svi_update_alpha_for_k(k, rho=rho)
            self._svi_update_theta_for_k(k, rho=rho)

        self.update_params_z()


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
            Z_priors=model_config.z_priors,
        )

        # Starting params
        if starting_params is None:
            starting_params = {}
        for m in range(self.M):
            starting_params.setdefault(m, {})

        z_mean, z_var = starting_params_z(starting_params, self.N, self.K)
        cohorts = views.cohorts
        if cohorts is None and "cohort_labels" in starting_params:
            cohorts = np.asarray(starting_params["cohort_labels"])
        z_prior_objs = model_config.create_z_priors(cohorts=cohorts)
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
            self._register_w_prior_nodes(
                m, w_prior_obj, prior_nodes, self.D[m], is_ctm=False
            )

            node_tau_m = nodeFA_tau_m(
                0.001, 0.001, m, params=self.params, D=self.D[m], is_ctm=False
            )
            self.nodelist_tau.append(node_tau_m)

        self.elbo = 0

    def _register_w_prior_nodes(
        self,
        m: int,
        w_prior,
        prior_nodes: dict,
        D_m: int,
        *,
        is_ctm: bool,
    ) -> nodeFA_w_m:
        self.nodelist_w_not_sparse.append(prior_nodes["w_m_node_not_sparse"])
        self.nodelist_hat_w.append(prior_nodes["hat_w_m_node"])
        self.nodelist_s.append(prior_nodes["s_m_node"])
        self.nodelist_alpha.append(prior_nodes["alpha_m_node"])
        self.nodelist_theta.append(prior_nodes["theta_m_node"])

        node_w_m = nodeFA_w_m(
            m, params=self.params, w_prior=w_prior, D=D_m, is_ctm=is_ctm
        )
        node_w_m.w_m_node_not_sparse = prior_nodes["w_m_node_not_sparse"]
        node_w_m.hat_w_m_node = prior_nodes["hat_w_m_node"]
        node_w_m.s_m_node = prior_nodes["s_m_node"]
        node_w_m.alpha_m_node = prior_nodes["alpha_m_node"]
        node_w_m.theta_m_node = prior_nodes["theta_m_node"]
        self.nodelist_w.append(node_w_m)
        return node_w_m

    def _mb_w_sparsity_nodes(self, m: int) -> None:
        additional_nodes = self.nodelist_w[m].w_prior.get_additional_nodes_to_update()
        if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
            self.nodelist_alpha[m].MB(
                self.nodelist_hat_w[m], self.nodelist_s[m], self.nodelist_w[m]
            )
        if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
            self.nodelist_theta[m].MB(self.nodelist_s[m])

    def _elbo_w_sparsity_nodes(self, m: int) -> None:
        additional_nodes = self.nodelist_w[m].w_prior.get_additional_nodes_to_update()
        if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
            self.nodelist_theta[m].ELBO()
        if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
            self.nodelist_alpha[m].ELBO()

    def _sparsity_elbo_contribution(self, m: int) -> float:
        additional_nodes = self.nodelist_w[m].w_prior.get_additional_nodes_to_update()
        contribution = 0.0
        if "theta" in additional_nodes and self.nodelist_theta[m] is not None:
            contribution += self.nodelist_theta[m].elbo
        if "alpha" in additional_nodes and self.nodelist_alpha[m] is not None:
            contribution += self.nodelist_alpha[m].elbo
        return contribution

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
        self.params.D.append(D_m)

        prior_nodes = w_prior.create_nodes(m, self.params, starting_params, D_m, self.K)
        self._register_w_prior_nodes(m, w_prior, prior_nodes, D_m, is_ctm=True)
        self.nodelist_tau.append(tau_node)

        return m

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

            self._mb_w_sparsity_nodes(m)
            self.nodelist_tau[m].MB(self.nodelist_y[m], self.nodelist_w[m], self.node_z)

    def update(self, indices: np.ndarray | None = None):
        self.node_z.update(indices=indices)

        for m in range(len(self.nodelist_w)):
            self.nodelist_w[m].update()

        # Tau is updated only for simple views, not CTM proxy nodes.
        for m in range(self.M):
            self.nodelist_tau[m].update()

    def update_svi(self, indices: np.ndarray, rho: float):
        if indices is None:
            indices = np.random.choice(self.N, size=min(32, self.N), replace=False)

        # Local updates
        self.node_z.update(indices=indices)

        # Global updates
        for m in range(len(self.nodelist_w)):
            self.nodelist_w[m].svi_update(indices=indices, rho=rho)

        for m in range(self.M):
            self.nodelist_tau[m].svi_update(indices=indices, rho=rho)

    def ELBO(self):
        self.node_z.ELBO()

        for m in range(len(self.nodelist_w)):
            self.nodelist_w[m].ELBO()
            self._elbo_w_sparsity_nodes(m)

        for m in range(self.M):
            self.nodelist_tau[m].ELBO()
            self.nodelist_y[m].ELBO()

        elbo = self.node_z.elbo
        for m in range(len(self.nodelist_w)):
            elbo += self.nodelist_w[m].elbo
            elbo += self._sparsity_elbo_contribution(m)

        for m in range(self.M):
            elbo += self.nodelist_tau[m].elbo
            elbo += self.nodelist_y[m].elbo

        self.elbo = elbo

    def get_elbo(self) -> float:
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
