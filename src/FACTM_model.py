"""
Factor Analysis with Correlated Topic Model (FACTM).

Joins the FA part (over simple views) with one CTM per structured view.
All M-string indexing is gone; simple and structured views are iterated
over their own lists independently.
"""

import numpy as np

from .CTM_model import CTM
from .FA_model import FA
from .model_config import ModelConfig
from .nodes import nodeFA_tau_m, nodeFA_y_m
from .views import Views


class FACTM:
    """
    Factor Analysis with Correlated Topic Model.

    Parameters
    ----------
    views:        Views container with .simple and .structured lists
    K:            number of latent factors
    model_config: ModelConfig with simple_view_configs and
                  structured_view_configs
    starting_params_fa:  optional dict for FA initialisation
    starting_params_ctm: optional list[dict], one per structured view
    """

    def __init__(
        self,
        views: Views,
        K: int,
        model_config: ModelConfig,
        starting_params_fa=None,
        starting_params_ctm=None,
    ):
        self.views = views
        self.K = K
        self.model_config = model_config

        self.N = views.N

        # ------------------------------------------------------------------
        # FA — operates on simple views only
        # ------------------------------------------------------------------
        self.fa = FA(
            views=views,
            K=K,
            model_config=model_config,
            starting_params=starting_params_fa,
        )

        # ------------------------------------------------------------------
        # CTMs — one per structured view
        # ------------------------------------------------------------------
        num_structured = views.num_structured

        if starting_params_ctm is None:
            starting_params_ctm = [{} for _ in range(num_structured)]

        # For each structured view we:
        #   1. Build a CTM object
        #   2. Inject a proxy y/tau node into FA so Z sees the eta signal
        #   3. Record the FA node-list index so we can sync parameters later
        self.ctms: list[CTM] = []
        self._fa_indices: list[int] = []  # FA nodelist index per structured view

        for structured_view, cfg, starting_param_ctm in zip(
            views.structured,
            model_config.structured_view_configs,
            starting_params_ctm,
            strict=True,
        ):
            ctm = CTM(
                data=structured_view.data,
                N=self.N,
                L=cfg.L,
                G=structured_view.G,
                K=K,
                starting_params=starting_param_ctm,
            )
            self.ctms.append(ctm)

            # Proxy y node carries eta − mu0 from this CTM into FA
            proxy_y = nodeFA_y_m(
                data_n=ctm.node_eta.E_eta_minus_mu0,
                m=None,
                params=self.fa.params,
                D=cfg.L,
            )

            w_prior_obj = cfg.create_w_prior()

            fa_idx = self.fa.inject_structured_view(
                y_node=proxy_y,
                tau_node=nodeFA_tau_m(
                    0.001, 0.001, m=None, params=self.fa.params, D=cfg.L, is_ctm=True
                ),
                w_prior=w_prior_obj,
                starting_params={},
                D_m=cfg.L,
            )
            self._fa_indices.append(fa_idx)

    # ------------------------------------------------------------------
    # MB
    # ------------------------------------------------------------------

    def MB(self):
        self.fa.MB()
        for ctm in self.ctms:
            ctm.MB()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, update_factor: float | None = None):
        # 1. Update FA (Z and simple-view W nodes)
        indices = (
            np.random.choice(self.N, size=int(update_factor * self.N), replace=False)
            if update_factor
            else None
        )
        self.fa.update(indices=indices)

        # 2. Sync FA → CTM and update each CTM
        for ctm, fa_idx in zip(self.ctms, self._fa_indices, strict=True):
            w_node_fa = self.fa.nodelist_w[fa_idx]

            # Push current FA estimates of W and Z into the CTM's w_z node
            ctm.node_w_z.E_w = w_node_fa.E_w.copy()
            ctm.node_w_z.E_w_squared = w_node_fa.E_w_squared.copy()
            ctm.node_w_z.E_z = self.fa.node_z.E_z.copy()
            ctm.node_w_z.E_z_squared = self.fa.node_z.E_z_squared
            ctm.node_w_z.E_w_z = w_node_fa.E_w_z.copy()
            ctm.node_w_z.E_w_z_squared = w_node_fa.E_w_z_squared.copy()

            ctm.update(indices=indices)

            # 3. Push CTM outputs back into FA's proxy nodes
            self.fa.nodelist_y[fa_idx].data = ctm.node_eta.E_eta_minus_mu0.copy()
            self.fa.nodelist_tau[fa_idx].Sigma0_inv = ctm.node_Sigma0.inv_Sigma0.copy()

    def update_svi(self, indices: np.ndarray, rho: float):
        if indices is None:
            raise ValueError("indices must be provided for SVI update.")

        self.fa.update_svi(indices=indices, rho=rho)

        for ctm, fa_idx in zip(self.ctms, self._fa_indices, strict=True):
            w_node_fa = self.fa.nodelist_w[fa_idx]
            ctm.node_w_z.E_w = w_node_fa.E_w.copy()
            ctm.node_w_z.E_w_squared = w_node_fa.E_w_squared.copy()
            ctm.node_w_z.E_z = self.fa.node_z.E_z.copy()
            ctm.node_w_z.E_z_squared = self.fa.node_z.E_z_squared
            ctm.node_w_z.E_w_z = w_node_fa.E_w_z.copy()
            ctm.node_w_z.E_w_z_squared = w_node_fa.E_w_z_squared.copy()

            ctm.update(indices=indices)

            self.fa.nodelist_y[fa_idx].data = ctm.node_eta.E_eta_minus_mu0.copy()
            self.fa.nodelist_tau[fa_idx].Sigma0_inv = ctm.node_Sigma0.inv_Sigma0.copy()

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------

    def ELBO(self):
        self.fa.ELBO()
        for ctm in self.ctms:
            ctm.ELBO()

    def get_elbo(self) -> float:
        return self.fa.elbo + sum(ctm.elbo for ctm in self.ctms)
