"""
Factor Analysis with Correlated Topic Model (FACTM).

Joins the FA part (over simple views) with one CTM per structured view.
"""

import numpy as np

from .CTM_model import CTM
from .FA_model import FA, nodeFA_w_m
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
        starting_params_fa: dict | None = None,
        starting_params_ctm: list[dict] | None = None,
    ):
        self.views = views
        self.K = K
        self.model_config = model_config
        self.N = views.N

        self.fa = FA(
            views=views,
            K=K,
            model_config=model_config,
            starting_params=starting_params_fa,
        )

        if starting_params_ctm is None:
            starting_params_ctm = [{} for _ in range(views.num_structured)]

        self.ctms: list[CTM] = []
        self._fa_indices: list[int] = []

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

            proxy_y = nodeFA_y_m(
                data_n=ctm.node_eta.E_eta_minus_mu0,
                m=None,
                params=self.fa.params,
                D=cfg.L,
            )
            fa_idx = self.fa.inject_structured_view(
                y_node=proxy_y,
                tau_node=nodeFA_tau_m(
                    0.001, 0.001, m=None, params=self.fa.params, D=cfg.L, is_ctm=True
                ),
                w_prior=cfg.create_w_prior(),
                starting_params={},
                D_m=cfg.L,
            )
            self._fa_indices.append(fa_idx)

    @staticmethod
    def _sample_indices(N: int, update_factor: float | None) -> np.ndarray | None:
        if update_factor is None:
            return None
        return np.random.choice(N, size=int(update_factor * N), replace=False)

    @staticmethod
    def _sync_fa_to_ctm(
        ctm: CTM, w_node_fa: nodeFA_w_m, E_z: np.ndarray, E_z_squared: np.ndarray
    ) -> None:
        ctm.node_w_z.E_w = w_node_fa.E_w.copy()
        ctm.node_w_z.E_w_squared = w_node_fa.E_w_squared.copy()
        ctm.node_w_z.E_z = E_z.copy()
        ctm.node_w_z.E_z_squared = E_z_squared
        ctm.node_w_z.E_w_z = w_node_fa.E_w_z.copy()
        ctm.node_w_z.E_w_z_squared = w_node_fa.E_w_z_squared.copy()

    def _sync_ctm_to_fa(self, ctm: CTM, fa_idx: int) -> None:
        self.fa.nodelist_y[fa_idx].data = ctm.node_eta.E_eta_minus_mu0.copy()
        self.fa.nodelist_tau[fa_idx].Sigma0_inv = ctm.node_Sigma0.inv_Sigma0.copy()

    def _update_ctms(self, indices: np.ndarray | None, *, svi: bool, rho: float = 0.0):
        for ctm, fa_idx in zip(self.ctms, self._fa_indices, strict=True):
            w_node_fa = self.fa.nodelist_w[fa_idx]
            self._sync_fa_to_ctm(
                ctm,
                w_node_fa,
                self.fa.node_z.E_z,
                self.fa.node_z.E_z_squared,
            )

            if svi:
                ctm.update_svi(indices=indices, rho=rho)
            else:
                ctm.update(indices=indices)

            self._sync_ctm_to_fa(ctm, fa_idx)

    def MB(self):
        self.fa.MB()
        for ctm in self.ctms:
            ctm.MB()

    def update(self, update_factor: float | None = None):
        indices = self._sample_indices(self.N, update_factor)
        self.fa.update(indices=indices)
        self._update_ctms(indices, svi=False)

    def update_svi(self, indices: np.ndarray, rho: float):
        if indices is None:
            raise ValueError("indices must be provided for SVI update.")

        self.fa.update_svi(indices=indices, rho=rho)
        self._update_ctms(indices, svi=True, rho=rho)

    def ELBO(self):
        self.fa.ELBO()
        for ctm in self.ctms:
            ctm.ELBO()

    def get_elbo(self) -> float:
        return self.fa.elbo + sum(ctm.elbo for ctm in self.ctms)
