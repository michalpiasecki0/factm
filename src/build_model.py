"""
High-level entry point: FACTModel.

Accepts a :class:`~views.Views` container and a :class:`~model_config.ModelConfig`
instead of the old ``{"M0": ..., "M1": ...}`` dict + flat config lists.
All M-string indexing has been removed.
"""

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .CTM_model import CTM
from .enums import FA_Pretrain, WPrior
from .FACTM_model import FACTM
from .model_config import ModelConfig
from .views import Views


class FACTModel(FACTM):
    """
    Convenience wrapper around FACTM that adds pre-training and fitting helpers.

    Parameters
    ----------
    views:        :class:`~views.Views` with .simple and .structured lists
    K:            number of latent factors
    model_config: :class:`~model_config.ModelConfig`
    seed:         optional random seed
    """

    def __init__(
        self,
        views: Views,
        K: int,
        model_config: ModelConfig,
        seed: int | None = None,
        starting_params_fa: dict | None = None,
        starting_params_ctm: list[dict] | None = None,
    ):
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

        self.K = K
        self.views = views
        self.model_config = model_config

        # Validate consistency between views and config
        if views.num_simple != model_config.num_simple:
            raise ValueError(
                f"Number of simple views in data ({views.num_simple}) must match "
                f"model_config ({model_config.num_simple})."
            )
        if views.num_structured != model_config.num_structured:
            raise ValueError(
                f"Number of structured views in data ({views.num_structured}) must "
                f"match model_config ({model_config.num_structured})."
            )
        if model_config.num_factors != K:
            raise ValueError(
                f"model_config has {model_config.num_factors} Z priors but K={K}."
            )

        super().__init__(
            views=views,
            K=K,
            model_config=model_config,
            starting_params_fa=starting_params_fa,
            starting_params_ctm=starting_params_ctm,
        )

        self.__first_fit = True
        self.elbo_sequence = []

    # ------------------------------------------------------------------
    # Pre-training
    # ------------------------------------------------------------------

    def pretrain(
        self,
        FA_pretrain: FA_Pretrain = FA_Pretrain.PCA,
    ) -> None:
        """Pre-train CTMs then initialise FA weights via PCA or sklearn FA."""
        if FA_pretrain not in (FA_Pretrain.PCA, FA_Pretrain.FA):
            raise TypeError("FA_pretrain must be one of the FA_Pretrain enum values.")

        # 1. Pre-train each CTM independently
        for i, (sv, ctm) in enumerate(
            zip(self.views.structured, self.ctms, strict=True)
        ):
            print(f"Pretraining CTM for structured view {i}")

            tmp_ctm = CTM(
                data=sv.data,
                N=self.N,
                L=ctm.L,
                G=ctm.G,
                K=self.fa.K,
                FA=False,
            )
            tmp_ctm.MB()
            for _ in range(5):
                tmp_ctm.update()

            tmp_ctm.FA = True
            self.ctms[i] = tmp_ctm

            # Push updated eta signal into FA's proxy y node
            fa_idx = self._fa_indices[i]
            self.fa.nodelist_y[fa_idx].data = (
                tmp_ctm.node_eta.vi_mu - tmp_ctm.node_mu0.mu0
            )

        # 2. Pre-train FA weights with PCA / sklearn FactorAnalysis
        print("Pretraining FA")

        # Collect and standardise all view data (simple + CTM proxy)
        data_parts = []
        for m in range(len(self.fa.nodelist_y)):
            d = self.fa.nodelist_y[m].data
            data_parts.append(
                (d - np.nanmean(d, axis=0)) / (np.nanstd(d, axis=0) + 1e-10)
            )
        data_combined = np.hstack(data_parts)

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        data_combined = imp.fit_transform(data_combined)

        if FA_pretrain == FA_Pretrain.PCA:
            reducer = PCA(n_components=self.fa.K, whiten=True)
        else:
            reducer = FactorAnalysis(n_components=self.fa.K)

        reducer.fit(data_combined)
        loadings_all = reducer.components_.T  # (total_D, K)
        latent_factors = reducer.transform(data_combined)

        # Distribute loadings back to per-view W nodes
        D_cumsum = [0] + list(
            np.cumsum(
                [
                    self.fa.nodelist_y[m].data.shape[1]
                    for m in range(len(self.fa.nodelist_y))
                ]
            )
        )

        for m in range(len(self.fa.nodelist_w)):
            loadings_m = loadings_all[D_cumsum[m] : D_cumsum[m + 1], :]
            std_m = np.std(self.fa.nodelist_y[m].data, axis=0)

            cfg = (
                self.model_config.simple_view_configs[m]
                if m < self.views.num_simple
                else self.model_config.structured_view_configs[
                    m - self.views.num_simple
                ]
            )

            if cfg.w_prior in (WPrior.ARD, WPrior.ARD_SS):
                self.fa.nodelist_hat_w[m].vi_mu = (std_m * loadings_m.T).T
            elif cfg.w_prior == WPrior.NONE:
                self.fa.nodelist_w_not_sparse[m].vi_mu = (std_m * loadings_m.T).T

        # Initialise Z
        min_max_scaler = MinMaxScaler((-1, 1))
        self.fa.node_z.vi_mu = min_max_scaler.fit_transform(latent_factors)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        max_iter: int = 1000,
        elbo_tres: float = 0.0,
        pretrain: bool = True,
    ) -> None:
        if self.__first_fit:
            self.MB()

            if pretrain:
                self.pretrain()

            # One initial deterministic update to populate expectations
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

        self.__first_fit = False
        print("Fitting model")

        svi = getattr(self.model_config, "svi", None)
        if svi is not None and svi.enabled:
            t = 0
            for it in tqdm(range(max_iter)):
                if svi.batch_size is not None:
                    S = min(int(svi.batch_size), self.N)
                else:
                    frac = (
                        svi.batch_fraction
                        if svi.batch_fraction is not None
                        else self.model_config.node_update_factor
                    )
                    S = max(1, int(round(frac * self.N)))

                indices = np.random.choice(self.N, size=S, replace=False)

                if it < svi.warmup_iters or not svi.stochastic_fa_globals:
                    # Warmup: keep global updates deterministic/full-data.
                    self.update(update_factor=float(S) / float(self.N))
                else:
                    # Experimental: repeat indices to length N instead of using
                    # N/S scaling inside the minibatch sufficient statistics.
                    if svi.repeat_minibatch and S < self.N:
                        reps = int(np.ceil(float(self.N) / float(S)))
                        indices = np.tile(indices, reps)[: self.N]

                    rho = float(svi.rho_tau) / (
                        (1.0 + float(svi.rho_kappa) * float(t)) ** float(svi.rho_power)
                    )
                    rho_min, rho_max = svi.rho_clip
                    rho = float(np.clip(rho, rho_min, rho_max))
                    self.update_svi(indices=indices, rho=rho)
                    t += 1

                if (it % svi.elbo_every) == 0:
                    self.ELBO()
                    self.elbo_sequence.append(self.get_elbo())

                    if (
                        elbo_tres
                        and len(self.elbo_sequence) >= 2
                        and (self.elbo_sequence[-1] - self.elbo_sequence[-2])
                        < elbo_tres
                    ):
                        break
            return

        for _ in tqdm(range(max_iter)):
            self.update(update_factor=self.model_config.node_update_factor)
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

            if self.elbo_sequence[-1] - self.elbo_sequence[-2] < elbo_tres:
                break

    # ------------------------------------------------------------------
    # Point estimators
    # ------------------------------------------------------------------

    def get_latent_factors(self) -> np.ndarray:
        return self.fa.node_z.vi_mu

    def get_loadings(self, view_idx: int) -> np.ndarray:
        """Loadings for simple view at index ``view_idx``."""
        return self.fa.nodelist_w[view_idx].E_w

    def get_mu0(self, structured_idx: int) -> np.ndarray:
        return self.ctms[structured_idx].node_mu0.mu0

    def get_Sigma0(self, structured_idx: int) -> np.ndarray:
        return self.ctms[structured_idx].node_Sigma0.Sigma0

    def get_topics(self, structured_idx: int) -> np.ndarray:
        ctm = self.ctms[structured_idx]
        L = ctm.L
        return np.array(
            [
                ctm.node_beta.vi_alpha[li, :] / np.sum(ctm.node_beta.vi_alpha[li, :])
                for li in range(L)
            ]
        )

    def get_eta(self, structured_idx: int) -> np.ndarray:
        return self.ctms[structured_idx].node_eta.vi_mu

    def get_eta_probabilities_of_topics(self, structured_idx: int) -> np.ndarray:
        ctm = self.ctms[structured_idx]
        prob = np.exp(self.get_eta(structured_idx))
        return prob / np.outer(np.sum(prob, axis=1), np.ones(ctm.L))

    def get_probabilities_of_topics(self, structured_idx: int) -> np.ndarray:
        return self.ctms[structured_idx].node_xi.vi_par

    def get_clusters(self, structured_idx: int) -> list:
        return [
            np.argmax(self.ctms[structured_idx].node_xi.vi_par[n], axis=1)
            for n in range(self.N)
        ]

    def get_predictions_FA(self, view_idx: int) -> np.ndarray:
        """Predicted values for simple view at index ``view_idx``."""
        return self.fa.nodelist_w[view_idx].E_w_z
