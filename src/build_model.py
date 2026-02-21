import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .CTM_model import CTM
from .enums import FA_Pretrain, Likelihood, WPrior
from .FACTM_model import FACTM
from .model_config import ModelConfig


class FACTModel(FACTM):
    def __init__(
        self,
        data,
        K,
        model_config: ModelConfig,
        seed=None,
        *args,
        **kwargs,
    ):
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)

        self.K = K
        self.data = data
        self.model_config = model_config

        self.__assign_params()

        super(FACTModel, self).__init__(
            data,
            self.N,
            self.M,
            self.K,
            self.D,
            self.G,
            self.likelihoods,
            self.z_priors,
            self.w_priors,
            starting_params_fa=None,
            starting_params_ctm=None,
            view_configs=self.model_config.view_configs,
            model_config=self.model_config,
        )

        self.__first_fit = True

        self.elbo_sequence = []

    def pretrain(self, FA_pretrain=FA_Pretrain.PCA, CTM_pretrain=Likelihood.CTM):
        if FA_pretrain not in [FA_Pretrain.PCA, FA_Pretrain.FA]:
            raise TypeError(
                "Pretraining for FA part should be one of FA_Pretrain enum values"
            )
        if CTM_pretrain != CTM_pretrain.CTM:
            raise TypeError(
                "Pretraining for CTM part should be one of CTM_pretrain enum values"
            )

        if CTM_pretrain == CTM_pretrain.CTM:
            for m in range(self.M):
                if self.likelihoods[m] == Likelihood.CTM:
                    print("Pretraining CTM for a modality " + str(m))

                    mod_ctm_tmp = CTM(
                        self.ctm_list["M" + str(m)].node_y.data,
                        self.N,
                        self.ctm_list["M" + str(m)].L,
                        self.ctm_list["M" + str(m)].G,
                        self.fa.K,
                        FA=False,
                    )

                    mod_ctm_tmp.MB()

                    for _ in range(5):
                        mod_ctm_tmp.update()

                    mod_ctm_tmp.FA = True
                    self.ctm_list["M" + str(m)] = mod_ctm_tmp

                    self.fa.nodelist_y[m].data = (
                        self.ctm_list["M" + str(m)].node_eta.vi_mu
                        - self.ctm_list["M" + str(m)].node_mu0.mu0
                    )

        print("Pretraining FA")

        if FA_pretrain == FA_Pretrain.PCA:
            modFA = PCA(n_components=self.fa.K, whiten=True)
        if FA_pretrain == FA_Pretrain.FA:
            modFA = FactorAnalysis(n_components=self.fa.K)

        # scaled and centered data
        if CTM_pretrain == CTM_pretrain.CTM:
            data_tmp = []

            for m in range(self.M):
                data_tmp_m = self.fa.nodelist_y[m].data
                data_tmp.append(
                    (data_tmp_m - np.nanmean(data_tmp_m, axis=0))
                    / np.nanstd(data_tmp_m, axis=0)
                )
            data_tmp = np.hstack(data_tmp)

        D_tmp = [self.D[m] for m in range(self.M)]

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        data_tmp = imp.fit_transform(data_tmp)
        modFA.fit(data_tmp)

        views_segments = [0] + np.cumsum(np.array(D_tmp)).tolist()
        loadings_tmp = modFA.components_.T
        latent_factors_tmp = modFA.transform(data_tmp)

        for m in range(self.M):
            # get weights + scale back according to the variance of the features
            if self.w_priors[m] in [WPrior.ARD, WPrior.ARD_SS]:
                self.fa.nodelist_hat_w[m].vi_mu = (
                    np.std(self.fa.nodelist_y[m].data, axis=0)
                    * loadings_tmp[views_segments[m] : views_segments[m + 1], :].T
                ).T
            if self.w_priors[m] == WPrior.NONE:
                self.fa.nodelist_w_not_sparse[m].vi_mu = (
                    np.std(self.fa.nodelist_y[m].data, axis=0)
                    * loadings_tmp[views_segments[m] : views_segments[m + 1], :].T
                ).T

        # scale to [-1, 1] as in MOFA
        min_max_scaler = MinMaxScaler((-1, 1))
        self.fa.node_z.vi_mu = min_max_scaler.fit_transform(latent_factors_tmp)

    def fit(self, max_iter=1000, elbo_tres=0, pretrain=True):
        if self.__first_fit:
            self.MB()

            # pretrain
            if pretrain:
                self.pretrain()

            # elbo at start
            # update to make sure all the params make sense
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

        self.__first_fit = False

        print("Fitting a model")

        for _ in tqdm(range(max_iter)):
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

            if self.elbo_sequence[-1] - self.elbo_sequence[-2] < elbo_tres:
                break

    # Point estimators
    def get_latent_factors(self):
        return self.fa.node_z.vi_mu

    def get_loadings(self, m):
        return self.fa.nodelist_w[m].E_w

    def get_mu0(self, m):
        return self.ctm_list["M" + str(m)].node_mu0.mu0

    def get_Sigma0(self, m):
        return self.ctm_list["M" + str(m)].node_Sigma0.Sigma0

    def get_topics(self, m):
        L_m = self.L[np.where(np.array(self.index_CTM) == m)[0][0]]
        return np.array(
            [
                self.ctm_list["M" + str(m)].node_beta.vi_alpha[l_from_L_m, :]
                / np.sum(self.ctm_list["M" + str(m)].node_beta.vi_alpha[l_from_L_m, :])
                for l_from_L_m in range(L_m)
            ]
        )

    def get_eta(self, m):
        return self.ctm_list["M" + str(m)].node_eta.vi_mu

    def get_eta_probabilities_of_topics(self, m):
        L_m = self.L[np.where(np.array(self.index_CTM) == m)[0][0]]
        prob_est = np.exp(self.get_eta(m))
        prob_est = prob_est / np.outer(np.sum(prob_est, axis=1), np.ones(L_m))
        return prob_est

    def get_probabilities_of_topics(self, m):
        return self.ctm_list["M" + str(m)].node_xi.vi_par

    def get_clusters(self, m):
        return [
            np.argmax(self.ctm_list["M" + str(m)].node_xi.vi_par[n], axis=1)
            for n in range(self.N)
        ]

    def get_predictions_FA(self, m):
        return self.fa.nodelist_w[m].E_w_z

    def __assign_params(self):
        """Extract parameters from model_config."""
        # Validate number of views matches data
        self.M = len(self.data)
        if self.model_config.num_views != self.M:
            raise ValueError(
                f"Number of views in model_config ({self.model_config.num_views}) "
                f"must match number of views in data ({self.M})"
            )

        # Validate number of factors
        if self.model_config.num_factors != self.K:
            raise ValueError(
                f"Number of factors in model_config ({self.model_config.num_factors}) "
                f"must match K ({self.K})"
            )

        # Extract likelihoods and W_priors from model_config
        # Convert to strings for backward compatibility with FACTM/FA classes
        self.likelihoods = self.model_config.likelihoods
        self.w_priors = self.model_config.w_priors
        self.z_priors = self.model_config.z_priors

        # Extract L values from CTM view configs
        self.L = self.model_config.L

        # N - determine from first view
        if self.likelihoods[0] == Likelihood.CTM:
            # if M0 a structured view
            self.N = len(self.data["M0"])
        else:
            # if M0 a simple view
            self.N = self.data["M0"].shape[0]

        # D, G and the number of structured views
        D = []
        G = []

        m_ctm = 0

        for m in range(self.M):
            if self.likelihoods[m] != Likelihood.CTM:
                D.append(self.data["M" + str(m)].shape[1])
            else:
                D.append(self.L[m_ctm])
                m_ctm += 1
                G.append(self.data["M" + str(m)][0].shape[1])

        self.M_CTM = m_ctm
        self.D = D
        self.G = G
