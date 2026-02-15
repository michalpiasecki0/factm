"""
This module provides Factor Analysis with Correlated Topic Model
by joining FA and CTM parts.
"""

import numpy as np
from CTM_model import CTM
from FA_model import FA
from model_config import Likelihood


class FACTM:
    """
    Class to define Factor Analysis with Correlated Topic Model.
    """

    def __init__(
        self,
        data,
        N,
        M,
        K,
        D,
        G,
        likelihoods,
        Z_priors,
        W_priors,
        starting_params_fa=None,
        starting_params_ctm=None,
        *args,
        **kwargs,
    ):
        self.likelihoods = likelihoods
        self.Z_priors = Z_priors
        self.W_priors = W_priors

        self.N = N
        self.M = M
        self.D = D
        # Numbers of niches (topics) in not observed modalities
        self.L = [D[m] for m in range(M) if self.likelihoods[m] == Likelihood.CTM]
        # Numbers of types (e.g. cell types, words) in not observed modalities
        self.G = G

        # create FA
        self.fa = FA(
            data,
            N,
            M,
            K,
            D,
            likelihoods,
            Z_priors,
            W_priors,
            starting_params_fa,
            *args,
            **kwargs,
        )

        # create all CTMs
        self.M_CTM = np.sum(np.array(self.likelihoods) == Likelihood.CTM)
        self.index_CTM = np.where(np.array(self.likelihoods) == Likelihood.CTM)[0]
        self.key_CTM = [
            "M" + str(m) for m in range(M) if self.likelihoods[m] == Likelihood.CTM
        ]

        if starting_params_ctm is None:
            starting_params_ctm = []
            for _ in range(self.M_CTM):
                starting_params_ctm.append({})
        self.ctm_list = dict()
        for m_ctm in range(self.M_CTM):
            self.ctm_list[self.key_CTM[m_ctm]] = CTM(
                data[self.key_CTM[m_ctm]],
                self.N,
                self.L[m_ctm],
                self.G[m_ctm],
                K,
                starting_params_ctm[m_ctm],
            )

            # data is structered use eta minus mu0 values in FA
            self.fa.nodelist_y[self.index_CTM[m_ctm]].data = self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_eta.E_eta_minus_mu0

    def MB(self):
        self.fa.MB()

        for m_ctm in range(self.M_CTM):
            self.ctm_list[self.key_CTM[m_ctm]].MB()

    def update(self):
        self.fa.update()

        for m_ctm in range(self.M_CTM):
            # update parameters of CTM shared with FA
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w = self.fa.nodelist_w[
                self.index_CTM[m_ctm]
            ].E_w.copy()
            self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_w_z.E_w_squared = self.fa.nodelist_w[
                self.index_CTM[m_ctm]
            ].E_w_squared.copy()
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_z = self.fa.node_z.E_z.copy()
            self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_w_z.E_z_squared = self.fa.node_z.E_z_squared
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w_z = self.fa.nodelist_w[
                self.index_CTM[m_ctm]
            ].E_w_z.copy()
            self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_w_z.E_w_z_squared = self.fa.nodelist_w[
                self.index_CTM[m_ctm]
            ].E_w_z_squared.copy()

            # update CTM parameters
            self.ctm_list[self.key_CTM[m_ctm]].update()

            # update FA parameters based on CTM
            self.fa.nodelist_y[self.index_CTM[m_ctm]].data = self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_eta.E_eta_minus_mu0.copy()
            self.fa.nodelist_tau[self.index_CTM[m_ctm]].Sigma0_inv = self.ctm_list[
                self.key_CTM[m_ctm]
            ].node_Sigma0.Sigma0.copy()

    def ELBO(self):
        self.fa.ELBO()

        for m_ctm in range(self.M_CTM):
            self.ctm_list[self.key_CTM[m_ctm]].ELBO()

    def get_elbo(self):
        return self.fa.elbo + np.sum(
            [self.ctm_list[self.key_CTM[m_ctm]].elbo for m_ctm in range(self.M_CTM)]
        )
