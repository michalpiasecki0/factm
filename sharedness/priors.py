"""
Priors for the sharedness experiments.

Two additions, both plugged in without touching ``src/``:

* ``ViewSwitchWPrior`` -- a view-block spike-and-slab on the loadings. For each
  view m and factor k one switch h_mk turns the whole loading column
  W^(m)_{.k} on or off:

      h_mk ~ Bern(pi),  V^(m)_{dk} | alpha_mk ~ N(0, 1/alpha_mk),
      W^(m)_{.k} = h_mk V^(m)_{.k}.

  The variational family is q(h, V) = q(h) q(V | h), with q(V | h=1) Gaussian
  per feature and q(V | h=0) = N(0, v0_mk) (v0 is optimal at 1/E[alpha]).
  Optimal switch log-odds, with slab precision
  Lambda_d = sum_n E[z_nk^2] E[tau_d] + E[alpha_mk] and the usual
  residual term nom_d:

      logit E[h_mk] = E[log pi] - E[log(1-pi)]
                      + sum_d [ 1/2 log(E[alpha_mk] / Lambda_d)
                                + nom_d^2 / (2 Lambda_d) ].

  Note the sign of log Lambda: it is -1/2, which is *not* what
  src/w_priors/ARD_SS.py uses.

  The switch probability pi is either fixed, learned once for the whole
  model ("global", M*K Bernoulli draws) or learned per factor ("per_factor",
  M draws each), always with a Beta(a0, b0) prior. A boolean ``frozen_mask``
  (M x K) instead fixes h to a prescribed 0/1 pattern.

* ``LearnablePiCohortPrior`` -- CohortFACTM's cohort prior with pi learned
  under a Beta prior, either shared by all factors or per factor.

Hyperparameters travel through a small shared state object because the repo
creates one W prior per view and one Z prior per factor, with no cross-view
or cross-factor handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import betaln, digamma

from src.model_config import SimpleViewConfig
from src.starting_params import starting_params_hat_w_m
from src.utils import log_eps, xlogx
from src.w_priors.ARD import nodeFA_alpha_m, nodeFA_hat_w_m
from src.w_priors.base import WPriorBase
from src.z_priors.Cohort import CohortZPrior

H_CLIP = 1e-10  # keeps E[h] strictly inside (0, 1) for the learned switch


# ---------------------------------------------------------------------------
# Beta-distributed switch probability, shared between prior instances
# ---------------------------------------------------------------------------


@dataclass
class BetaPi:
    """q(pi) = Beta(a, b) for one pi. ``n_trials`` counts its Bernoulli draws."""

    a0: float = 1.0
    b0: float = 1.0
    a: float = 1.0
    b: float = 1.0

    def update(self, successes: float, n_trials: float) -> None:
        self.a = self.a0 + successes
        self.b = self.b0 + n_trials - successes

    @property
    def E_log_pi(self) -> float:
        return float(digamma(self.a) - digamma(self.a + self.b))

    @property
    def E_log_1mpi(self) -> float:
        return float(digamma(self.b) - digamma(self.a + self.b))

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def sd(self) -> float:
        s = self.a + self.b
        return float(np.sqrt(self.a * self.b / (s * s * (s + 1))))

    def elbo(self) -> float:
        """E[log p(pi)] + H[q(pi)]."""
        log_p = (
            (self.a0 - 1) * self.E_log_pi
            + (self.b0 - 1) * self.E_log_1mpi
            - betaln(self.a0, self.b0)
        )
        entropy = (
            betaln(self.a, self.b)
            - (self.a - 1) * digamma(self.a)
            - (self.b - 1) * digamma(self.b)
            + (self.a + self.b - 2) * digamma(self.a + self.b)
        )
        return float(log_p + entropy)


# ---------------------------------------------------------------------------
# View-block switch on the loadings
# ---------------------------------------------------------------------------


@dataclass
class SwitchState:
    """State shared by the per-view ``ViewSwitchWPrior`` objects of one model.

    pi_mode: "fixed" (pi = pi_fixed), "global" (one Beta pi) or
    "per_factor" (one Beta pi per factor). ``frozen_mask`` (M x K, 0/1)
    overrides everything and fixes h.
    """

    M: int
    K: int
    pi_mode: str = "global"
    pi_fixed: float = 0.5
    a0: float = 1.0
    b0: float = 1.0
    frozen_mask: np.ndarray | None = None
    h_init: float = 0.99
    slab: str = "ard"  # "ard": learned alpha_mk; "fixed": alpha = 1 / slab_var
    slab_var: float = 1.0
    E_h: np.ndarray = field(init=False)
    pis: list[BetaPi] = field(init=False)

    def __post_init__(self) -> None:
        if self.pi_mode not in ("fixed", "global", "per_factor"):
            raise ValueError(f"unknown pi_mode {self.pi_mode!r}")
        if self.slab not in ("ard", "fixed"):
            raise ValueError(f"unknown slab {self.slab!r}")
        if self.frozen_mask is not None:
            mask = np.asarray(self.frozen_mask, dtype=float)
            if mask.shape != (self.M, self.K):
                raise ValueError("frozen_mask must be M x K")
            self.E_h = mask.copy()
        else:
            self.E_h = np.full((self.M, self.K), self.h_init)
        n_pi = self.K if self.pi_mode == "per_factor" else 1
        self.pis = [BetaPi(self.a0, self.b0, self.a0, self.b0) for _ in range(n_pi)]

    @property
    def frozen(self) -> bool:
        return self.frozen_mask is not None

    @property
    def learns_pi(self) -> bool:
        return not self.frozen and self.pi_mode != "fixed"

    def log_odds_prior(self, k: int) -> float:
        if self.pi_mode == "fixed":
            p = float(np.clip(self.pi_fixed, 1e-12, 1 - 1e-12))
            return float(np.log(p) - np.log(1 - p))
        pi = self.pis[k] if self.pi_mode == "per_factor" else self.pis[0]
        return pi.E_log_pi - pi.E_log_1mpi

    def E_logs(self, k: int) -> tuple[float, float]:
        if self.pi_mode == "fixed":
            p = float(np.clip(self.pi_fixed, 1e-12, 1 - 1e-12))
            return float(np.log(p)), float(np.log(1 - p))
        pi = self.pis[k] if self.pi_mode == "per_factor" else self.pis[0]
        return pi.E_log_pi, pi.E_log_1mpi

    def update_pi(self, k: int) -> None:
        if not self.learns_pi:
            return
        if self.pi_mode == "per_factor":
            self.pis[k].update(self.E_h[:, k].sum(), self.M)
        else:
            self.pis[0].update(self.E_h.sum(), self.M * self.K)

    def pi_summary(self) -> dict:
        if self.pi_mode == "fixed" or self.frozen:
            return {"mode": "fixed" if not self.frozen else "frozen",
                    "mean": [self.pi_fixed], "sd": [0.0]}
        return {"mode": self.pi_mode,
                "mean": [p.mean for p in self.pis],
                "sd": [p.sd for p in self.pis]}


class ViewSwitchWPrior(WPriorBase):
    """ARD slab plus one on/off switch per (view, factor). See module docstring."""

    def __init__(self, state: SwitchState):
        self.state = state
        self.m: int | None = None
        self.v0: np.ndarray | None = None

    def create_nodes(self, m, params, starting_params, D, K) -> dict:
        self.m = m
        w_mu, w_var = starting_params_hat_w_m(starting_params, m, D, K)
        node_hat = nodeFA_hat_w_m(w_mu, w_var, D, params=params)
        node_alpha = nodeFA_alpha_m(1e-14, 1e-14, D, params=params)
        if self.state.slab == "fixed":
            self._pin_alpha(node_alpha)
        self.v0 = 1.0 / node_alpha.E_alpha
        return {
            "w_m_node_not_sparse": None,
            "hat_w_m_node": node_hat,
            "s_m_node": None,
            "alpha_m_node": node_alpha,
            "theta_m_node": None,
        }

    def _pin_alpha(self, node) -> None:
        """Fixed slab: alpha_mk = 1 / slab_var exactly. q(alpha) is a point mass,
        so its own ELBO term is dropped (``get_additional_nodes_to_update``)."""
        a = 1.0 / self.state.slab_var
        node.E_alpha = np.full(self.state.K, a)
        node.E_inv_alpha = np.full(self.state.K, 1.0 / a)
        node.E_log_alpha = np.full(self.state.K, np.log(a))
        node.update_k = lambda k: None

    def update_w_k(self, w_node, k, z_node, y_node, tau_node):
        if w_node.is_ctm:
            raise NotImplementedError("ViewSwitchWPrior is for simple views only")
        st, m = self.state, self.m

        # Residual term and slab precision, exactly as in ARDWPrior.
        second_tmp = np.dot(w_node.E_w, z_node.E_z.T * z_node.E_z[:, k]).T
        second = second_tmp - np.outer(z_node.E_z[:, k] ** 2, w_node.E_w[:, k])
        nom = np.ma.sum(
            tau_node.E_tau * ((y_node.data.T * z_node.E_z[:, k]).T - second), axis=0
        )
        E_alpha = w_node.alpha_m_node.E_alpha[k]
        lam = np.ma.dot(z_node.E_z_squared[:, k], tau_node.E_tau) + E_alpha
        nom = np.ma.getdata(nom)
        lam = np.ma.getdata(lam)

        # q(V | h=1) and q(V | h=0).
        w_node.hat_w_m_node.update_k(k, nom, lam)
        self.v0[k] = 1.0 / E_alpha

        # Switch.
        if not st.frozen:
            u = st.log_odds_prior(k) + np.sum(
                0.5 * (log_eps(E_alpha) - log_eps(lam)) + nom**2 / (2.0 * lam)
            )
            u = float(np.clip(u, -50.0, 50.0))
            st.E_h[m, k] = np.clip(1.0 / (1.0 + np.exp(-u)), H_CLIP, 1 - H_CLIP)
            st.update_pi(k)

        w_node.update_params()
        w_node.update_params_z()
        w_node.alpha_m_node.update_k(k)

    def update_params(self, w_node) -> None:
        g = self.state.E_h[self.m]  # (K,)
        mu, var = w_node.hat_w_m_node.vi_mu, w_node.hat_w_m_node.vi_var
        w_node.E_w = g * mu
        w_node.E_w_squared = g * (var + mu**2)
        w_node.E_hat_w_squared = w_node.E_w_squared + (1 - g) * self.v0

    def compute_elbo(self, w_node) -> float:
        st, m = self.state, self.m
        g = st.E_h[m]
        a = w_node.alpha_m_node
        mu, var = w_node.hat_w_m_node.vi_mu, w_node.hat_w_m_node.vi_var
        # Slab part, weighted by g: E log p(V|alpha) + H[q(V|h=1)] per element.
        slab = (
            0.5 * a.E_log_alpha
            - 0.5 * a.E_alpha * (var + mu**2)
            + 0.5 * log_eps(var)
            + 0.5
        )
        # Spike part (V | h=0 ~ N(0, v0)), weighted by 1 - g.
        spike = (
            0.5 * a.E_log_alpha
            - 0.5 * a.E_alpha * self.v0
            + 0.5 * log_eps(self.v0)
            + 0.5
        )
        elbo = float(np.sum(g * slab) + w_node.D * np.sum((1 - g) * spike))
        if not st.frozen:
            for k in range(st.K):
                lp, l1p = st.E_logs(k)
                elbo += g[k] * lp + (1 - g[k]) * l1p
                elbo -= xlogx(g[k]) + xlogx(1 - g[k])
            if st.learns_pi and m == 0:  # Beta terms counted once per model
                elbo += sum(p.elbo() for p in st.pis)
        return elbo

    def get_additional_nodes_to_update(self) -> list:
        return ["alpha"] if self.state.slab == "ard" else []


@dataclass
class SwitchViewConfig(SimpleViewConfig):
    """A SimpleViewConfig whose W prior is a ``ViewSwitchWPrior``.

    ``w_prior`` stays ``WPrior.ARD`` so that ``FACTModel.pretrain`` initialises
    the slab means exactly as for ARD (same ``hat_w`` node).
    """

    state: SwitchState | None = None

    def create_w_prior(self):
        return ViewSwitchWPrior(self.state)


# ---------------------------------------------------------------------------
# Cohort prior: learned pi, and an optional paired gamma-delta factorisation
# ---------------------------------------------------------------------------


class CohortPiState:
    """pi for the K cohort priors of one model: fixed, one shared Beta, or one
    Beta per factor."""

    def __init__(self, K: int, C: int, mode: str = "global", pi_fixed: float = 0.5,
                 a0: float = 1.0, b0: float = 1.0):
        if mode not in ("fixed", "global", "per_factor"):
            raise ValueError(f"unknown pi mode {mode!r}")
        self.K, self.C, self.mode, self.pi_fixed = K, C, mode, pi_fixed
        n = K if mode == "per_factor" else 1
        self.pis = [BetaPi(a0, b0, a0, b0) for _ in range(n)]
        self.priors: list[LearnablePiCohortPrior] = []

    @property
    def learns(self) -> bool:
        return self.mode != "fixed"

    @property
    def per_factor(self) -> bool:
        return self.mode == "per_factor"

    def pi_for(self, k: int) -> BetaPi:
        return self.pis[k] if self.per_factor else self.pis[0]

    def E_logs(self, k: int) -> tuple[float, float]:
        if not self.learns:
            p = float(np.clip(self.pi_fixed, 1e-12, 1 - 1e-12))
            return float(np.log(p)), float(np.log(1 - p))
        pi = self.pi_for(k)
        return pi.E_log_pi, pi.E_log_1mpi

    def refresh_logits(self) -> None:
        for j, p in enumerate(self.priors):
            lp, l1p = self.E_logs(j)
            p.logit_pi = lp - l1p
            p.pi = float(np.exp(lp) / (np.exp(lp) + np.exp(l1p)))

    def update(self, k: int) -> None:
        if self.learns:
            if self.per_factor:
                self.pis[k].update(self.priors[k].E_gamma.sum(), self.C)
            else:
                total = sum(p.E_gamma.sum() for p in self.priors)
                self.pis[0].update(total, self.C * self.K)
        self.refresh_logits()


class LearnablePiCohortPrior(CohortZPrior):
    """CohortFACTM's cohort prior with two optional changes.

    * pi from ``CohortPiState`` (fixed or learned under a Beta prior).
    * ``paired=True``: q(gamma, delta) = q(gamma) q(delta | gamma) with
      q(delta | gamma=0) = N(0, v0), v0 optimal at 1/E[lambda]. Then

          logit E[gamma_c] = E[log pi] - E[log(1-pi)]
                             + 1/2 log(E[lambda] / P_c) + (E[tau_c] S_c)^2 / (2 P_c),
          P_c = E[lambda] + E[tau_c] n_c,   S_c = sum_{n in c} E[z_nk],

      which contains the Occam term -1/2 log(1 + E[tau_c] n_c / E[lambda]).
      The original q(gamma) q(delta) has no such term: with delta at its
      optimum, switching gamma on costs nothing beyond the prior, so a cohort
      with no shift sits at E[gamma] ~ pi rather than near 0.
    * ``fixed_lambda``: pin E[lambda] (shift scale cannot shrink to zero).
    """

    def __init__(self, cohorts, k: int, state: CohortPiState, paired: bool = False,
                 fixed_lambda: float | None = None, **hp):
        super().__init__(cohorts, pi=0.5, **hp)
        self.k = k
        self.state = state
        self.paired = paired
        self.fixed_lambda = fixed_lambda
        self._pin_lambda()
        self.v0 = np.full(self.cohort.C, 1.0 / self.E_lambda)

    def _pin_lambda(self) -> None:
        """q(lambda) concentrated at fixed_lambda; its ELBO terms are then
        constant across iterations, so monotonicity is unaffected."""
        if self.fixed_lambda is None:
            return
        self.lambda_a = 1e8
        self.lambda_b = 1e8 / self.fixed_lambda
        self._update_expectations()

    def _update_cohort_params(self, z_node, k: int) -> None:
        if not self.paired:
            super()._update_cohort_params(z_node, k)
        else:
            self._update_paired(z_node)
        self._pin_lambda()
        self.state.update(self.k)

    def _sums(self, z_node):
        idx, C = self.cohort.cohort_indices, self.cohort.C
        S = np.bincount(idx, weights=np.asarray(z_node.E_z[:, self.k], dtype=float),
                        minlength=C)
        S2 = np.bincount(idx, weights=np.asarray(z_node.E_z_squared[:, self.k],
                                                 dtype=float), minlength=C)
        return S, S2

    def _update_gamma_paired(self, S) -> None:
        """q(delta | gamma=1), v0 and E[gamma] given current tau and lambda."""
        counts = self.cohort.counts
        prec = self.E_lambda + self.E_tau * counts
        self.delta_var = 1.0 / np.maximum(prec, 1e-20)
        self.delta_mu = self.delta_var * self.E_tau * S
        self.v0 = np.full(self.cohort.C, 1.0 / self.E_lambda)
        u = (self.logit_pi
             + 0.5 * (log_eps(self.E_lambda) - log_eps(prec))
             + (self.E_tau * S) ** 2 / (2.0 * prec))
        u = np.clip(u, -60.0, 60.0)
        self.gamma_prob = np.clip(1.0 / (1.0 + np.exp(-u)), 1e-20, 1 - 1e-20)
        self._update_expectations()

    def _update_paired(self, z_node) -> None:
        counts, C = self.cohort.counts, self.cohort.C
        S, S2 = self._sums(z_node)
        self._update_gamma_paired(S)
        # tau: identical to the original (it only sees E[gamma delta], E[gamma delta^2]).
        self.tau_a = self.a0_tau + counts / 2.0
        self.tau_b = np.maximum(self.b0_tau + 0.5 * (
            S2 - 2.0 * self.E_gamma * self.E_delta * S
            + counts * self.E_gamma * self.E_delta_squared), 1e-20)
        self._update_expectations()
        # lambda sees E[delta^2] = g (v + mu^2) + (1 - g) v0.
        Ed2 = self.E_gamma * self.E_delta_squared + (1 - self.E_gamma) * self.v0
        self.lambda_a = self.a_lambda + C / 2.0
        self.lambda_b = max(self.b_lambda + 0.5 * float(np.sum(Ed2)), 1e-20)
        self._update_expectations()

    def compute_elbo_k(self, z_node, k: int) -> float:
        base = super().compute_elbo_k(z_node, k)
        # 1) log p(gamma | pi): replace the fixed-pi term by its expectation.
        fixed = np.sum(self.E_gamma * np.log(self.pi)
                       + (1.0 - self.E_gamma) * np.log(1.0 - self.pi))
        lp, l1p = self.state.E_logs(self.k)
        elbo = base - fixed + np.sum(self.E_gamma * lp + (1.0 - self.E_gamma) * l1p)
        # 2) paired: replace the delta prior + entropy terms.
        if self.paired:
            log_2pi = np.log(2.0 * np.pi)
            orig = (0.5 * np.sum(self.E_log_lambda - log_2pi
                                 - self.E_lambda * self.E_delta_squared)
                    + 0.5 * np.sum(1.0 + log_2pi + log_eps(self.delta_var)))
            g = self.E_gamma
            slab = (0.5 * self.E_log_lambda - 0.5 * self.E_lambda * self.E_delta_squared
                    + 0.5 * log_eps(self.delta_var) + 0.5)
            spike = (0.5 * self.E_log_lambda - 0.5 * self.E_lambda * self.v0
                     + 0.5 * log_eps(self.v0) + 0.5)
            elbo += np.sum(g * slab + (1 - g) * spike) - orig
        # 3) Beta terms, counted once per Beta.
        if self.state.learns and (self.state.per_factor or self.k == 0):
            elbo += self.state.pi_for(self.k).elbo()
        return float(elbo)
