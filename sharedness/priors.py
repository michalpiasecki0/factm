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
    slab_var: float | np.ndarray = 1.0  # one value, or one per view
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
        """Fixed slab: alpha_mk = 1 / slab_var exactly (``slab_var`` may be one value
        per view). q(alpha) is a point mass, so its own ELBO term is dropped
        (``get_additional_nodes_to_update``)."""
        sv = np.atleast_1d(np.asarray(self.state.slab_var, dtype=float))
        a = 1.0 / (sv[self.m] if len(sv) > 1 else sv[0])
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
            # A prior may exclude cohorts from the switch (reference coding).
            free = [p.free_gamma() if hasattr(p, "free_gamma") else p.E_gamma
                    for p in self.priors]
            if self.per_factor:
                self.pis[k].update(free[k].sum(), len(free[k]))
            else:
                self.pis[0].update(sum(f.sum() for f in free), sum(len(f) for f in free))
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


# ---------------------------------------------------------------------------
# Scale-free paired cohort prior (the cohort prior of the final model)
# ---------------------------------------------------------------------------


class ScaleFreeCohortPrior(CohortZPrior):
    """Cohort prior for factor k with a paired, scale-free spike-and-slab shift.

    Model (cohort c, factor k):
        z_nk | c        ~ N(gamma_c delta_c, 1 / tau_c)
        delta_c | tau_c ~ N(0, 1 / (kappa tau_c))      <- shift in units of the
        tau_c           ~ Gamma(a0, b0)                    cohort's own spread
        gamma_c         ~ Bern(pi),  pi from ``CohortPiState``
    Variational family (paired):
        q(tau_c) = Gamma(a_c, b_c), q(gamma_c) = Bern(g_c),
        q(delta_c | gamma_c = 1) = N(mu_c, v_c),  q(delta_c | gamma_c = 0) = N(0, v0_c).
    Optimal updates, with n_c samples and S_c = sum_{n in c} E[z_nk]:
        mu_c = S_c / (kappa + n_c),   v_c = 1 / (E[tau_c] (kappa + n_c)),
        v0_c = 1 / (kappa E[tau_c]),
        logit g_c = E log pi - E log(1 - pi) - 1/2 log(1 + n_c / kappa)
                    + E[tau_c] S_c^2 / (2 (kappa + n_c)),
        a_c = a0 + (n_c + 1) / 2,
        b_c = b0 + 1/2 (sum E[z^2] - 2 g mu S + n g (v + mu^2))
                 + kappa/2 (g (v + mu^2) + (1 - g) v0).
    The Occam term -1/2 log(1 + n_c / kappa) does not depend on the scale of
    the factor. ``reference``: that cohort has no switch (gamma = 0, excluded
    from pi), so the other cohorts' shifts are measured against it.
    """

    def __init__(self, cohorts, k: int, state: "CohortPiState", kappa: float = 1.0,
                 reference=None, a0_tau: float = 1.0, b0_tau: float = 1.0):
        super().__init__(cohorts, pi=0.5, a_lambda=1.0, b_lambda=1.0,
                         a0_tau=a0_tau, b0_tau=b0_tau)
        self.k, self.state, self.kappa = k, state, float(kappa)
        C = self.cohort.C
        self.free = np.ones(C, dtype=bool)
        if reference is not None:
            self.free[list(self.cohort.labels).index(reference)] = False
        self.tau_a = self.a0_tau + (self.cohort.counts + 1) / 2.0
        self.tau_b = np.full(C, self.b0_tau + 1.0)
        self.gamma_prob = np.where(self.free, 0.5, 1e-12)
        self.delta_mu = np.zeros(C)
        self.delta_var = np.ones(C)
        self._update_expectations()
        self.v0 = 1.0 / (self.kappa * self.E_tau)

    # ---- pi bookkeeping (used by CohortPiState) ----
    def free_gamma(self) -> np.ndarray:
        return self.E_gamma[self.free]

    # ---- updates ----
    def _update_cohort_params(self, z_node, k: int) -> None:
        idx, counts = self.cohort.cohort_indices, self.cohort.counts
        C = self.cohort.C
        S = np.bincount(idx, weights=np.asarray(z_node.E_z[:, self.k], float), minlength=C)
        S2 = np.bincount(idx, weights=np.asarray(z_node.E_z_squared[:, self.k], float),
                         minlength=C)
        Et = self.E_tau
        # q(delta | gamma) and q(gamma), given q(tau)
        self.delta_mu = S / (self.kappa + counts)
        self.delta_var = 1.0 / (Et * (self.kappa + counts))
        self.v0 = 1.0 / (self.kappa * Et)
        u = (self.logit_pi - 0.5 * np.log1p(counts / self.kappa)
             + Et * S**2 / (2.0 * (self.kappa + counts)))
        g = 1.0 / (1.0 + np.exp(-np.clip(u, -60.0, 60.0)))
        self.gamma_prob = np.where(self.free, np.clip(g, 1e-12, 1 - 1e-12), 1e-12)
        self._update_expectations()
        # q(tau)
        g, mu, v, v0 = self.E_gamma, self.delta_mu, self.delta_var, self.v0
        self.tau_a = self.a0_tau + (counts + 1) / 2.0
        self.tau_b = np.maximum(
            self.b0_tau
            + 0.5 * (S2 - 2.0 * g * mu * S + counts * g * (v + mu**2))
            + 0.5 * self.kappa * (g * (v + mu**2) + (1 - g) * v0), 1e-20)
        self._update_expectations()
        self.state.update(self.k)

    # ---- ELBO ----
    def compute_elbo_k(self, z_node, k: int) -> float:
        from scipy.special import digamma as psi, gammaln

        idx, counts = self.cohort.cohort_indices, self.cohort.counts
        C = self.cohort.C
        S = np.bincount(idx, weights=np.asarray(z_node.E_z[:, k], float), minlength=C)
        S2 = np.bincount(idx, weights=np.asarray(z_node.E_z_squared[:, k], float),
                         minlength=C)
        slv = np.bincount(idx, weights=np.asarray(log_eps(z_node.vi_var[:, k]), float),
                          minlength=C)
        l2p = np.log(2 * np.pi)
        g, mu, v, v0 = self.E_gamma, self.delta_mu, self.delta_var, self.v0
        Et, Elt = self.E_tau, self.E_log_tau
        # E log p(z | gamma, delta, tau) + H[q(z)]
        resid = S2 - 2 * g * mu * S + counts * g * (v + mu**2)
        log_p_z = 0.5 * np.sum(counts * (Elt - l2p) - Et * resid)
        ent_z = 0.5 * np.sum(counts * (1 + l2p) + slv)
        # E log p(delta | tau) + H[q(delta | gamma)]   (2 pi terms cancel)
        delta = np.sum(
            0.5 * (np.log(self.kappa) + Elt)
            - 0.5 * self.kappa * Et * (g * (v + mu**2) + (1 - g) * v0)
            + 0.5 * (g * log_eps(v) + (1 - g) * log_eps(v0)) + 0.5)
        # E log p(gamma | pi) + H[q(gamma)], free cohorts only
        lp, l1p = self.state.E_logs(self.k)
        gf = g[self.free]
        gam = np.sum(gf * lp + (1 - gf) * l1p) - np.sum(xlogx(gf) + xlogx(1 - gf))
        # E log p(tau) + H[q(tau)]
        a, b, a0, b0 = self.tau_a, self.tau_b, self.a0_tau, self.b0_tau
        tau = np.sum((a0 - 1) * Elt - b0 * Et + a0 * np.log(b0) - gammaln(a0)) + np.sum(
            a - log_eps(b) + gammaln(a) + (1 - a) * psi(a))
        elbo = log_p_z + ent_z + delta + gam + tau
        if self.state.learns and (self.state.per_factor or self.k == 0):
            elbo += self.state.pi_for(self.k).elbo()
        return float(elbo)


# ---------------------------------------------------------------------------
# Frozen view pattern + corrected element spike-and-slab
# ---------------------------------------------------------------------------

from src.w_priors.ARD_SS import ARD_SSWPrior  # noqa: E402  (after the corrected file)


class MaskedSSWPrior(ARD_SSWPrior):
    """Corrected ARD_SS (element spike-and-slab) for one view, with a frozen view
    pattern. If H[m, k] = 0 the loading column is identically zero, and the
    column's element switches s_dk, slab values w_hat_dk, inclusion rate
    theta_mk and ARD precision alpha_mk are *not part of the masked model*:
    they do not touch the likelihood, so they are left out of the ELBO
    (keeping them with s fixed at 0 would cost about -log(D_m + 1) nats per
    masked cell and bias any comparison between patterns against masking).
    Columns with H[m, k] = 1 are ordinary element spike-and-slab columns."""

    def __init__(self, mask_row: np.ndarray):
        self.mask_row = np.asarray(mask_row, dtype=int)

    def create_nodes(self, m, params, starting_params, D, K) -> dict:
        from scipy.special import digamma as psi, gammaln

        nodes = super().create_nodes(m, params, starting_params, D, K)
        s, th, al, row = nodes["s_m_node"], nodes["theta_m_node"], nodes["alpha_m_node"], self.mask_row
        off = row == 0
        s.vi_lambda[:, off] = -50.0
        s.vi_gamma[:, off] = 0.0
        th.vi_a[off], th.vi_b[off] = th.a0, th.b0  # q(theta) = prior: contributes 0
        th.update_params()
        s_update, th_update, al_elbo = s.update_k, th.update_k, al.ELBO

        def s_update_k(k, nominator, denominator, E_alpha, E_log_LR_theta):
            if row[k] == 0:
                s.v0[k] = 1.0 / E_alpha
                s.vi_lambda[:, k] = -50.0
                s.vi_gamma[:, k] = 0.0
            else:
                s_update(k, nominator, denominator, E_alpha, E_log_LR_theta)

        def th_update_k(k):
            if row[k] == 0:
                th.vi_a[k], th.vi_b[k] = th.a0, th.b0
                th.update_params()
            else:
                th_update(k)

        def al_ELBO():
            al_elbo()
            if off.any():  # remove the masked columns' alpha terms
                a, b = al.vi_a[off], al.vi_b[off]
                per_k = (-gammaln(al.a0) + al.a0 * np.log(al.b0)
                         + (al.a0 - 1) * al.E_log_alpha[off] - al.b0 * al.E_alpha[off]
                         + a + gammaln(a) + (1 - a) * psi(a) - np.log(b))
                al.elbo -= float(np.sum(per_k))

        s.update_k, th.update_k, al.ELBO = s_update_k, th_update_k, al_ELBO
        return nodes

    def compute_elbo(self, w_node) -> float:
        elbo = super().compute_elbo(w_node)
        off = self.mask_row == 0
        if not off.any():
            return elbo
        # super() counted, for each masked column and every d: the spike branch of
        # w_hat and E log(1 - theta) (g = 0, so the Bernoulli entropy is 0). Remove them.
        a, v0, th = w_node.alpha_m_node, w_node.s_m_node.v0, w_node.theta_m_node
        spike = 0.5 * a.E_log_alpha - 0.5 * a.E_alpha * v0 + 0.5 * np.log(v0) + 0.5
        per_col = spike + th.E_log_1minustheta
        return float(elbo - w_node.D * np.sum(per_col[off]))


@dataclass
class MaskedSSViewConfig(SimpleViewConfig):
    """SimpleViewConfig for ``MaskedSSWPrior``. ``w_prior`` stays WPrior.ARD_SS so
    that pretraining initialises the slab means as for ARD_SS."""

    mask_row: np.ndarray | None = None

    def create_w_prior(self):
        return MaskedSSWPrior(self.mask_row)
