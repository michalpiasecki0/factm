"""
Building, fitting and probing models for the sharedness experiments.

``fit_model`` covers every arm used in the notebooks:

    w = "NONE" | "ARD" | "switch"        (loadings prior)
    z = "std"  | "cohort" | "cohort_learn_pi"   (factor prior)

``freeze_z_gap`` is the frozen-versus-learned test: hold the factor means at a
rotated copy of the truth, let everything else re-optimise, and record the ELBO.
"""
from __future__ import annotations

import contextlib
import io
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.build_model import FACTModel
from src.enums import Likelihood, WPrior, ZPrior
from src.model_config import CohortPriorConfig, ModelConfig, SimpleViewConfig
from src.views import Views

from .priors import (
    CohortPiState,
    LearnablePiCohortPrior,
    SwitchState,
    SwitchViewConfig,
)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@dataclass
class Fit:
    model: FACTModel
    switch: SwitchState | None
    cohort_pi: CohortPiState | None
    elbo: list[float]
    arm: dict

    @property
    def Z(self) -> np.ndarray:
        return np.asarray(np.ma.getdata(self.model.fa.node_z.vi_mu), dtype=float)

    def W(self, m: int) -> np.ndarray:
        return np.asarray(np.ma.getdata(self.model.fa.nodelist_w[m].E_w), dtype=float)

    @property
    def Ws(self) -> list[np.ndarray]:
        return [self.W(m) for m in range(self.model.fa.M)]

    @property
    def alpha(self) -> np.ndarray | None:
        """E[alpha_mk] as M x K, or None if the W prior has no ARD."""
        a = self.model.fa.nodelist_alpha
        if any(x is None for x in a):
            return None
        return np.vstack([np.asarray(x.E_alpha, dtype=float) for x in a])

    @property
    def E_h(self) -> np.ndarray | None:
        return None if self.switch is None else self.switch.E_h.copy()

    @property
    def E_gamma(self) -> np.ndarray | None:
        ps = self.model.fa.node_z.z_priors
        if not hasattr(ps[0], "E_gamma"):
            return None
        return np.column_stack([p.E_gamma for p in ps])

    @property
    def monotone(self) -> bool:
        d = np.diff(self.elbo)
        return bool(np.all(d > -1e-6 * max(1.0, abs(self.elbo[-1]))))


def build_model(
    views: Views,
    labels: np.ndarray,
    K: int,
    w: str = "ARD",
    z: str = "std",
    seed: int = 0,
    pi_mode: str = "global",
    pi_fixed: float = 0.5,
    frozen_mask: np.ndarray | None = None,
    cohort_pi: float = 0.5,
    cohort_pi_mode: str = "global",
    cohort_paired: bool = False,
    beta_a0: float = 1.0,
    beta_b0: float = 1.0,
    slab: str = "ard",
    slab_var: float = 1.0,
    h_init: float = 0.99,
    cohort_fixed_lambda: float | None = None,
) -> tuple[FACTModel, SwitchState | None, CohortPiState | None]:
    M = views.num_simple
    switch = None
    if w == "switch":
        switch = SwitchState(M=M, K=K, pi_mode=pi_mode, pi_fixed=pi_fixed,
                             a0=beta_a0, b0=beta_b0, frozen_mask=frozen_mask,
                             slab=slab, slab_var=slab_var, h_init=h_init)
        vcfg = [SwitchViewConfig(likelihood=Likelihood.NORMAL, w_prior=WPrior.ARD,
                                 state=switch) for _ in range(M)]
    else:
        enum = {"NONE": WPrior.NONE, "ARD": WPrior.ARD}[w]
        vcfg = [SimpleViewConfig(likelihood=Likelihood.NORMAL, w_prior=enum)
                for _ in range(M)]

    z_enum = ZPrior.STD_NORMAL if z == "std" else ZPrior.COHORT
    hp = CohortPriorConfig(pi=cohort_pi)
    cfg = ModelConfig(simple_view_configs=vcfg, structured_view_configs=[],
                      z_priors=[z_enum] * K, cohort_prior_config=hp)
    with quiet():
        model = FACTModel(views=views, K=K, model_config=cfg, seed=seed)

    cps = None
    if z == "cohort_learn_pi":
        C = len(np.unique(labels))
        cps = CohortPiState(K=K, C=C, mode=cohort_pi_mode, pi_fixed=cohort_pi,
                            a0=beta_a0, b0=beta_b0)
        priors = [
            LearnablePiCohortPrior(labels, k=k, state=cps, paired=cohort_paired,
                                   fixed_lambda=cohort_fixed_lambda,
                                   a_lambda=hp.a_lambda,
                                   b_lambda=hp.b_lambda, a0_tau=hp.a0_tau,
                                   b0_tau=hp.b0_tau)
            for k in range(K)
        ]
        cps.priors = priors
        cps.refresh_logits()
        model.fa.node_z.z_priors = priors
    return model, switch, cps


def fit_model(ds, K: int, max_iter: int = 300, elbo_tres: float = 1e-3,
              seed: int = 0, **arm) -> Fit:
    """Fit one arm on a ``Dataset``. ``arm`` goes to ``build_model``."""
    model, switch, cps = build_model(ds.views, ds.labels, K, seed=seed, **arm)
    np.random.seed(seed)
    with quiet():
        model.fit(max_iter=max_iter, elbo_tres=elbo_tres, pretrain=True)
    return Fit(model, switch, cps, list(model.elbo_sequence), dict(arm, K=K, seed=seed))


# ---------------------------------------------------------------------------
# Frozen-versus-learned: ELBO with the factor means held at a rotated truth
# ---------------------------------------------------------------------------


def givens(K: int, i: int, j: int, theta: float) -> np.ndarray:
    G = np.eye(K)
    c, s = np.cos(theta), np.sin(theta)
    G[i, i] = G[j, j] = c
    G[i, j], G[j, i] = -s, s
    return G


def _refresh(model: FACTModel) -> None:
    fa = model.fa
    fa.node_z.update_params()
    for w in fa.nodelist_w:
        w.update_params()
        w.update_params_z()
    for t in fa.nodelist_tau:
        t.update_params_w_z()


def freeze_z_elbo(ds, Zbar: np.ndarray, n_iter: int = 150, z_var: float = 1e-2,
                  seed: int = 0, **arm) -> dict:
    """ELBO when the factor means are fixed at ``Zbar`` (N x K) and every other
    variational quantity (loadings, switches, ARD, noise, pi) is re-optimised.
    The factor prior is N(0, I), so an orthogonal rotation of Zbar leaves the
    factor-prior part of the ELBO unchanged; any gap comes from the loadings
    side alone."""
    K = Zbar.shape[1]
    model, switch, _ = build_model(ds.views, ds.labels, K, z="std", seed=seed, **arm)
    np.random.seed(seed)
    with quiet():
        model.MB()
        z = model.fa.node_z
        z.vi_mu = np.array(Zbar, dtype=float)
        z.vi_var = np.full_like(z.vi_mu, z_var)
        for m, Y in enumerate(ds.Ys):  # least-squares start for the loadings
            Wls = np.linalg.lstsq(Zbar, Y, rcond=None)[0].T
            if model.fa.nodelist_hat_w[m] is not None:
                model.fa.nodelist_hat_w[m].vi_mu = Wls.copy()
                model.fa.nodelist_hat_w[m].vi_var = np.full_like(Wls, 1e-2)
            else:
                model.fa.nodelist_w_not_sparse[m].vi_mu = Wls.copy()
                model.fa.nodelist_w_not_sparse[m].vi_var = np.full_like(Wls, 1e-2)
        _refresh(model)
        trace = []
        for _ in range(n_iter):
            model.update(update_z=False)
            model.ELBO()
            trace.append(model.get_elbo())
    return dict(elbo=trace[-1], trace=trace,
                E_h=None if switch is None else switch.E_h.copy())


def _freeze_job(args) -> tuple:
    key, ds, Zbar, n_iter, arm = args
    out = freeze_z_elbo(ds, Zbar, n_iter=n_iter, **arm)
    return key, out


def run_parallel(fn: Callable, jobs: list, max_workers: int | None = None) -> list:
    """Run ``fn`` over ``jobs`` in a process pool (fn must be importable)."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, jobs))


def _fit_job(args) -> tuple:
    key, ds, K, kw = args
    f = fit_model(ds, K, **kw)
    # Return plain arrays: models do not pickle cheaply.
    return key, dict(
        Z=f.Z, Ws=f.Ws, alpha=f.alpha, E_h=f.E_h, E_gamma=f.E_gamma,
        elbo=f.elbo, monotone=f.monotone,
        switch_pi=None if f.switch is None else f.switch.pi_summary(),
        cohort_pi=None if f.cohort_pi is None else dict(
            mean=[p.mean for p in f.cohort_pi.pis],
            sd=[p.sd for p in f.cohort_pi.pis]),
        n_iter=len(f.elbo),
    )


def fit_from_state(ds, K: int, Z0: np.ndarray, W0s: list[np.ndarray],
                   max_iter: int = 300, elbo_tres: float = 1e-3, seed: int = 0,
                   z_var: np.ndarray | float = 1e-2, w_var: list | float = 1e-2,
                   alpha_from=None, tau_from=None, **arm) -> Fit:
    """Start a model at given factor and loading means, then run CAVI.

    Used for (a) the warm-started switch (start = a converged ARD fit) and
    (b) the truth start of Q2b (start = the generating Z and W)."""
    model, switch, _ = build_model(ds.views, ds.labels, K, z="std", seed=seed, **arm)
    np.random.seed(seed)
    with quiet():
        model.MB()
        z = model.fa.node_z
        z.vi_mu = np.array(Z0, dtype=float)
        z.vi_var = (np.full_like(z.vi_mu, z_var) if np.isscalar(z_var)
                    else np.array(z_var, dtype=float))
        for m in range(model.fa.M):
            node = (model.fa.nodelist_hat_w[m] if model.fa.nodelist_hat_w[m] is not None
                    else model.fa.nodelist_w_not_sparse[m])
            node.vi_mu = np.array(W0s[m], dtype=float)
            node.vi_var = (np.full_like(node.vi_mu, w_var) if np.isscalar(w_var)
                           else np.array(w_var[m], dtype=float))
            ard_slab = switch is None or switch.slab == "ard"
            if (alpha_from is not None and model.fa.nodelist_alpha[m] is not None
                    and ard_slab):
                a_src, a_dst = alpha_from[m], model.fa.nodelist_alpha[m]
                a_dst.vi_a, a_dst.vi_b = a_src.vi_a.copy(), a_src.vi_b.copy()
                a_dst.update_all_params()
            if tau_from is not None:
                t_src, t_dst = tau_from[m], model.fa.nodelist_tau[m]
                t_dst.vi_a, t_dst.vi_b = np.copy(t_src.vi_a), np.copy(t_src.vi_b)
                t_dst.update_all_params()
        _refresh(model)
        elbo = []
        for _ in range(max_iter):
            model.update()
            model.ELBO()
            elbo.append(model.get_elbo())
            if len(elbo) > 1 and elbo[-1] - elbo[-2] < elbo_tres:
                break
    return Fit(model, switch, None, elbo, dict(arm, K=K, seed=seed))


def fit_switch_warm(ds, K: int, ard_iter: int = 300, max_iter: int = 300,
                    elbo_tres: float = 1e-3, seed: int = 0, **arm) -> Fit:
    """Fit ARD first, then start the view-switch model from the ARD solution.
    The switch can only turn a column off if the factors are already near a
    sparse rotation; a dense start rarely gets there."""
    ard = fit_model(ds, K, max_iter=ard_iter, elbo_tres=elbo_tres, seed=seed, w="ARD")
    fa = ard.model.fa
    f = fit_from_state(
        ds, K, ard.Z, [np.asarray(h.vi_mu) for h in fa.nodelist_hat_w],
        max_iter=max_iter, elbo_tres=elbo_tres, seed=seed,
        z_var=np.asarray(np.ma.getdata(fa.node_z.vi_var)),
        w_var=[np.asarray(h.vi_var) for h in fa.nodelist_hat_w],
        alpha_from=fa.nodelist_alpha, tau_from=fa.nodelist_tau, w="switch", **arm)
    f.arm["warm"] = "ARD"
    return f
