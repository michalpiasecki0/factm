"""
The improved fitting pipeline (the "after" of notebook 04).

1. Initialisation aware of shared/private structure: generalised CCA (MAXVAR).
2. Multi-start: several random rotations of that start, keep the best ELBO.
   Justified by Q2b: the ELBO ranks the true factors above the mixed ones.
3. Convergence by relative tolerance instead of a fixed iteration cap.
4. Two stages: learn the view pattern from the variance shares of the best
   fit, freeze it, refit with element-wise sparsity (corrected ARD_SS); then
   flip borderline cells if that raises the ELBO.
"""
from __future__ import annotations

import numpy as np

from . import metrics as mt
from .models import Fit, fit_from_state


# ---------------------------------------------------------------------------
# Initialisations
# ---------------------------------------------------------------------------


def _ls_loadings(Z: np.ndarray, Ys: list[np.ndarray]) -> list[np.ndarray]:
    return [np.linalg.lstsq(Z, Y, rcond=None)[0].T for Y in Ys]


def pca_init(Ys: list[np.ndarray], K: int) -> np.ndarray:
    """Top-K principal components of the stacked views, scaled to unit variance."""
    U, _, _ = np.linalg.svd(np.hstack(Ys), full_matrices=False)
    return U[:, :K] * np.sqrt(len(U))


def gcca_init(Ys: list[np.ndarray], K: int, shared_thr: float = 1.5,
              r: int | None = None) -> tuple[np.ndarray, int]:
    """Generalised CCA (MAXVAR) start.

    Whiten each view by its top r principal directions (U_m, N x r), stack
    G = [U_1 ... U_M] and take the SVD G = A diag(sigma) B^T. A direction used by
    v views has sigma^2 ~ v, a private one sigma^2 ~ 1. Directions with
    sigma^2 >= shared_thr become the shared/partially shared starting factors;
    the remaining factors are the leading residual principal components of the
    individual views (largest residual variance first).
    Returns (Z0 with unit-variance columns, number of shared directions).
    """
    N = Ys[0].shape[0]
    r = r or K
    Us = []
    for Y in Ys:
        U, s, _ = np.linalg.svd(Y, full_matrices=False)
        keep = min(r, int((s > 1e-8 * s[0]).sum()))
        Us.append(U[:, :keep])
    A, sig, _ = np.linalg.svd(np.hstack(Us), full_matrices=False)
    n_shared = int(min(K, (sig**2 >= shared_thr).sum()))
    cols = [A[:, j] for j in range(n_shared)]
    if n_shared:
        Q = A[:, :n_shared]
        resid = [Y - Q @ (Q.T @ Y) for Y in Ys]
    else:
        resid = list(Ys)
    cands = []  # (variance, direction) of each view's residual components
    for R in resid:
        U, s, _ = np.linalg.svd(R, full_matrices=False)
        cands += [(s[j] ** 2, U[:, j]) for j in range(min(K, len(s)))]
    cands.sort(key=lambda t: -t[0])
    for _, u in cands:
        if len(cols) >= K:
            break
        # keep the start well-conditioned: skip near-duplicates
        if cols:
            Q, _ = np.linalg.qr(np.column_stack(cols))
            u = u - Q @ (Q.T @ u)
            if np.linalg.norm(u) < 0.3:
                continue
        cols.append(u / np.linalg.norm(u))
    Z0 = np.column_stack(cols)
    Z0 = (Z0 - Z0.mean(0)) / Z0.std(0)
    return Z0, n_shared


def random_rotation(K: int, rng) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(K, K)))
    return Q * np.sign(np.diag(R))


# ---------------------------------------------------------------------------
# Multi-start
# ---------------------------------------------------------------------------


def fit_best(ds, K: int, n_starts: int = 8, init: str = "gcca", max_iter: int = 1500,
             rel_tol: float = 1e-7, seed: int = 0, return_all: bool = False, **arm):
    """Fit from ``n_starts`` starts (start 0 = the initialisation itself, the rest
    = random rotations of it) and keep the fit with the highest ELBO."""
    rng = np.random.default_rng(seed)
    Z0 = gcca_init(ds.Ys, K)[0] if init == "gcca" else pca_init(ds.Ys, K)
    fits = []
    for s in range(n_starts):
        Zs = Z0 if s == 0 else Z0 @ random_rotation(K, rng)
        f = fit_from_state(ds, K, Zs, _ls_loadings(Zs, ds.Ys), max_iter=max_iter,
                           rel_tol=rel_tol, seed=seed * 1000 + s, **arm)
        fits.append(f)
    best = max(fits, key=lambda f: f.elbo[-1])
    best.arm.update(n_starts=n_starts, init=init,
                    start_elbos=[f.elbo[-1] for f in fits])
    return (best, fits) if return_all else best


# ---------------------------------------------------------------------------
# Two-stage: learn the view pattern, freeze it, refit
# ---------------------------------------------------------------------------


def learned_pattern(f: Fit, Ys, thr: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Mask H_mk = 1{a_mk >= thr}, a = variance share normalised per factor."""
    a = mt.activity(mt.variance_share(f.Z, f.Ws, Ys))
    return (a >= thr).astype(int), a


def calibrated_slab(f: Fit, Ys, thr: float = 0.1) -> np.ndarray:
    """Per-view slab variance for the view switch: median, over the columns the
    view uses, of the mean squared loading in the stage-1 fit."""
    H, _ = learned_pattern(f, Ys, thr)
    out = []
    for m, W in enumerate(f.Ws):
        cols = np.where(H[m] == 1)[0]
        v = np.median(np.mean(W[:, cols] ** 2, axis=0)) if len(cols) else 1.0
        out.append(float(max(v, 1e-3)))
    return np.array(out)


def fit_two_stage(ds, K: int, n_starts: int = 8, thr: float = 0.1, refine: bool = True,
                  refine_band: tuple[float, float] = (0.05, 0.3), refine_iter: int = 300,
                  seed: int = 0, stage1: Fit | None = None, z: str = "std",
                  max_iter: int = 1500, **zarm) -> dict:
    """Stage 1: best-of-n ARD fit. Stage 2: freeze the learned pattern and refit
    with the corrected element spike-and-slab (masked columns are exact zeros).
    Refinement: for each cell whose activity lies in ``refine_band``, flip it and
    keep the flip if the frozen refit reaches a higher ELBO."""
    if stage1 is None:
        stage1 = fit_best(ds, K, n_starts=n_starts, seed=seed, w="ARD", z=z,
                          max_iter=max_iter, **zarm)
    H, a = learned_pattern(stage1, ds.Ys, thr)
    fa = stage1.model.fa

    def refit(mask, iters):
        return fit_from_state(
            ds, K, stage1.Z, [np.asarray(h.vi_mu) for h in fa.nodelist_hat_w],
            max_iter=iters, rel_tol=1e-7, seed=seed, w="masked_SS", frozen_mask=mask,
            z=z, z_var=np.asarray(np.ma.getdata(fa.node_z.vi_var)),
            w_var=[np.asarray(h.vi_var) for h in fa.nodelist_hat_w],
            alpha_from=fa.nodelist_alpha, tau_from=fa.nodelist_tau, **zarm)

    stage2 = refit(H, max_iter)
    flips = []
    if refine:
        lo, hi = refine_band
        cells = [(m, k) for m in range(H.shape[0]) for k in range(H.shape[1])
                 if lo <= a[m, k] < hi]
        base = refit(H, refine_iter).elbo[-1]
        for m, k in cells:
            H2 = H.copy()
            H2[m, k] = 1 - H2[m, k]
            e = refit(H2, refine_iter).elbo[-1]
            if e > base:
                H, base = H2, e
                flips.append((m, k, int(H2[m, k])))
        if flips:
            stage2 = refit(H, max_iter)
    return dict(stage1=stage1, stage2=stage2, H=H, activity=a, flips=flips)
