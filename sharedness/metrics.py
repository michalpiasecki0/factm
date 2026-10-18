"""
Measuring sharedness.

Three kinds of number, never to be mixed up:

(A) descriptive -- how shared does a factor look in this fit?
    ``variance_share``, ``activity``, ``sharedness_entropy``
(B) recovery -- did the fit find the true pattern? (synthetic data only)
    ``match_factors``, ``pattern_auc``, ``pattern_errors``
(C) identification -- is the pattern pinned at all?
    computed in the notebooks from ``models.freeze_z_elbo`` gaps.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def variance_share(Z: np.ndarray, Ws: list[np.ndarray], Ys: list[np.ndarray]) -> np.ndarray:
    """r[m, k] = ||z_k w_k^(m)T||_F^2 / ||Y_m||_F^2  (M x K, non-negative).

    The fraction of view m's total variance carried by factor k alone. Unlike
    ``FA.variance_explained_per_factor_view`` it never goes negative, so a
    factor that is absent from a view scores ~0 there.
    """
    M, K = len(Ws), Z.shape[1]
    r = np.zeros((M, K))
    for m, (W, Y) in enumerate(zip(Ws, Ys)):
        tot = float(np.sum(Y**2))
        for k in range(K):
            r[m, k] = float(np.sum(Z[:, k] ** 2) * np.sum(W[:, k] ** 2)) / tot
    return r


def activity(score: np.ndarray) -> np.ndarray:
    """Normalise a per-(view, factor) score to [0, 1] within each factor:
    a[m, k] = score[m, k] / max_m score[m, k]. Factors with no signal get 0."""
    mx = score.max(axis=0, keepdims=True)
    return np.divide(score, mx, out=np.zeros_like(score, dtype=float), where=mx > 0)


def ard_score(alpha: np.ndarray) -> np.ndarray:
    """1 / E[alpha_mk]: the expected loading variance of factor k in view m."""
    return 1.0 / alpha


def sharedness_entropy(r: np.ndarray) -> np.ndarray:
    """S_k = -sum_m p_mk log p_mk / log M, p_mk = r_mk / sum_m r_mk.
    1 = spread evenly over all views (shared), 0 = one view only (private).
    NaN for a factor with no variance anywhere."""
    M = r.shape[0]
    tot = r.sum(axis=0)
    out = np.full(r.shape[1], np.nan)
    for k in range(r.shape[1]):
        if tot[k] <= 0:
            continue
        p = r[:, k] / tot[k]
        p = p[p > 0]
        out[k] = float(-(p * np.log(p)).sum() / np.log(M))
    return out


def match_factors(Z_true: np.ndarray, Z_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each true factor, the fitted factor that matches it best (Hungarian
    on |corr|). Works for K_fit >= K_true. Returns (index, |corr|)."""
    Zt = (Z_true - Z_true.mean(0)) / Z_true.std(0)
    sd = Z_fit.std(0)
    sd[sd == 0] = 1.0
    Zf = (Z_fit - Z_fit.mean(0)) / sd
    corr = np.abs(Zt.T @ Zf) / len(Zt)
    rows, cols = linear_sum_assignment(-corr)
    order = np.argsort(rows)
    return cols[order], corr[rows[order], cols[order]]


def pattern_auc(score: np.ndarray, H: np.ndarray) -> float:
    """Threshold-free recovery: P(score of an active cell > score of an
    inactive cell). 1 = perfect, 0.5 = chance. Ties count one half."""
    pos, neg = score[H == 1], score[H == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def pattern_errors(score: np.ndarray, H: np.ndarray, thr: float = 0.5) -> dict:
    """Errors after thresholding at ``thr``. The two kinds are kept apart:
    'leak' = called active where the truth is 0 (private looks shared),
    'miss' = called inactive where the truth is 1."""
    call = (score >= thr).astype(int)
    return dict(
        leak=int(((call == 1) & (H == 0)).sum()),
        miss=int(((call == 0) & (H == 1)).sum()),
        exact_columns=int(np.all(call == H, axis=0).sum()),
        K=int(H.shape[1]),
    )


def decisiveness(score: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """Fraction of cells that are clearly on or off (outside (lo, hi))."""
    return float(np.mean((score <= lo) | (score >= hi)))


def matched_view(score: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Columns of a (M x K_fit) score reordered to the true factors."""
    return score[:, idx]
