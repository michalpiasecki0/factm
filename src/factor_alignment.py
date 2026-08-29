"""Hungarian alignment of latent factors and W stability between model fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .build_model import FACTModel


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; returns 0 when either vector is (near) constant."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r) or np.isinf(r):
        return 0.0
    return r


def hungarian_factor_alignment(
    Z_a: np.ndarray, Z_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match columns of Z_b to Z_a by maximum sum of |correlation|.

    Returns (perm, signs) so Z_b[:, perm] * signs aligns with Z_a.
    """
    k = Z_a.shape[1]
    corr = np.array(
        [[safe_corr(Z_a[:, i], Z_b[:, j]) for j in range(k)] for i in range(k)]
    )
    cost = np.nan_to_num(-np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    row, col = linear_sum_assignment(cost)
    signs = np.sign(corr[row, col])
    signs[signs == 0] = 1
    return col.astype(int), signs.astype(float)


def w_corr_vs_reference_model(
    model_ref: FACTModel, model_other: FACTModel, k: int
) -> list[np.ndarray]:
    """Per-view |corr(W_ref, W_other)| per factor after Hungarian alignment on Z."""
    corrs, _, _ = w_stability_between_models(model_ref, model_other, k)
    return corrs


def w_stability_between_models(
    model_a: FACTModel, model_b: FACTModel, k: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Per-view per-factor |corr(W_a, W_b)| after Hungarian alignment on latent Z.

    Returns (corrs_per_view, perm, signs).
    """
    perm, signs = hungarian_factor_alignment(
        model_a.get_latent_factors(), model_b.get_latent_factors()
    )
    corrs: list[np.ndarray] = []
    for v in range(len(model_a.fa.nodelist_w)):
        w_a = model_a.fa.nodelist_w[v].E_w
        w_b = model_b.fa.nodelist_w[v].E_w
        corrs.append(
            np.array(
                [
                    abs(safe_corr(w_a[:, i], w_b[:, perm[i]] * signs[i]))
                    for i in range(k)
                ]
            )
        )
    return corrs, perm, signs
