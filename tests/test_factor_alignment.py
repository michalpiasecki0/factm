import numpy as np

from src.factor_alignment import hungarian_factor_alignment, safe_corr


def test_safe_corr_constant_vector():
    x = np.ones(20)
    y = np.linspace(0, 1, 20)
    assert safe_corr(x, y) == 0.0


def test_hungarian_no_nan_with_constant_latent_columns():
    z_a = np.random.default_rng(0).normal(size=(15, 4))
    z_b = np.ones((15, 4))
    perm, signs = hungarian_factor_alignment(z_a, z_b)
    assert len(perm) == 4
    assert np.all(np.isfinite(signs))
