"""Checks that the new sharedness updates are correct, not just consistent.

1. ELBO never decreases over CAVI sweeps (for every new arm).
2. Stationarity: each closed-form update lands at a maximum of the ELBO in
   its own coordinate. If a derived update were wrong, nudging the variable
   away from it would *raise* the ELBO for some nudge.
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pytest

from sharedness.data import H_EASY, make_view_data
from sharedness.models import fit_model


@pytest.fixture(scope="module")
def ds():
    return make_view_data(H_EASY, n_per_cohort=40, C=3, dims=15, noise=0.5,
                          cohort_rate=0.4, seed=3)


ARMS = {
    "switch_fixed": dict(w="switch", pi_mode="fixed", pi_fixed=0.5),
    "switch_global": dict(w="switch", pi_mode="global"),
    "switch_per_factor": dict(w="switch", pi_mode="per_factor"),
    "switch_frozen": dict(w="switch", frozen_mask=H_EASY),
    "switch_fixed_slab": dict(w="switch", pi_mode="global", slab="fixed"),
    "cohort_learn_global": dict(w="ARD", z="cohort_learn_pi"),
    "cohort_learn_per_factor": dict(w="ARD", z="cohort_learn_pi",
                                    cohort_pi_mode="per_factor"),
    "cohort_paired_fixed_pi": dict(w="ARD", z="cohort_learn_pi",
                                   cohort_pi_mode="fixed", cohort_paired=True),
    "cohort_paired_learn_pi": dict(w="ARD", z="cohort_learn_pi", cohort_paired=True),
}


@pytest.mark.parametrize("name", list(ARMS))
def test_elbo_monotone(ds, name):
    f = fit_model(ds, K=4, max_iter=60, elbo_tres=-np.inf, seed=0, **ARMS[name])
    d = np.diff(f.elbo)
    assert np.all(d > -1e-6 * abs(f.elbo[-1])), (name, d.min())


def _elbo(model):
    model.ELBO()
    return model.get_elbo()


def _refresh(model):
    for w in model.fa.nodelist_w:
        w.update_params()
        w.update_params_z()
    for t in model.fa.nodelist_tau:
        t.update_params_w_z()


@pytest.mark.parametrize("mode", ["global", "per_factor"])
def test_switch_pi_is_stationary(ds, mode):
    """q(pi) = Beta(a0 + sum E[h], b0 + n - sum E[h]) maximises the ELBO in (a, b)."""
    f = fit_model(ds, K=4, max_iter=40, elbo_tres=-np.inf, seed=0,
                  w="switch", pi_mode=mode)
    st, model = f.switch, f.model
    for k in range(st.K):
        st.update_pi(k)
    for pi in st.pis:
        a, b = pi.a, pi.b
        best = _elbo(model)
        for da, db in [(0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3), (0.3, 0.3), (-0.3, -0.3)]:
            pi.a, pi.b = a + da, b + db
            assert _elbo(model) <= best + 1e-9, (mode, da, db)
        pi.a, pi.b = a, b


def test_switch_h_is_stationary(ds):
    """The h_mk log-odds update maximises the ELBO in E[h_mk] (all else fixed)."""
    f = fit_model(ds, K=4, max_iter=40, elbo_tres=-np.inf, seed=0,
                  w="switch", pi_mode="fixed", pi_fixed=0.3)
    model, st = f.model, f.switch
    fa = model.fa
    for m in range(st.M):
        w = fa.nodelist_w[m]
        # Freeze alpha for this check so h is the last thing updated.
        alpha_update = w.alpha_m_node.update_k
        w.alpha_m_node.update_k = lambda k: None
        for k in range(st.K):
            w.update_k(k)
            _refresh(model)
            g = st.E_h[m, k]
            best = _elbo(model)
            for eps in (1e-3, -1e-3, 0.05, -0.05):
                gg = float(np.clip(g + eps, 1e-10, 1 - 1e-10))
                if gg == g:
                    continue
                st.E_h[m, k] = gg
                _refresh(model)
                assert _elbo(model) <= best + 1e-7 * abs(best), (m, k, g, eps)
            st.E_h[m, k] = g
            _refresh(model)
        w.alpha_m_node.update_k = alpha_update


@pytest.mark.parametrize("mode,paired", [("global", False), ("per_factor", False),
                                        ("global", True)])
def test_cohort_pi_is_stationary(ds, mode, paired):
    f = fit_model(ds, K=4, max_iter=40, elbo_tres=-np.inf, seed=0, w="ARD",
                  z="cohort_learn_pi", cohort_pi_mode=mode, cohort_paired=paired)
    cps, model = f.cohort_pi, f.model
    for k in range(cps.K):
        cps.update(k)
    for pi in cps.pis:
        a, b = pi.a, pi.b
        best = _elbo(model)
        for da, db in [(0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)]:
            pi.a, pi.b = a + da, b + db
            assert _elbo(model) <= best + 1e-9, (mode, paired, da, db)
        pi.a, pi.b = a, b


def test_cohort_paired_gamma_is_stationary(ds):
    """The paired gamma log-odds maximises the ELBO in E[gamma_c] (rest fixed)."""
    f = fit_model(ds, K=4, max_iter=40, elbo_tres=-np.inf, seed=0, w="ARD",
                  z="cohort_learn_pi", cohort_pi_mode="fixed", cohort_pi=0.3,
                  cohort_paired=True)
    model, z = f.model, f.model.fa.node_z
    for p in z.z_priors:
        S, _ = p._sums(z)
        p._update_gamma_paired(S)  # gamma is the last thing updated
        best = _elbo(model)
        for c in range(p.cohort.C):
            g = p.gamma_prob[c]
            for eps in (1e-3, -1e-3, 0.05, -0.05):
                gg = float(np.clip(g + eps, 1e-12, 1 - 1e-12))
                if gg == g:
                    continue
                p.gamma_prob[c] = gg
                p._update_expectations()
                assert _elbo(model) <= best + 1e-7 * abs(best), (p.k, c, g, eps)
            p.gamma_prob[c] = g
            p._update_expectations()


# ---------------------------------------------------------------------------
# Corrected element spike-and-slab (src/w_priors/ARD_SS.py)
# ---------------------------------------------------------------------------


def _fit_ard_ss(ds, iters=60):
    from src.build_model import FACTModel
    from src.enums import Likelihood, WPrior, ZPrior
    from src.model_config import ModelConfig, SimpleViewConfig

    cfg = ModelConfig(
        simple_view_configs=[SimpleViewConfig(likelihood=Likelihood.NORMAL,
                                              w_prior=WPrior.ARD_SS) for _ in ds.Ys],
        structured_view_configs=[], z_priors=[ZPrior.STD_NORMAL] * 4)
    model = FACTModel(views=ds.views, K=4, model_config=cfg, seed=0)
    np.random.seed(0)
    model.fit(max_iter=iters, elbo_tres=-np.inf, pretrain=True)
    return model


def test_ard_ss_elbo_monotone(ds):
    model = _fit_ard_ss(ds)
    d = np.diff(model.elbo_sequence)
    assert np.all(d > -1e-6 * abs(model.elbo_sequence[-1])), d.min()


def test_ard_ss_s_is_stationary(ds):
    """E[s_dk] from the corrected log-odds maximises the ELBO (rest fixed)."""
    model = _fit_ard_ss(ds, iters=40)
    fa = model.fa
    for m in range(fa.M):
        w = fa.nodelist_w[m]
        keep_alpha, keep_theta = w.alpha_m_node.update_k, w.theta_m_node.update_k
        w.alpha_m_node.update_k = lambda k: None
        w.theta_m_node.update_k = lambda k: None
        for k in range(4):
            w.update_k(k)
            _refresh(model)
            best = _elbo(model)
            for d in range(0, w.D, 7):
                g = w.s_m_node.vi_gamma[d, k]
                for eps in (1e-3, -1e-3, 0.05, -0.05):
                    gg = float(np.clip(g + eps, 1e-12, 1 - 1e-12))
                    if gg == g:
                        continue
                    w.s_m_node.vi_gamma[d, k] = gg
                    _refresh(model)
                    assert _elbo(model) <= best + 1e-7 * abs(best), (m, k, d, g, eps)
                w.s_m_node.vi_gamma[d, k] = g
                _refresh(model)
        w.alpha_m_node.update_k, w.theta_m_node.update_k = keep_alpha, keep_theta


# ---------------------------------------------------------------------------
# Scale-free paired cohort prior
# ---------------------------------------------------------------------------

SF_ARMS = {
    "sf_fixed_pi": dict(w="ARD", z="cohort_sf", cohort_pi_mode="fixed", cohort_pi=0.3),
    "sf_global_pi": dict(w="ARD", z="cohort_sf", cohort_pi_mode="global"),
    "sf_global_pi_reference": dict(w="ARD", z="cohort_sf", cohort_pi_mode="global",
                                   reference="largest"),
}


@pytest.mark.parametrize("name", list(SF_ARMS))
def test_scale_free_elbo_monotone(ds, name):
    f = fit_model(ds, K=4, max_iter=60, elbo_tres=-np.inf, seed=0, **SF_ARMS[name])
    d = np.diff(f.elbo)
    assert np.all(d > -1e-6 * abs(f.elbo[-1])), (name, d.min())


def test_scale_free_gamma_and_pi_stationary(ds):
    f = fit_model(ds, K=4, max_iter=40, elbo_tres=-np.inf, seed=0, **SF_ARMS["sf_global_pi"])
    model, z, cps = f.model, f.model.fa.node_z, f.cohort_pi
    for p in z.z_priors:
        p._update_cohort_params(z, p.k)
    # gamma: re-solve with tau fixed so gamma is the last update
    for p in z.z_priors:
        idx, counts = p.cohort.cohort_indices, p.cohort.counts
        S = np.bincount(idx, weights=np.asarray(z.E_z[:, p.k], float), minlength=p.cohort.C)
        Et = p.E_tau
        p.delta_mu = S / (p.kappa + counts)
        p.delta_var = 1.0 / (Et * (p.kappa + counts))
        p.v0 = 1.0 / (p.kappa * Et)
        u = p.logit_pi - 0.5 * np.log1p(counts / p.kappa) + Et * S**2 / (2 * (p.kappa + counts))
        p.gamma_prob = 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60)))
        p._update_expectations()
    best = _elbo(model)
    for p in z.z_priors:
        for c in range(p.cohort.C):
            g = p.gamma_prob[c]
            for eps in (1e-3, -1e-3, 0.05, -0.05):
                gg = float(np.clip(g + eps, 1e-12, 1 - 1e-12))
                if gg == g:
                    continue
                p.gamma_prob[c] = gg
                p._update_expectations()
                assert _elbo(model) <= best + 1e-7 * abs(best), (p.k, c, g, eps)
            p.gamma_prob[c] = g
            p._update_expectations()
    # pi
    for k in range(cps.K):
        cps.pis[0].update(sum(q.free_gamma().sum() for q in cps.priors),
                          sum(len(q.free_gamma()) for q in cps.priors))
    pi = cps.pis[0]
    a, b = pi.a, pi.b
    best = _elbo(model)
    for da, db in [(0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)]:
        pi.a, pi.b = a + da, b + db
        assert _elbo(model) <= best + 1e-9
    pi.a, pi.b = a, b


# ---------------------------------------------------------------------------
# Frozen view pattern + corrected element spike-and-slab
# ---------------------------------------------------------------------------


def test_masked_ss_monotone_and_exact_zeros(ds):
    f = fit_model(ds, K=4, max_iter=60, elbo_tres=-np.inf, seed=0, w="masked_SS",
                  frozen_mask=H_EASY)
    d = np.diff(f.elbo)
    assert np.all(d > -1e-6 * abs(f.elbo[-1])), d.min()
    for m, W in enumerate(f.Ws):
        assert np.all(W[:, H_EASY[m] == 0] == 0.0), m


def test_pipeline_two_stage_runs(ds):
    from sharedness.pipeline import fit_two_stage, gcca_init

    Z0, n_shared = gcca_init(ds.Ys, 4)
    assert Z0.shape == (ds.N, 4) and 0 <= n_shared <= 4
    out = fit_two_stage(ds, 4, n_starts=2, max_iter=80, refine=False)
    assert out["H"].shape == (4, 4)
    for m, W in enumerate(out["stage2"].Ws):
        assert np.all(W[:, out["H"][m] == 0] == 0.0)


def test_masked_ss_all_ones_equals_ard_ss(ds):
    """With nothing masked, the masked prior is exactly the corrected ARD_SS."""
    a = fit_model(ds, K=4, max_iter=30, elbo_tres=-np.inf, seed=0, w="ARD_SS")
    b = fit_model(ds, K=4, max_iter=30, elbo_tres=-np.inf, seed=0, w="masked_SS",
                  frozen_mask=np.ones((4, 4), int))
    assert np.allclose(a.elbo, b.elbo, rtol=0, atol=1e-6)


def test_masked_columns_leave_the_elbo(ds):
    """A masked column contributes nothing besides its effect on the fit: its
    theta stays at the prior and its alpha/w_hat/s terms are not counted."""
    f = fit_model(ds, K=4, max_iter=30, elbo_tres=-np.inf, seed=0, w="masked_SS",
                  frozen_mask=H_EASY)
    for m in range(4):
        th = f.model.fa.nodelist_theta[m]
        off = H_EASY[m] == 0
        assert np.allclose(th.vi_a[off], th.a0) and np.allclose(th.vi_b[off], th.b0)
