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
