"""
Before vs after: the same scores as notebooks 01-03, for the original pipeline
and for the improved one (notebook 04).

    before         single PCA start, ARD, 400 iterations (as in notebooks 01-03)
    multistart-pca 8 starts around the PCA start, best ELBO, relative tolerance
    multistart     8 starts around the generalised-CCA start, best ELBO
    two-stage      'multistart' + learned pattern frozen + corrected element
                   spike-and-slab refit + ELBO refinement of borderline cells
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

from . import metrics as mt
from .data import center_on_reference
from .experiments import LEVELS, _pool, level_data
from .models import fit_from_state, fit_model
from .pipeline import calibrated_slab, fit_best, fit_two_stage, learned_pattern


def _scores(ds, Z, Ws, H_call=None, strong=None) -> dict:
    idx, cor = mt.match_factors(ds.Z, Z)
    r = mt.variance_share(Z, Ws, ds.Ys)
    a = mt.activity(r)[:, idx]
    out = dict(mean_corr=float(cor.mean()), min_corr=float(cor.min()),
               auc=mt.pattern_auc(a, ds.H))
    if strong is not None:
        out["min_corr_strong"] = float(cor[strong].min())
    call = (a >= 0.1).astype(int) if H_call is None else H_call[:, idx]
    out["cell_acc"] = float(np.mean(call == ds.H))
    out["exact_cols"] = int(np.all(call == ds.H, axis=0).sum())
    out["leaks"] = int(((call == 1) & (ds.H == 0)).sum())
    out["misses"] = int(((call == 0) & (ds.H == 1)).sum())
    return out


def _syn_job(args):
    level, seed, n_starts = args
    ds, K = level_data(level, seed)
    strong = np.array([k for k, s in enumerate(np.broadcast_to(
        LEVELS[level]["loading_scale"], (ds.H.shape[1],))) if s >= 1.0])
    rows = []
    f0 = fit_model(ds, K, max_iter=400, seed=seed, w="ARD")
    rows.append(dict(arm="before", elbo=f0.elbo[-1], n_iter=len(f0.elbo),
                     **_scores(ds, f0.Z, f0.Ws, strong=strong)))
    fp = fit_best(ds, K, n_starts=n_starts, init="pca", seed=seed, w="ARD")
    rows.append(dict(arm="multistart-pca", elbo=fp.elbo[-1], n_iter=len(fp.elbo),
                     **_scores(ds, fp.Z, fp.Ws, strong=strong)))
    fg = fit_best(ds, K, n_starts=n_starts, init="gcca", seed=seed, w="ARD")
    rows.append(dict(arm="multistart", elbo=fg.elbo[-1], n_iter=len(fg.elbo),
                     **_scores(ds, fg.Z, fg.Ws, strong=strong)))
    ts = fit_two_stage(ds, K, stage1=fg, seed=seed, refine=True)
    s2 = ts["stage2"]
    rows.append(dict(arm="two-stage", elbo=s2.elbo[-1], n_iter=len(s2.elbo),
                     flips=len(ts["flips"]),
                     **_scores(ds, s2.Z, s2.Ws, H_call=ts["H"], strong=strong)))
    # view switch with a calibrated fixed slab, started from the multistart fit
    sv = calibrated_slab(fg, ds.Ys)
    fa = fg.model.fa
    fs = fit_from_state(ds, K, fg.Z, [np.asarray(h.vi_mu) for h in fa.nodelist_hat_w],
                        max_iter=1500, rel_tol=1e-7, seed=seed, w="switch", slab="fixed",
                        slab_var=sv, pi_mode="global", tau_from=fa.nodelist_tau)
    idx, _ = mt.match_factors(ds.Z, fs.Z)
    Eh = fs.E_h[:, idx]
    rows.append(dict(arm="switch, calibrated slab", elbo=fs.elbo[-1], n_iter=len(fs.elbo),
                     **_scores(ds, fs.Z, fs.Ws, H_call=(fs.E_h > 0.5).astype(int),
                               strong=strong),
                     on_share=float(np.mean(Eh > 0.5)), switch_auc=mt.pattern_auc(Eh, ds.H)))
    for r in rows:
        r.update(level=level, seed=seed)
    return rows


def run_synthetic(levels=("easy", "hard"), seeds=range(5), n_starts: int = 8) -> pd.DataFrame:
    jobs = [(lv, s, n_starts) for lv in levels for s in seeds]
    return pd.DataFrame([r for rows in _pool(_syn_job, jobs) for r in rows])


# ---------------------------------------------------------------------------
# Cohort pi: before (current factorisation) vs after (scale-free paired)
# ---------------------------------------------------------------------------

COHORT_ARMS = {
    "before: current q, fixed pi=0.5": dict(z="cohort_learn_pi", cohort_pi_mode="fixed",
                                            cohort_pi=0.5),
    "before: current q, learned pi": dict(z="cohort_learn_pi", cohort_pi_mode="global"),
    "paired q, learned pi (notebook 01)": dict(z="cohort_learn_pi", cohort_pi_mode="global",
                                               cohort_paired=True),
    "after: scale-free paired, learned pi": dict(z="cohort_sf", cohort_pi_mode="global",
                                                 kappa=1.0),
    "after: + reference cohort": dict(z="cohort_sf", cohort_pi_mode="global", kappa=1.0,
                                      reference="largest"),
}


def _cohort_job(args):
    level, rate, seed, arm = args
    ds, K = level_data(level, seed, C=6, n_per_cohort=LEVELS[level]["n_per_cohort"] // 2,
                       cohort_rate=rate)
    if "reference" in arm:  # reference coding = centre the data on the reference cohort
        ds, _ = center_on_reference(ds)
    f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD", **COHORT_ARMS[arm])
    idx, cor = mt.match_factors(ds.Z, f.Z)
    Eg = f.E_gamma[:, idx]
    codes = np.unique(ds.labels, return_inverse=True)[1]
    C = ds.info["C"]
    mu = np.vstack([ds.Z[codes == c].mean(0) for c in range(C)])
    if "reference" in arm:  # truth relative to the reference cohort
        r = int(np.argmax(np.bincount(codes)))
        G = (np.abs(mu - mu[r]) > 1.0).astype(int)
        keep = np.arange(C) != r
        G, Eg = G[keep], Eg[keep]
    else:
        G = ds.info["G_eff"]
    learns = f.cohort_pi.learns
    p = f.cohort_pi.pis[0]
    return dict(level=level, rate=rate, seed=seed, arm=arm, true_rate=float(G.mean()),
                E_pi=p.mean if learns else 0.5,
                pi_err=abs((p.mean if learns else 0.5) - G.mean()),
                gamma_auc=mt.pattern_auc(Eg, G),
                null_gamma=float(Eg[G == 0].mean()) if (G == 0).any() else np.nan,
                alt_gamma=float(Eg[G == 1].mean()) if (G == 1).any() else np.nan,
                min_corr=float(cor.min()), monotone=f.monotone)


def run_cohort(levels=("easy", "hard"), rates=(0.1, 0.3, 0.6), seeds=range(3)) -> pd.DataFrame:
    jobs = [(lv, r, s, a) for lv in levels for r in rates for s in seeds for a in COHORT_ARMS]
    return pd.DataFrame(_pool(_cohort_job, jobs))


# ---------------------------------------------------------------------------
# Real data: seed stability before vs after, and the final model's pattern
# ---------------------------------------------------------------------------


def _real_job(args):
    name, ds, K, seed, arm, n_starts, max_iter = args
    if arm == "before":
        f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD")
        return dict(name=name, seed=seed, arm=arm, Z=f.Z, Ws=f.Ws, elbo=f.elbo[-1], H=None)
    fg = fit_best(ds, K, n_starts=n_starts, init="gcca", seed=seed, w="ARD",
                  max_iter=max_iter)
    if arm == "multistart":
        return dict(name=name, seed=seed, arm=arm, Z=fg.Z, Ws=fg.Ws, elbo=fg.elbo[-1], H=None)
    ts = fit_two_stage(ds, K, stage1=fg, seed=seed, refine=False, max_iter=max_iter)
    s2 = ts["stage2"]
    return dict(name=name, seed=seed, arm=arm, Z=s2.Z, Ws=s2.Ws, elbo=s2.elbo[-1], H=ts["H"])


def run_real(data: dict, K: dict, seeds=range(3), n_starts: int = 4, max_iter: int = 800):
    jobs = [(n, d, K[n], s, a, n_starts, max_iter) for n, d in data.items() for s in seeds
            for a in ("before", "multistart", "two-stage")]
    res = _pool(_real_job, jobs)
    rows, refs = [], {}
    for name, ds in data.items():
        for arm in ("before", "multistart", "two-stage"):
            fits = sorted([r for r in res if r["name"] == name and r["arm"] == arm],
                          key=lambda r: r["seed"])
            ref = fits[0]
            a0 = mt.activity(mt.variance_share(ref["Z"], ref["Ws"], ds.Ys))
            corr_act, match = [], []
            for f in fits[1:]:
                idx, cor = mt.match_factors(ref["Z"], f["Z"])
                a = mt.activity(mt.variance_share(f["Z"], f["Ws"], ds.Ys))[:, idx]
                corr_act.append(float(np.corrcoef(a.ravel(), a0.ravel())[0, 1]))
                match.append(cor)
            match = np.array(match)
            rows.append(dict(dataset=name, arm=arm,
                             activity_corr=float(np.mean(corr_act)),
                             mean_factor_match=float(match.mean()),
                             min_factor_match=float(match.min(axis=1).mean()),
                             best_elbo=float(max(f["elbo"] for f in fits))))
            refs[(name, arm)] = ref
    return pd.DataFrame(rows), refs


def run_real_cohort(data: dict, K: dict, seed: int = 0) -> pd.DataFrame:
    jobs = [(n, d, K[n], seed, a) for n, d in data.items()
            for a in ("before: current q, learned pi", "paired q, learned pi (notebook 01)",
                      "after: scale-free paired, learned pi")]
    return pd.DataFrame(_pool(_real_cohort_job, jobs))


def _real_cohort_job(args):
    name, ds, K, seed, arm = args
    f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD", **COHORT_ARMS[arm])
    p = f.cohort_pi.pis[0]
    G = f.E_gamma
    return dict(dataset=name, arm=arm, E_pi=p.mean, sd_pi=p.sd,
                frac_on=float(np.mean(G > 0.9)), frac_off=float(np.mean(G < 0.1)),
                monotone=f.monotone)
