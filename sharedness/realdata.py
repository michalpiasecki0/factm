"""
Q1-Q6 on real data. No truth exists, so "recovery" becomes **stability**
(the same answer from different seeds) and identification is tested with the
freeze-Z rotation on a subsample versus the full data.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

from dataclasses import replace

import numpy as np
import pandas as pd

from src.views import Views

from . import metrics as mt
from .experiments import Q1_ARMS, Q5_ARMS, _pool
from .models import fit_model, fit_switch_warm, freeze_z_elbo, givens


def subsample(ds, frac: float, seed: int = 0):
    """A random subset of the samples (same columns, re-centred)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.N, size=int(round(frac * ds.N)), replace=False))
    Ys = [Y[idx] - Y[idx].mean(0) for Y in ds.Ys]
    labels = ds.labels[idx]
    return replace(ds, Ys=Ys, labels=labels, views=Views.from_list(Ys, cohorts=labels)), idx


def _match_to(Z_ref, Z):
    """Indices of Z's columns matching Z_ref's columns (Hungarian on |corr|)."""
    idx, cor = mt.match_factors(Z_ref, Z)
    return idx, cor


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------


def _q1r_job(args):
    name, ds, K, seed, arm = args
    f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD", **Q1_ARMS[arm])
    p = f.cohort_pi.pis[0] if f.cohort_pi.learns else None
    G = f.E_gamma
    return dict(dataset=name, seed=seed, arm=arm, E_pi=p.mean if p else 0.5,
                sd_pi=p.sd if p else 0.0, frac_gamma_on=float(np.mean(G > 0.9)),
                frac_gamma_off=float(np.mean(G < 0.1)), monotone=f.monotone,
                n_iter=len(f.elbo))


def run_q1_real(data: dict, K: dict, seeds=range(3)) -> pd.DataFrame:
    jobs = [(n, d, K[n], s, a) for n, d in data.items() for s in seeds for a in Q1_ARMS]
    return pd.DataFrame(_pool(_q1r_job, jobs))


def decide_q1_real(df: pd.DataFrame, name: str) -> tuple[str, str]:
    d = df[df.dataset == name].groupby("arm")[["E_pi", "frac_gamma_off"]].mean()
    cur = d.loc["learned pi (current q)"]
    pai = d.loc["learned pi (paired q)"]
    txt = (f"current q: E[pi] = {cur.E_pi:.2f}, {cur.frac_gamma_off:.0%} of switches off; "
           f"paired q: E[pi] = {pai.E_pi:.2f}, {pai.frac_gamma_off:.0%} off.")
    if cur.E_pi > 0.8 and cur.frac_gamma_off < 0.1 and pai.frac_gamma_off >= 0.2:
        return "like synthetic O2 → fixed by pairing", txt
    if cur.E_pi > 0.8 and pai.E_pi > 0.8:
        return "like synthetic O2, not fixed by pairing", txt
    return "no drift", txt


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------


def _fit_ard_job(args):
    name, ds, K, seed = args
    f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD")
    return name, seed, f.Z, f.Ws, f.alpha, f.monotone


def run_q2_real(data: dict, K: dict, seeds=range(3)) -> dict:
    res = _pool(_fit_ard_job, [(n, d, K[n], s) for n, d in data.items() for s in seeds])
    out = {}
    for name, ds in data.items():
        fits = {s: (Z, Ws) for n, s, Z, Ws, _, _ in res if n == name}
        Z0, W0 = fits[0]
        r0 = mt.variance_share(Z0, W0, ds.Ys)
        a0 = mt.activity(r0)
        corrs, cors = [], []
        for s, (Z, Ws) in fits.items():
            if s == 0:
                continue
            idx, cor = _match_to(Z0, Z)
            a = mt.activity(mt.variance_share(Z, Ws, ds.Ys))[:, idx]
            corrs.append(float(np.corrcoef(a.ravel(), a0.ravel())[0, 1]))
            cors.append(cor)
        out[name] = dict(r_ref=r0, S_ref=mt.sharedness_entropy(r0),
                         stability=float(np.mean(corrs)),
                         match_corr=np.mean(cors, axis=0),
                         monotone=all(m for n, _, _, _, _, m in res if n == name))
    return out


def decide_q2_real(res: dict) -> tuple[str, str]:
    st, mc = res["stability"], float(np.min(res["match_corr"]))
    txt = (f"activity matrices agree across seeds with corr {st:.2f}; the least stable "
           f"factor matches at |corr| {mc:.2f}.")
    if st >= 0.9 and mc >= 0.9:
        return "stable read-out", txt
    if st >= 0.7:
        return "partly stable", txt
    return "unstable", txt


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------


REAL_SLAB_VAR = 0.25  # loadings of standardised views are small; see notebook 03


def _switch_job(args):
    name, ds, K, seed, mode, slab = args
    kw = dict(slab=slab) if slab == "ard" else dict(slab="fixed", slab_var=REAL_SLAB_VAR)
    f = fit_switch_warm(ds, K, max_iter=400, seed=seed, pi_mode=mode, **kw)
    return name, seed, mode, f.Z, f.E_h, f.switch.pi_summary(), f.alpha, f.monotone


def run_q4_real(data: dict, K: dict, seeds=range(3)) -> dict:
    res = _pool(_switch_job, [(n, d, K[n], s, "global", "ard") for n, d in data.items()
                              for s in seeds])
    out = {}
    for name in data:
        fits = {s: (Z, Eh, al) for n, s, _, Z, Eh, _, al, _ in res if n == name}
        Z0, Eh0, al0 = fits[0]
        call0 = Eh0 > 0.5
        agree = []
        for s, (Z, Eh, _) in fits.items():
            if s == 0:
                continue
            idx, _ = _match_to(Z0, Z)
            agree.append(float(np.mean((Eh[:, idx] > 0.5) == call0)))
        both = call0.any() and (~call0).any()
        out[name] = dict(
            E_h_ref=Eh0, Z_ref=Z0,
            on_share=float(np.mean(call0)),
            decisive=mt.decisiveness(Eh0),
            seed_agreement=float(np.mean(agree)),
            ard_vs_switch_auc=mt.pattern_auc(mt.activity(1 / al0), call0.astype(int))
            if both else np.nan,
            monotone=all(m for n, *_, m in res if n == name),
        )
    return out


def decide_q4_real(res: dict) -> tuple[str, str]:
    txt = (f"{res['on_share']:.0%} of switches on, {res['decisive']:.0%} decisive, "
           f"seed agreement {res['seed_agreement']:.0%}, ARD agrees with the switch at "
           f"AUC {res['ard_vs_switch_auc']:.2f}.")
    if res["on_share"] >= 0.95:
        return "switch stays on (like O2)", txt
    if res["decisive"] >= 0.9 and res["seed_agreement"] >= 0.9:
        return "clean and stable (like O1)", txt
    return "mixed", txt


# ---------------------------------------------------------------------------
# Q5
# ---------------------------------------------------------------------------


def _q5r_job(args):
    name, ds, Zbar, mask, theta, arm, N_label, pair = args
    K = Zbar.shape[1]
    kw = dict(Q5_ARMS[arm])
    if kw.pop("frozen", False):
        kw["frozen_mask"] = mask
    R = givens(K, pair[0], pair[1], np.deg2rad(theta))
    out = freeze_z_elbo(ds, Zbar @ R, n_iter=200, seed=0, **kw)
    return dict(dataset=name, N=N_label, theta=theta, arm=arm, elbo=out["elbo"],
                monotone=bool(np.all(np.diff(out["trace"]) > -1e-6 * abs(out["elbo"]))))


def run_q5_real(data: dict, K: dict, q4: dict, thetas=(0, 15, 30, 45),
                frac: float = 0.25) -> pd.DataFrame:
    jobs = []
    for name, ds in data.items():
        Z, Eh = q4[name]["Z_ref"], q4[name]["E_h_ref"]
        mask = (Eh > 0.5).astype(int)
        n_on = mask.sum(0)
        shared = int(np.argmax(n_on + 1e-3 * Eh.sum(0)))
        cand = [k for k in range(Z.shape[1]) if k != shared]
        private = int(min(cand, key=lambda k: (n_on[k] if n_on[k] > 0 else 99, -Eh[:, k].max())))
        pair = (shared, private)
        sub, idx = subsample(ds, frac, seed=0)
        for N_label, d, Zb in [(sub.N, sub, Z[idx] - Z[idx].mean(0)), (ds.N, ds, Z)]:
            for t in thetas:
                for a in Q5_ARMS:
                    jobs.append((name, d, Zb, mask, t, a, N_label, pair))
    df = pd.DataFrame(_pool(_q5r_job, jobs))
    base = df[df.theta == 0].set_index(["dataset", "N", "arm"])["elbo"]
    df["gap"] = [base[(r.dataset, r.N, r.arm)] - r.elbo for r in df.itertuples()]
    return df


# ---------------------------------------------------------------------------
# Q6
# ---------------------------------------------------------------------------


def run_q6_real(data: dict, K: dict, seeds=range(3)) -> pd.DataFrame:
    res = _pool(_switch_job, [(n, d, K[n], s, m, "fixed") for n, d in data.items()
                              for s in seeds for m in ("global", "per_factor")])
    calls = {(n, s, m): Eh > 0.5 for n, s, m, _, Eh, *_ in res}
    rows = []
    for n, s, m, Z, Eh, pis, _, mono in res:
        rows.append(dict(dataset=n, seed=s, mode=m, pi_mean=float(np.mean(pis["mean"])),
                         pi_sd=float(np.mean(pis["sd"])),
                         agreement_with_global=float(np.mean(calls[(n, s, m)] ==
                                                             calls[(n, s, "global")])),
                         monotone=mono))
    return pd.DataFrame(rows)


def decide_q6_real(df: pd.DataFrame, name: str) -> tuple[str, str]:
    d = df[(df.dataset == name) & (df["mode"] == "per_factor")]
    sd, ag = d.pi_sd.mean(), d.agreement_with_global.mean()
    txt = f"per-factor pi_k posterior sd {sd:.2f}; calls agree with global pi in {ag:.0%} of cells."
    if sd >= 0.15 and ag >= 0.9:
        return "use one global pi (like O2)", txt
    if ag >= 0.9:
        return "either works (like O1)", txt
    return "per-factor changes the answer", txt
