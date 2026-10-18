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
    """Each learned arm is labelled separately: 'drifts to 1' (E[pi] > 0.8 and
    fewer than 10% of switches off, as in synthetic O2) or 'stays inside (0,1)'."""
    d = df[df.dataset == name].groupby("arm")[["E_pi", "sd_pi", "frac_gamma_off"]].mean()
    labels, parts = [], []
    for arm, short in [("learned pi (current q)", "current q"),
                       ("learned pi (paired q)", "paired q")]:
        r = d.loc[arm]
        drift = r.E_pi > 0.8 and r.frac_gamma_off < 0.1
        labels.append(f"{short}: {'drifts to 1' if drift else 'stays inside (0,1)'}")
        parts.append(f"{short}: E[pi] = {r.E_pi:.2f} ± {r.sd_pi:.2f}, "
                     f"{r.frac_gamma_off:.0%} of switches off")
    return "; ".join(labels), "; ".join(parts) + "."


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
        out[name] = dict(r_ref=r0, S_ref=mt.sharedness_entropy(r0), Z_ref=Z0,
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


def run_q4_real(data: dict, K: dict, seeds=range(3), slabs=("ard", "fixed")) -> dict:
    """Returns out[dataset][slab] (slab "ard": learned alpha; "fixed": alpha = 1/0.25)."""
    jobs = [(n, d, K[n], s, "global", sl) for n, d in data.items() for s in seeds
            for sl in slabs]
    res = _pool(_switch_job, jobs)
    res = [r + (j[5],) for j, r in zip(jobs, res)]
    out = {}
    for name in data:
        out[name] = {}
        for sl in slabs:
            out[name][sl] = _summarise_switch(
                {s: (Z, Eh, al) for n, s, _, Z, Eh, _, al, _, slab in res
                 if n == name and slab == sl},
                all(m for n, *_, m, slab in res if n == name and slab == sl))
    return out


def _summarise_switch(fits: dict, monotone: bool) -> dict:
    """Reference = seed 0; stability = share of identical on/off calls in other seeds."""
    Z0, Eh0, al0 = fits[0]
    call0 = Eh0 > 0.5
    agree = []
    for s, (Z, Eh, _) in fits.items():
        if s == 0:
            continue
        idx, _ = _match_to(Z0, Z)
        agree.append(float(np.mean((Eh[:, idx] > 0.5) == call0)))
    both = call0.any() and (~call0).any()
    ard_ok = al0 is not None and np.ptp(al0) > 0
    return dict(
        E_h_ref=Eh0, Z_ref=Z0,
        on_share=float(np.mean(call0)),
        decisive=mt.decisiveness(Eh0),
        seed_agreement=float(np.mean(agree)) if agree else np.nan,
        ard_vs_switch_auc=mt.pattern_auc(mt.activity(1 / al0), call0.astype(int))
        if (both and ard_ok) else np.nan,
        monotone=monotone,
    )


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


ACTIVE_THR = 0.1  # a view "uses" a factor if it carries >= 10% of that factor's peak share


def run_q5_real(data: dict, K: dict, q2: dict, thetas=(0, 15, 30, 45),
                frac: float = 0.25) -> pd.DataFrame:
    """Rotate the most shared factor (max S_k) into the most private one (min S_k)
    of the seed-0 ARD fit. The frozen arm fixes the pattern
    mask[m, k] = 1{activity_mk >= ACTIVE_THR} read off that same fit, so it has
    real zeros (the learned ARD-slab switch never produces any on these data)."""
    jobs = []
    for name, ds in data.items():
        Z, r, S = q2[name]["Z_ref"], q2[name]["r_ref"], q2[name]["S_ref"]
        mask = (mt.activity(r) >= ACTIVE_THR).astype(int)
        S = np.nan_to_num(S, nan=0.0)
        shared, private = int(np.argmax(S)), int(np.argmin(S))
        pair = (shared, private)
        sub, idx = subsample(ds, frac, seed=0)
        for N_label, d, Zb in [(sub.N, sub, Z[idx] - Z[idx].mean(0)), (ds.N, ds, Z)]:
            for t_ in thetas:
                for a in Q5_ARMS:
                    jobs.append((name, d, Zb, mask, t_, a, N_label, pair))
    df = pd.DataFrame(_pool(_q5r_job, jobs))
    # On real data the frozen pattern is read off the fit, not known.
    df["arm"] = df["arm"].replace({"frozen (true pattern)": "frozen (prescribed pattern)"})
    base = df[df.theta == 0].set_index(["dataset", "N", "arm"])["elbo"]
    df["gap"] = [base[(r.dataset, r.N, r.arm)] - r.elbo for r in df.itertuples()]
    return df


def q5_real_masks(q2: dict) -> dict:
    return {n: (mt.activity(v["r_ref"]) >= ACTIVE_THR).astype(int) for n, v in q2.items()}


def decide_q5_real(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Like experiments.decide_q5, but robust to a gap that is ~0 or negative at
    the small N (a negative gap means the rotated point re-optimised to a better
    ELBO than the reference, i.e. the reference is not a strict optimum)."""
    tmax = df.theta.max()
    d = df[df.theta == tmax].groupby(["arm", "N"])["gap"].mean().unstack()
    Ns = sorted(d.columns)
    n_ratio = Ns[-1] / Ns[0]
    out = {}
    for arm, row in d.iterrows():
        g_small, g_big = row[Ns[0]], row[Ns[-1]]
        if g_big < 5:
            out[arm] = ("O1", f"FLAT: gap {g_big:.1f} nats at N={Ns[-1]} (theta={tmax}).")
        elif g_small < 1:
            out[arm] = ("O3", f"GROWS, RATE UNDEFINED: gap {g_small:.1f} -> {g_big:.0f} nats; "
                               f"the small-N gap is ~0, so no growth rate can be read.")
        elif g_big / g_small >= 0.6 * n_ratio:
            out[arm] = ("O2", f"GROWS LIKE N: gap {g_small:.0f} -> {g_big:.0f} nats "
                               f"(x{g_big / g_small:.1f} for x{n_ratio:.0f} samples).")
        else:
            out[arm] = ("O3", f"GROWS SLOWER THAN N: gap {g_small:.0f} -> {g_big:.0f} nats "
                               f"(x{g_big / g_small:.1f} for x{n_ratio:.0f} samples).")
    return out


# ---------------------------------------------------------------------------
# Q6
# ---------------------------------------------------------------------------


def run_q6_real(data: dict, K: dict, seeds=range(3)) -> pd.DataFrame:
    res = _pool(_switch_job, [(n, d, K[n], s, m, "fixed") for n, d in data.items()
                              for s in seeds for m in ("global", "per_factor")])
    calls = {(n, s, m): Eh > 0.5 for n, s, m, _, Eh, *_ in res}
    rows = []
    for n, s, m, Z, Eh, pis, _, mono in res:
        rows.append(dict(dataset=n, seed=s, mode=m, on_share=float(np.mean(Eh > 0.5)),
                         pi_mean=float(np.mean(pis["mean"])),
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
