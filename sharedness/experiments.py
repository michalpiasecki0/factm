"""
Runners for questions Q1-Q6, one per question, plus a decision rule for each.

Every runner returns a tidy ``pandas.DataFrame`` (one row per fit). Every
decision rule maps that frame to one of the outcomes written down in
notebook 00 *before* any result was seen, and returns ``(code, sentence)``.
The thresholds live here, in code, so the notebooks cannot move them after
looking at the numbers.

Difficulty levels (see notebook 00 for the reasoning):

    easy  -- 4 views of 30 features, N = 300, noise sd 0.5, uncorrelated
             loadings, every active factor strong, K fitted = K true.
    hard  -- 4 views of 80/40/15/10 features, N = 150, noise sd 1.2,
             correlated loadings (rho = 0.5), one weak factor (scale 0.4),
             K fitted = K true + 2.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import metrics as mt
from .data import H_EASY, make_view_data, pattern_for
from .models import fit_from_state, fit_model, fit_switch_warm, freeze_z_elbo, givens

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

H_HARD = np.array(
    [  # f0 all | f1 v1-v2 | f2 v2-v4 | f3 priv v1 | f4 priv v3 | f5 weak priv v4
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ]
)

LEVELS = {
    "easy": dict(H=H_EASY, n_per_cohort=100, C=3, dims=30, noise=0.5, rho=0.0,
                 loading_scale=1.0, extra_K=0),
    "hard": dict(H=H_HARD, n_per_cohort=50, C=3, dims=(80, 40, 15, 10), noise=1.2,
                 rho=0.5, loading_scale=[1, 1, 1, 1, 1, 0.4], extra_K=2),
}


def level_data(level: str, seed: int, **override):
    cfg = dict(LEVELS[level])
    extra = cfg.pop("extra_K")
    cfg.update(override)
    ds = make_view_data(seed=seed, **cfg)
    return ds, ds.H.shape[1] + extra


def _pool(fn, jobs):
    if len(jobs) == 1 or N_WORKERS == 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        return list(ex.map(fn, jobs))


# Fitting arms used by several questions. "warm" = start from a converged ARD fit.
ARMS = {
    "NONE": dict(w="NONE"),
    "ARD": dict(w="ARD"),
    "switch_cold": dict(w="switch", slab="ard", pi_mode="global"),
    "switch_warm": dict(w="switch", slab="ard", pi_mode="global", warm=True),
    "switch_warm_fixedslab": dict(w="switch", slab="fixed", slab_var=1.0,
                                  pi_mode="global", warm=True),
    "ARD_truthstart": dict(w="ARD", truth_start=True),
}


def fit_arm(ds, K, arm: dict, seed: int, max_iter: int = 400):
    arm = dict(arm)
    if arm.pop("truth_start", False):
        rng = np.random.default_rng(seed + 1000)
        Ktrue = ds.Z.shape[1]
        Z0 = np.hstack([ds.Z, 0.1 * rng.normal(size=(ds.N, K - Ktrue))])
        W0 = [np.hstack([W, 0.01 * rng.normal(size=(W.shape[0], K - Ktrue))]) for W in ds.Ws]
        return fit_from_state(ds, K, Z0, W0, max_iter=max_iter, seed=seed, **arm)
    if arm.pop("warm", False):
        arm.pop("w")
        return fit_switch_warm(ds, K, max_iter=max_iter, seed=seed, **arm)
    return fit_model(ds, K, max_iter=max_iter, seed=seed, **arm)


def score_fit(ds, f) -> dict:
    """Every per-fit number used anywhere. Truth-based ones only if ds has H."""
    r = mt.variance_share(f.Z, f.Ws, ds.Ys)
    out = dict(n_iter=len(f.elbo), elbo=f.elbo[-1], monotone=f.monotone)
    out["r"] = r
    out["S_fit"] = mt.sharedness_entropy(r)
    # A fixed slab pins alpha, so its "ARD activity" carries no information.
    ard_learned = f.alpha is not None and (f.switch is None or f.switch.slab == "ard")
    if ard_learned:
        out["ard"] = mt.ard_score(f.alpha)
    if f.E_h is not None:
        out["E_h"] = f.E_h
        out["pi"] = f.switch.pi_summary()
    if ds.H is not None:
        idx, cor = mt.match_factors(ds.Z, f.Z)
        out["match_idx"], out["match_corr"] = idx, cor
        out["min_corr"] = float(cor.min())
        out["auc_r"] = mt.pattern_auc(mt.activity(r)[:, idx], ds.H)
        if ard_learned:
            out["auc_ard"] = mt.pattern_auc(mt.activity(out["ard"])[:, idx], ds.H)
        if f.E_h is not None:
            Eh = f.E_h[:, idx]
            out["auc_h"] = mt.pattern_auc(Eh, ds.H)
            out["decisive_h"] = mt.decisiveness(Eh)
            e = mt.pattern_errors(Eh, ds.H)
            out.update(leak_h=e["leak"], miss_h=e["miss"], exact_h=e["exact_columns"])
        e = mt.pattern_errors(mt.activity(r)[:, idx], ds.H)
        out.update(leak_r=e["leak"], miss_r=e["miss"], exact_r=e["exact_columns"])
        # How much variance the unmatched (surplus) factors carry.
        surplus = np.setdiff1d(np.arange(r.shape[1]), idx)
        out["surplus_share"] = float(r[:, surplus].sum(1).max()) if len(surplus) else 0.0
    return out


# ---------------------------------------------------------------------------
# Q1 -- can CohortFACTM learn how often cohorts differ?
# ---------------------------------------------------------------------------

Q1_ARMS = {
    "fixed pi=0.5 (current)": dict(z="cohort_learn_pi", cohort_pi_mode="fixed",
                                   cohort_pi=0.5),
    "learned pi (current q)": dict(z="cohort_learn_pi", cohort_pi_mode="global"),
    "learned pi (paired q)": dict(z="cohort_learn_pi", cohort_pi_mode="global",
                                  cohort_paired=True),
}


def _q1_job(args):
    level, rate, seed, arm_name = args
    ds, K = level_data(level, seed, C=6, n_per_cohort=LEVELS[level]["n_per_cohort"] // 2,
                       cohort_rate=rate)
    f = fit_model(ds, K, max_iter=400, seed=seed, w="ARD", **Q1_ARMS[arm_name])
    idx, cor = mt.match_factors(ds.Z, f.Z)
    Eg = f.E_gamma[:, idx]
    G = ds.info["G_eff"]
    p = f.cohort_pi.pis[0] if f.cohort_pi.learns else None
    return dict(
        level=level, rate=rate, seed=seed, arm=arm_name,
        true_rate=float(G.mean()),
        E_pi=p.mean if p else 0.5, sd_pi=p.sd if p else 0.0,
        mean_gamma=float(Eg.mean()),
        mean_gamma_null=float(Eg[G == 0].mean()) if (G == 0).any() else np.nan,
        mean_gamma_alt=float(Eg[G == 1].mean()) if (G == 1).any() else np.nan,
        auc_gamma=mt.pattern_auc(Eg, G),
        min_corr=float(cor.min()), monotone=f.monotone, n_iter=len(f.elbo),
    )


def run_q1(level: str, rates=(0.1, 0.3, 0.6), seeds=range(5)) -> pd.DataFrame:
    jobs = [(level, r, s, a) for r in rates for s in seeds for a in Q1_ARMS]
    return pd.DataFrame(_pool(_q1_job, jobs))


def decide_q1(df: pd.DataFrame, arm: str) -> tuple[str, str]:
    d = df[df.arm == arm]
    g = d.groupby("rate").agg(true=("true_rate", "mean"), est=("E_pi", "mean"))
    slope = np.polyfit(g["true"], g["est"], 1)[0] if len(g) > 1 else np.nan
    err = float(np.abs(g["est"] - g["true"]).max())
    null = float(d["mean_gamma_null"].mean())
    if err < 0.15 and slope > 0.5:
        return "O1", (f"LEARNABLE: E[pi] tracks the true rate (max error {err:.2f}, "
                      f"slope {slope:.2f}). pi can be learned from the data.")
    if g["est"].min() > 0.8:
        return "O2", (f"DRIFTS TO 1: E[pi] > 0.8 whatever the true rate (slope "
                      f"{slope:.2f}); cohorts with no shift still have mean "
                      f"E[gamma] = {null:.2f}. The switch has no evidence against it.")
    if (g["est"].max() - g["est"].min()) < 0.2:
        return "O3", (f"PRIOR-DOMINATED: E[pi] hardly moves with the true rate "
                      f"(range {g['est'].max() - g['est'].min():.2f}); too few switches "
                      f"(C*K) to learn from.")
    return "O4", (f"PARTIAL: E[pi] moves with the truth (slope {slope:.2f}) but is "
                  f"biased (max error {err:.2f}).")


# ---------------------------------------------------------------------------
# Q2 -- can a plain fit already show which factors are shared?
# ---------------------------------------------------------------------------


def _fit_score_job(args):
    level, seed, arm_name, override = args
    ds, K = level_data(level, seed, **override)
    f = fit_arm(ds, K, ARMS[arm_name], seed)
    s = score_fit(ds, f)
    s.update(level=level, seed=seed, arm=arm_name, **{f"set_{k}": v for k, v in override.items()})
    return s


def run_q2(level: str, seeds=range(5)) -> pd.DataFrame:
    jobs = [(level, s, a, {}) for s in seeds for a in ("NONE", "ARD", "ARD_truthstart")]
    return pd.DataFrame(_pool(_fit_score_job, jobs))


def decide_q2b(df: pd.DataFrame, ok_corr: float = 0.9, tol: float = 10.0) -> tuple[str, str]:
    """Q2b: when the ARD fit misses the true axes, is it the optimiser or the model?"""
    a = df[df.arm == "ARD"].set_index("seed")
    t = df[df.arm == "ARD_truthstart"].set_index("seed")
    fail = a.index[a.min_corr < ok_corr]
    if len(fail) == 0:
        return "O0", (f"NOT NEEDED: every ARD fit found the true axes "
                      f"(min |corr| >= {ok_corr}).")
    diff = (t.loc[fail, "elbo"] - a.loc[fail, "elbo"])
    stays = float((t.loc[fail, "min_corr"] >= ok_corr).mean())
    n = len(fail)
    if diff.mean() > tol and stays >= 0.8:
        return "O1", (f"OPTIMISER: in {n} seed(s) where ARD missed the axes, starting at the "
                      f"truth gives an ELBO {diff.mean():.0f} nats higher and stays there "
                      f"({stays:.0%}). The model prefers the truth; CAVI does not reach it.")
    if diff.mean() < -tol or stays < 0.5:
        return "O2", (f"MODEL: started at the truth, the fit ends {diff.mean():+.0f} nats "
                      f"vs the PCA start and keeps the true axes in only {stays:.0%} of {n} "
                      f"seed(s). The ELBO itself prefers a rotated solution.")
    return "O3", (f"FLAT: truth start and PCA start differ by {diff.mean():+.0f} nats "
                  f"over {n} seed(s) — the rotation is (nearly) unidentified.")


def decide_q2(df: pd.DataFrame) -> tuple[str, str]:
    df = df[df.arm.isin(["NONE", "ARD"])]
    none = df[df.arm == "NONE"]["auc_r"].mean()
    ard = df[df.arm == "ARD"]["auc_r"].mean()
    if none >= 0.95 and ard >= 0.95:
        return "O1", (f"YES, ANY FIT: the variance share reads the pattern with or "
                      f"without ARD (AUC {none:.2f} without, {ard:.2f} with).")
    if ard >= 0.95:
        return "O2", (f"YES, WITH ARD: AUC {ard:.2f} with ARD but {none:.2f} without. "
                      f"ARD is what makes the pattern readable.")
    if max(none, ard) >= 0.75:
        return "O3", (f"PARTLY: best AUC {max(none, ard):.2f}. Some cells are read "
                      f"wrongly; see leak (private read as shared) vs miss.")
    return "O4", f"NO: best AUC {max(none, ard):.2f}; the pattern is not readable."


# ---------------------------------------------------------------------------
# Q3 -- how many views do we need?
# ---------------------------------------------------------------------------


def q3_counting_table(Ms=(2, 3, 4, 6, 8)) -> pd.DataFrame:
    """Deterministic: pattern count, and how sure a per-factor pi_k can be."""
    rows = []
    for M in Ms:
        # Beta(1 + s, 1 + M - s) for a shared (s = M) and a private (s = 1) factor
        def sd(s):
            a, b = 1 + s, 1 + M - s
            return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
        rows.append(dict(
            M=M, n_patterns=2**M - 1,
            pi_k_shared_mean=(1 + M) / (2 + M), pi_k_shared_sd=sd(M),
            pi_k_private_mean=2 / (2 + M), pi_k_private_sd=sd(1),
        ))
    return pd.DataFrame(rows)


def run_q3(level: str, Ms=(2, 3, 4, 6), seeds=range(5)) -> pd.DataFrame:
    jobs = []
    for M in Ms:
        H = pattern_for(M)
        dims = 30 if level == "easy" else tuple(int(x) for x in np.linspace(80, 10, M))
        for s in seeds:
            jobs.append((level, s, "ARD", dict(H=H, dims=dims,
                         loading_scale=1.0 if level == "easy" else
                         [1.0] * (H.shape[1] - 1) + [0.4])))
    rows = _pool(_fit_score_job, jobs)
    for r in rows:
        r["M"] = r["set_H"].shape[0]
    return pd.DataFrame(rows)


def decide_q3(df: pd.DataFrame) -> tuple[str, str]:
    g = df.groupby("M")["auc_ard"].mean()
    lo, hi = g.iloc[0], g.iloc[-1]
    if g.min() >= 0.95:
        return "O1", (f"M DOES NOT LIMIT DETECTION: AUC >= 0.95 at every M "
                      f"(M={g.index[0]}: {lo:.2f}). With M = 2 the question is "
                      f"answerable, just coarse (3 patterns).")
    if hi - lo >= 0.05:
        return "O2", (f"MORE VIEWS HELP: AUC rises from {lo:.2f} (M={g.index[0]}) to "
                      f"{hi:.2f} (M={g.index[-1]}).")
    if lo - hi >= 0.05:
        return "O3", (f"MORE VIEWS HURT: AUC falls from {lo:.2f} to {hi:.2f}; more "
                      f"private factors split the signal.")
    return "O4", f"FLAT BUT IMPERFECT: AUC {lo:.2f}-{hi:.2f} across M."


# ---------------------------------------------------------------------------
# Q4 -- does an explicit on/off switch give a cleaner answer than ARD?
# ---------------------------------------------------------------------------

Q4_ARMS = ("ARD", "switch_cold", "switch_warm", "switch_warm_fixedslab")


def run_q4(level: str, seeds=range(5)) -> pd.DataFrame:
    jobs = [(level, s, a, {}) for s in seeds for a in Q4_ARMS]
    return pd.DataFrame(_pool(_fit_score_job, jobs))


def decide_q4(df: pd.DataFrame) -> tuple[str, str]:
    ard = df[df.arm == "ARD"]["auc_ard"].mean()
    best_arm, best = None, -1.0
    for a in Q4_ARMS[1:]:
        v = df[df.arm == a]["auc_h"].mean()
        if v > best:
            best_arm, best = a, v
    dec = df[df.arm == best_arm]["decisive_h"].mean()
    on = df[df.arm == best_arm]["E_h"].map(lambda x: float(np.mean(x > 0.5))).mean()
    if best >= ard - 0.02 and dec >= 0.9 and best >= 0.9:
        return "O1", (f"CLEANER: best switch ({best_arm}) matches ARD (AUC {best:.2f} vs "
                      f"{ard:.2f}) and {dec:.0%} of its calls are decisive (<0.1 or >0.9).")
    if on >= 0.95:
        return "O2", (f"SWITCH STAYS ON: even the best arm ({best_arm}) leaves {on:.0%} "
                      f"of switches on. The switch cannot tell itself apart from a "
                      f"shrunken slab (h is not identifiable next to alpha).")
    if best < ard - 0.05:
        return "O3", (f"WORSE THAN ARD: best switch AUC {best:.2f} vs ARD {ard:.2f}.")
    return "O4", (f"MIXED: best switch ({best_arm}) AUC {best:.2f} vs ARD {ard:.2f}, "
                  f"decisive {dec:.0%}.")


# ---------------------------------------------------------------------------
# Q5 -- do forced zeros pin the factors better than learned zeros?
# ---------------------------------------------------------------------------

Q5_ARMS = {
    "ARD (no switch)": dict(w="ARD"),
    "learned switch": dict(w="switch", slab="ard", pi_mode="global"),
    "frozen (true pattern)": dict(w="switch", slab="ard", frozen=True),
}


def _q5_job(args):
    level, seed, n_per_cohort, theta, arm_name, pair = args
    ds, _ = level_data(level, seed, n_per_cohort=n_per_cohort)
    K = ds.H.shape[1]
    arm = dict(Q5_ARMS[arm_name])
    if arm.pop("frozen", False):
        arm["frozen_mask"] = ds.H
    R = givens(K, pair[0], pair[1], np.deg2rad(theta))
    out = freeze_z_elbo(ds, ds.Z @ R, n_iter=200, seed=seed, **arm)
    return dict(level=level, seed=seed, N=ds.N, theta=theta, arm=arm_name,
                elbo=out["elbo"], monotone=bool(np.all(np.diff(out["trace"]) >
                                                       -1e-6 * abs(out["elbo"]))))


def q5_pair(H: np.ndarray) -> tuple[int, int]:
    """A fully shared factor and the first single-view private factor."""
    shared = int(np.where(H.sum(0) == H.shape[0])[0][0])
    private = int(np.where(H.sum(0) == 1)[0][0])
    return shared, private


def run_q5(level: str, n_per_cohort=(50, 200), thetas=(0, 10, 20, 30, 45),
           seeds=range(2)) -> pd.DataFrame:
    pair = q5_pair(LEVELS[level]["H"])
    jobs = [(level, s, n, t, a, pair) for s in seeds for n in n_per_cohort
            for t in thetas for a in Q5_ARMS]
    df = pd.DataFrame(_pool(_q5_job, jobs))
    base = df[df.theta == 0].set_index(["seed", "N", "arm"])["elbo"]
    df["gap"] = [base[(r.seed, r.N, r.arm)] - r.elbo for r in df.itertuples()]
    return df


def decide_q5(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """One outcome per arm, from the gap at the largest angle."""
    tmax = df.theta.max()
    d = df[df.theta == tmax].groupby(["arm", "N"])["gap"].mean().unstack()
    Ns = sorted(d.columns)
    n_ratio = Ns[-1] / Ns[0]
    out = {}
    for arm, row in d.iterrows():
        g_small, g_big = row[Ns[0]], row[Ns[-1]]
        ratio = g_big / g_small if g_small > 1e-9 else np.inf
        if g_big < 5:
            out[arm] = ("O1", f"FLAT: gap {g_big:.1f} nats at N={Ns[-1]}, theta={tmax}. "
                               f"Rotating shared into private costs ~nothing.")
        elif ratio >= 0.6 * n_ratio:
            out[arm] = ("O2", f"GROWS LIKE N: gap {g_small:.0f} -> {g_big:.0f} nats "
                               f"(x{ratio:.1f} for x{n_ratio:.0f} samples). The split is "
                               f"pinned at likelihood level.")
        else:
            out[arm] = ("O3", f"GROWS SLOWER THAN N: gap {g_small:.0f} -> {g_big:.0f} "
                               f"(x{ratio:.1f} for x{n_ratio:.0f} samples): a penalty, "
                               f"but not likelihood-level identification.")
    return out


# ---------------------------------------------------------------------------
# Q6 -- one sharing rate, or one per factor?
# ---------------------------------------------------------------------------


def _q6_job(args):
    level, M, seed, mode = args
    H = pattern_for(M)
    dims = 30 if level == "easy" else tuple(int(x) for x in np.linspace(80, 10, M))
    ds, K = level_data(level, seed, H=H, dims=dims,
                       loading_scale=1.0 if level == "easy" else
                       [1.0] * (H.shape[1] - 1) + [0.4])
    f = fit_switch_warm(ds, K, max_iter=400, seed=seed, slab="fixed", slab_var=1.0,
                        pi_mode=mode)
    s = score_fit(ds, f)
    idx = s["match_idx"]
    pis = f.switch.pi_summary()
    shared = H.sum(0) == M
    private = H.sum(0) == 1
    if mode == "per_factor":
        means = np.array(pis["mean"])[idx]
        sds = np.array(pis["sd"])[idx]
        sep = float(means[shared].mean() - means[private].mean())
        sd_mean = float(sds.mean())
    else:
        sep, sd_mean = np.nan, float(pis["sd"][0])
    return dict(level=level, M=M, seed=seed, mode=mode, auc_h=s["auc_h"],
                decisive_h=s["decisive_h"], pi_sep=sep, pi_sd=sd_mean,
                min_corr=s["min_corr"], monotone=s["monotone"])


def run_q6(level: str, Ms=(2, 4, 8), seeds=range(5)) -> pd.DataFrame:
    jobs = [(level, M, s, mode) for M in Ms for s in seeds
            for mode in ("global", "per_factor")]
    return pd.DataFrame(_pool(_q6_job, jobs))


def decide_q6(df: pd.DataFrame) -> tuple[str, str]:
    g = df.groupby(["M", "mode"])["auc_h"].mean().unstack()
    gain = (g["per_factor"] - g["global"])
    sd4 = df[(df["mode"] == "per_factor") & (df.M <= 4)]["pi_sd"].mean()
    if gain.max() >= 0.03:
        M_best = int(gain.idxmax())
        return "O3", (f"PER-FACTOR HELPS: AUC +{gain.max():.2f} at M={M_best}. "
                      f"Worth the extra parameters there.")
    if sd4 >= 0.15:
        return "O2", (f"USE ONE GLOBAL PI: per-factor pi_k is uncertain at M <= 4 "
                      f"(posterior sd {sd4:.2f}) and does not improve recovery "
                      f"(largest AUC gain {gain.max():+.2f}).")
    return "O1", (f"EITHER WORKS: per-factor pi_k is reasonably certain (sd {sd4:.2f}) "
                  f"but recovery is the same (largest gain {gain.max():+.2f}).")
