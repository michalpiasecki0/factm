"""
Data for the sharedness experiments.

* ``make_view_data`` -- synthetic data with a prescribed view-activity pattern
  H (M x K, 0/1): factor k loads on view m iff H[m, k] = 1. No generator in
  the repo did this before; all of them draw a full loading matrix per view.
  Optional cohort structure (sparse cohort means) supports the pi-on-cohorts
  experiment.
* ``load_covid_views`` -- COVID, M = 2 views (metabolomic, proteomic).
* ``load_four_view_cohort`` -- the long-format ``raw_data_cohort.csv`` at the
  repo root: M = 4 views (DNA, RNA, IF_dens, IF_inter), complete cases only.

Every loader returns a ``Dataset``; real datasets have no truth fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.views import Views

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Dataset:
    views: Views
    Ys: list[np.ndarray]
    labels: np.ndarray  # cohort label per sample (str)
    view_names: list[str]
    Z: np.ndarray | None = None  # truth (synthetic only)
    Ws: list[np.ndarray] | None = None
    H: np.ndarray | None = None  # true view-activity pattern, M x K
    G: np.ndarray | None = None  # true cohort-switch pattern, C x K
    info: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return len(self.Ys)

    @property
    def N(self) -> int:
        return self.Ys[0].shape[0]


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

# The default pattern: one factor in every view, one in two views, and two
# view-private factors.  Columns are factors, rows are views.
H_EASY = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
    ]
)


def pattern_for(M: int) -> np.ndarray:
    """A shared factor, a factor shared by the first two views (if M > 2) and
    one private factor per view. Works for any M >= 2."""
    cols = [np.ones(M, dtype=int)]
    if M > 2:
        c = np.zeros(M, dtype=int)
        c[:2] = 1
        cols.append(c)
    for m in range(M):
        c = np.zeros(M, dtype=int)
        c[m] = 1
        cols.append(c)
    return np.column_stack(cols)


def make_view_data(
    H: np.ndarray = H_EASY,
    n_per_cohort: int = 100,
    C: int = 3,
    dims: tuple[int, ...] | int = 30,
    noise: float = 0.5,
    rho: float = 0.0,
    loading_scale: float | np.ndarray = 1.0,
    cohort_rate: float = 0.0,
    cohort_shift: float = 2.0,
    seed: int = 0,
) -> Dataset:
    """Synthetic multi-view data with a known shared/private pattern.

    Model:  z_n | c(n) ~ N(G[c, :] * delta[c, :], I),
            y^(m)_n = W^(m) z_n + eps,  eps ~ N(0, noise^2 I),
            W^(m)_{.k} = H[m, k] * scale_k * B^(m)_{.k},
            B^(m) rows ~ N(0, R),  R = (1 - rho) I + rho 11^T.

    ``cohort_rate`` > 0 switches on a cohort mean for each (cohort, factor)
    pair independently with that probability (G); the shift is +-cohort_shift.
    With cohort_rate = 0 there is no cohort structure.
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(H, dtype=int)
    M, K = H.shape
    if isinstance(dims, int):
        dims = (dims,) * M
    if len(dims) != M:
        raise ValueError("dims must have one entry per view")
    scale = np.broadcast_to(np.asarray(loading_scale, dtype=float), (K,))

    codes = np.repeat(np.arange(C), n_per_cohort)
    G = (rng.random((C, K)) < cohort_rate).astype(int)
    delta = cohort_shift * rng.choice([-1.0, 1.0], size=(C, K))
    Z = rng.normal(size=(len(codes), K)) + (G * delta)[codes]

    L = np.linalg.cholesky((1 - rho) * np.eye(K) + rho * np.ones((K, K)))
    Ws, Ys = [], []
    for m, d in enumerate(dims):
        B = rng.normal(size=(d, K)) @ L.T
        W = B * scale * H[m]
        Ws.append(W)
        Ys.append(Z @ W.T + noise * rng.normal(size=(len(codes), d)))
    Ys = [Y - Y.mean(0) for Y in Ys]  # the FA likelihood has no intercept
    labels = np.array([f"c{c}" for c in codes])
    # What a cohort switch can actually see: the cohort mean of z relative to
    # the grand mean (the data are centred globally, so there is no other
    # reference). A cohort with no shift of its own still moves when others do.
    mu = np.vstack([Z[codes == c].mean(0) for c in range(C)]) - Z.mean(0)
    G_eff = (np.abs(mu) > cohort_shift / 2).astype(int)
    return Dataset(
        views=Views.from_list(Ys, cohorts=labels),
        Ys=Ys,
        labels=labels,
        view_names=[f"view{m + 1}" for m in range(M)],
        Z=Z,
        Ws=Ws,
        H=H,
        G=G,
        info=dict(G_eff=G_eff, cohort_means=mu, C=C, n_per_cohort=n_per_cohort, dims=list(dims), noise=noise,
                  rho=rho, loading_scale=scale.tolist(), cohort_rate=cohort_rate,
                  seed=seed),
    )


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def _top_var(X: np.ndarray, top: int | None) -> np.ndarray:
    if top is None or X.shape[1] <= top:
        return X
    idx = np.argsort(X.var(0))[::-1][:top]
    return X[:, np.sort(idx)]


def _standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


def load_covid_views(top: int = 150, cohort_mode: str = "grouped") -> Dataset:
    """COVID metabolomic + proteomic (M = 2), top-variance features per view."""
    from covid_data.load_covid import load_covid

    d = load_covid(cohort_mode=cohort_mode, standardize=True)
    Ys = [_standardise(_top_var(np.asarray(v.data, dtype=float), top))
          for v in d.views.simple]
    labels = np.asarray(d.cohorts).astype(str)
    return Dataset(
        views=Views.from_list(Ys, cohorts=labels),
        Ys=Ys,
        labels=labels,
        view_names=["metabolomic", "proteomic"],
        info=dict(source="covid", cohort_mode=cohort_mode, top=top),
    )


FOUR_VIEWS = ("DNA", "RNA", "IF_dens", "IF_inter")


def load_four_view_cohort(
    top: int = 100,
    dna_min_freq: float = 0.01,
    path: str | Path | None = None,
    max_per_cohort: int | None = None,
    seed: int = 0,
) -> Dataset:
    """The four-view cohort table at the repo root (complete cases).

    DNA is binary mutation calls (~99.6% zeros), so only mutations present in
    at least ``dna_min_freq`` of the kept samples are used (1% keeps 25
    mutations; 5% would keep only 3); every view is then
    standardised and fitted with a Gaussian likelihood. That is an
    approximation for DNA and is reported as such.
    """
    path = Path(path) if path else REPO / "raw_data_cohort.csv"
    df = pd.read_csv(path)
    have = df.drop_duplicates(["view", "sample"]).groupby("sample")["view"].nunique()
    complete = have[have == len(FOUR_VIEWS)].index
    df = df[df["sample"].isin(complete)]
    groups = df.drop_duplicates("sample").set_index("sample")["group"]

    samples = np.array(sorted(complete))
    if max_per_cohort is not None:
        rng = np.random.default_rng(seed)
        keep = []
        for g, idx in groups.loc[samples].groupby(groups.loc[samples]).groups.items():
            idx = np.array(list(idx))
            keep.extend(rng.choice(idx, size=min(max_per_cohort, len(idx)), replace=False))
        samples = np.array(sorted(keep))

    Ys, kept_counts = [], {}
    for v in FOUR_VIEWS:
        wide = (
            df[df["view"] == v]
            .pivot_table(index="sample", columns="feature", values="value", aggfunc="mean")
            .reindex(samples)
        )
        X = wide.to_numpy(dtype=float)
        if np.isnan(X).any():  # complete cases should leave none; be explicit if not
            col_mean = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), col_mean, X)
        X = X[:, X.std(0) > 0]
        if v == "DNA":
            X = X[:, X.mean(0) >= dna_min_freq]
        else:
            X = _top_var(X, top)
        kept_counts[v] = X.shape[1]
        Ys.append(_standardise(X))
    labels = groups.loc[samples].to_numpy().astype(str)
    return Dataset(
        views=Views.from_list(Ys, cohorts=labels),
        Ys=Ys,
        labels=labels,
        view_names=list(FOUR_VIEWS),
        info=dict(source="raw_data_cohort.csv", n_samples=len(samples),
                  features=kept_counts, dna_min_freq=dna_min_freq, top=top),
    )
