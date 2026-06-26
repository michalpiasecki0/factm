"""Immucan CTM / spatial visualization helpers for cohort test notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from immucan_data.loader import (
    load_cells_for_structure,
    prepare_immucan_manifests,
    resolve_table_path,
    spatial_windows_with_centers,
)


def structured_fa_view_index(model: Any, structured_idx: int = 0) -> int:
    return int(model._fa_indices[structured_idx])


def get_W_structured(model: Any, structured_idx: int = 0) -> np.ndarray:
    """
    Expected loadings E[W] for the structured FA proxy view, shape (L, K).

    FACTM injects one FA view per CTM with ``D_m = L`` (topics), not ``G`` cell types.
    Row ``l`` is the loading of latent factors onto CTM topic dimension ``l``.
    """
    fa_idx = structured_fa_view_index(model, structured_idx)
    return np.asarray(model.fa.nodelist_w[fa_idx].E_w)


def factor_topic_linkage(model: Any, structured_idx: int = 0) -> np.ndarray:
    """
    (K, L) matrix: how each FA factor loads onto each CTM topic dimension.

    This is ``E[W_structured].T`` where ``E[W_structured]`` has shape (L, K).
    """
    w = get_W_structured(model, structured_idx)
    return np.asarray(w).T


def first_sample_index_per_cohort(
    cohort_labels: np.ndarray | list[str],
    sample_ids: np.ndarray | list[str] | None = None,
) -> dict[str, int]:
    """First sample index per cohort label (order follows ``cohort_labels``)."""
    labels = np.asarray(cohort_labels)
    out: dict[str, int] = {}
    for i, c in enumerate(labels):
        key = str(c)
        if key not in out:
            out[key] = i
    return out


_MANIFEST_KWARGS = frozenset(
    {
        "cohorts",
        "exclude_cohorts",
        "sample_ids",
        "max_samples_per_cohort",
        "first_n_per_cohort",
        "intersect_samples",
        "seed",
    }
)


def resolve_sample_table_path(
    root: Path | str,
    sample_id: str,
    *,
    panel: str = "IF1",
    **manifest_kwargs: Any,
) -> Path:
    """Manifest-listed TSV path for ``sample_id`` (no scan of ``tables/``)."""
    root_path = Path(root)
    manifest_kw = {k: v for k, v in manifest_kwargs.items() if k in _MANIFEST_KWARGS}
    manifests = prepare_immucan_manifests(root_path, panels=(panel,), **manifest_kw)
    manifest = manifests[panel]
    rows = manifest.loc[manifest["sample_id"] == sample_id]
    if rows.empty:
        raise ValueError(f"sample_id {sample_id!r} not in {panel} manifest")
    return resolve_table_path(root_path, rows.iloc[0]["full_path"])


def topic_probs_and_centers_for_sample(
    model: Any,
    data: Any,
    sample_idx: int,
    *,
    structured_idx: int = 0,
    root: Path | str,
    panel: str = "IF1",
    spatial_window_size: int,
    max_windows_per_sample: int | None,
    seed: int,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    **manifest_kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Return (cells, center_idx, topic_probs) for one sample.

    ``topic_probs`` has shape (n_windows, L) and aligns with ``center_idx``.
    """
    sample_id = str(data.sample_ids[sample_idx])
    table_path = resolve_sample_table_path(
        root,
        sample_id,
        panel=panel,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        **manifest_kwargs,
    )
    cells = load_cells_for_structure(
        table_path,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
    )
    ct_map = {ct: i for i, ct in enumerate(data.celltype_vocab)}
    _, center_idx = spatial_windows_with_centers(
        cells,
        celltype_to_idx=ct_map,
        window_size=spatial_window_size,
        max_windows_per_sample=max_windows_per_sample,
        seed=seed,
    )
    probs = np.asarray(model.get_probabilities_of_topics(structured_idx)[sample_idx])
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    return cells, center_idx, probs


def plot_factor_topic_heatmap(
    model: Any,
    K: int | None = None,
    *,
    structured_idx: int = 0,
    title: str = "E[W] structured — factor × topic (K×L)",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of ``factor_topic_linkage``: rows = FA factors, cols = CTM topics."""
    mat = factor_topic_linkage(model, structured_idx)
    n_factors, l_topics = mat.shape
    if K is not None and K != n_factors:
        raise ValueError(f"K={K} but structured W has {n_factors} factor columns")
    if figsize is None:
        figsize = (0.55 * l_topics + 2.5, 0.45 * n_factors + 1.8)
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.abs(mat).max() or 1.0
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[f"T{topic_idx}" for topic_idx in range(l_topics)],
        yticklabels=[f"Z{k}" for k in range(n_factors)],
        ax=ax,
    )
    ax.set_xlabel("CTM topic (η dimension)")
    ax.set_ylabel("FA factor")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_topic_celltype_heatmap(
    model: Any,
    celltype_vocab: list[str],
    *,
    structured_idx: int = 0,
    title: str = "CTM topic × cell type (beta)",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    beta = np.asarray(model.get_topics(structured_idx))
    l_topics, _g = beta.shape
    if figsize is None:
        figsize = (max(8, 0.35 * len(celltype_vocab) + 2), 0.45 * l_topics + 1.8)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        beta,
        annot=len(celltype_vocab) <= 24,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        xticklabels=celltype_vocab,
        yticklabels=[f"T{topic_idx}" for topic_idx in range(l_topics)],
        ax=ax,
    )
    ax.set_xlabel("cell type")
    ax.set_ylabel("topic")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_spatial_topics_on_tissue(
    cells: pd.DataFrame,
    center_idx: np.ndarray,
    topic_probs: np.ndarray,
    *,
    sample_id: str,
    cohort: str,
    n_topic_panels: int = 3,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Spatial topic maps at kNN window centers on tissue coordinates.

    Column 0: dominant topic (argmax). Next columns: P(topic=l) for top topics
    by mean probability over windows.
    """
    x = cells["nucleus.x"].to_numpy(dtype=float)
    y = cells["nucleus.y"].to_numpy(dtype=float)
    cx = x[center_idx]
    cy = y[center_idx]
    l_topics = topic_probs.shape[1]
    dom = topic_probs.argmax(axis=1)
    mean_p = topic_probs.mean(axis=0)
    top_topics = np.argsort(-mean_p)[:n_topic_panels]

    ncols = 1 + len(top_topics)
    if figsize is None:
        figsize = (4.0 * ncols, 4.2)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax in axes:
        ax.scatter(x, y, s=1, c="#dddddd", alpha=0.35, linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    sc0 = axes[0].scatter(
        cx, cy, c=dom, s=28, cmap="tab10", vmin=0, vmax=max(l_topics - 1, 1)
    )
    axes[0].set_title("dominant topic")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, ticks=range(l_topics))

    for j, topic in enumerate(top_topics, start=1):
        sc = axes[j].scatter(
            cx, cy, c=topic_probs[:, topic], s=28, cmap="viridis", vmin=0, vmax=1
        )
        axes[j].set_title(f"P(T{topic})")
        plt.colorbar(sc, ax=axes[j], fraction=0.046, pad=0.04)

    fig.suptitle(f"{cohort} — {sample_id} (n_windows={len(center_idx)})", y=1.02)
    plt.tight_layout()
    return fig
