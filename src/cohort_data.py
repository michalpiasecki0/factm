"""
Build FACTM :class:`~views.Views` from long-format cohort tables.

Compatible with synthetic ``raw_data_cohort.csv`` and Immucan exports::

    view, group, feature, sample, value
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .views import Views

REQUIRED_COLUMNS = ("view", "group", "feature", "sample", "value")


def build_views_from_long_df(
    df: pd.DataFrame,
    *,
    sample_order: list[str] | None = None,
    fill_value: float = 0.0,
) -> tuple[Views, np.ndarray, dict[str, int], list[str], pd.Series]:
    """
    Pivot long-format data into per-view matrices and cohort labels.

    Returns
    -------
    views
        FACTM views; ``views.cohorts`` is set to string group labels per sample.
    cohort_labels
        String cohort label per sample (aligned to ``samples``).
    cohort_map
        Mapping group name → integer code (for plotting).
    samples
        Ordered sample ids.
    group_series
        Group label per sample.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if sample_order is None:
        samples = sorted(df["sample"].unique())
    else:
        samples = list(sample_order)

    group_series = (
        df[["sample", "group"]]
        .drop_duplicates(subset=["sample"])
        .set_index("sample")
        .reindex(samples)["group"]
    )
    if group_series.isna().any():
        missing_samples = group_series[group_series.isna()].index.tolist()
        raise ValueError(f"Samples missing group labels: {missing_samples}")

    cohort_labels = group_series.astype(str).to_numpy()
    cohort_names = list(group_series.unique())
    cohort_map = {g: i for i, g in enumerate(cohort_names)}

    view_arrays: list[np.ndarray] = []
    for view_name in sorted(df["view"].unique()):
        sub = df[df["view"] == view_name]
        pivot = sub.pivot_table(
            index="sample",
            columns="feature",
            values="value",
            aggfunc="first",
        ).reindex(samples)
        view_arrays.append(pivot.fillna(fill_value).to_numpy(dtype=float))

    views = Views.from_list(view_arrays, cohorts=cohort_labels)
    return views, cohort_labels, cohort_map, samples, group_series


def long_df_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summary for logging / notebooks."""
    return {
        "n_rows": len(df),
        "n_samples": int(df["sample"].nunique()),
        "n_views": int(df["view"].nunique()),
        "views": sorted(df["view"].unique().tolist()),
        "n_features": int(df["feature"].nunique()),
        "cohorts": df.groupby("group")["sample"].nunique().to_dict(),
    }
