"""Load Su COVID-19 multi-omics data into a Views object for CohortFACTM.

Returns metabolic + proteomic feature matrices aligned by sample_id, with cohort
labels derived from the WHO Ordinal Scale severity column. Three cohort modes:

* ``per_severity`` — one cohort per unique severity value (up to 8).
* ``grouped``      — 3 groups: 0-1 (mild) / 2-4 (moderate) / 5-7 (severe).
* ``binary``       — 2 groups: 0-2 (non_hosp) / 3-7 (hosp).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

# Ensure the repo root is importable so `from src.views import Views` works even
# when this module is imported from outside the repo (e.g. an installed kernel).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.views import Views  # noqa: E402

CohortMode = Literal["per_severity", "grouped", "binary"]

RESPONSE_FILE = "Su_COVID19_Response.csv"
METABOLIC_FILE = "Su_COVID19_RF-Imputted-Metabolic.csv"
PROTEOMIC_FILE = "Su_COVID19_RF-Imputted-Proteomic.csv"
SEVERITY_COL = "Who Ordinal Scale"


class CovidData(NamedTuple):
    views: Views
    cohorts: np.ndarray
    severity: np.ndarray
    sample_ids: np.ndarray


def _make_cohort_labels(severity: np.ndarray, mode: CohortMode) -> np.ndarray:
    s = severity.astype(int)
    if mode == "per_severity":
        return np.array([f"sev{v}" for v in s])
    if mode == "grouped":
        out = np.empty(s.shape, dtype=object)
        out[(s >= 0) & (s <= 1)] = "mild"
        out[(s >= 2) & (s <= 4)] = "moderate"
        out[(s >= 5) & (s <= 7)] = "severe"
        if (out == None).any():  # noqa: E711
            bad = np.unique(s[out == None])  # noqa: E711
            raise ValueError(f"Severity values outside 0-7 in 'grouped' mode: {bad}")
        return out.astype(str)
    if mode == "binary":
        out = np.empty(s.shape, dtype=object)
        out[(s >= 0) & (s <= 2)] = "non_hosp"
        out[(s >= 3) & (s <= 7)] = "hosp"
        if (out == None).any():  # noqa: E711
            bad = np.unique(s[out == None])  # noqa: E711
            raise ValueError(f"Severity values outside 0-7 in 'binary' mode: {bad}")
        return out.astype(str)
    raise ValueError(f"Unknown cohort_mode: {mode!r}")


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd_safe


def load_covid(
    cohort_mode: CohortMode = "grouped",
    standardize: bool = True,
    data_dir: str | os.PathLike[str] | None = None,
) -> CovidData:
    """Load the Su COVID-19 dataset and return a Views object ready for FA training.

    Parameters
    ----------
    cohort_mode : {"per_severity", "grouped", "binary"}
        How to partition samples into cohorts using the WHO Ordinal Scale.
    standardize : bool, default True
        If True, z-score each feature column independently within each view.
    data_dir : path-like, optional
        Directory containing the three CSVs. Defaults to this file's directory.
    """
    base = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent

    response = pd.read_csv(base / RESPONSE_FILE)
    response = response.set_index("sample_id")
    metab = pd.read_csv(base / METABOLIC_FILE, index_col=0)
    prot = pd.read_csv(base / PROTEOMIC_FILE, index_col=0)

    common_ids = sorted(
        set(response.index) & set(metab.index) & set(prot.index)
    )
    if not common_ids:
        raise RuntimeError("No common sample_ids across the three CSVs.")

    response = response.loc[common_ids]
    metab = metab.loc[common_ids]
    prot = prot.loc[common_ids]

    severity = response[SEVERITY_COL].to_numpy()
    cohorts = _make_cohort_labels(severity, cohort_mode)

    metab_arr = metab.to_numpy(dtype=float)
    prot_arr = prot.to_numpy(dtype=float)
    if standardize:
        metab_arr = _standardize(metab_arr)
        prot_arr = _standardize(prot_arr)

    views = Views.from_list([metab_arr, prot_arr], cohorts=cohorts)

    return CovidData(
        views=views,
        cohorts=cohorts,
        severity=severity.astype(int),
        sample_ids=np.array(common_ids),
    )
