"""Load the Munich ACS/CCS multi-omics dataset into a Views object for CohortFACTM.

Data source: Pekayvaz et al., Nature Medicine 30, 1696-1710 (2024).
HDF5 file: general_data_store (project root).

Key structure:
  Keys are '{patient_id}.{timepoint}', e.g. '10.2' = patient 10, timepoint 2.
  TP0 = CCS + Non-CCS patients (single visit, pooled — no label to separate them).
  TP1-4 = ACS patients sampled longitudinally during and after acute MI.

Timepoints (ACS):
  TP1: peri-interventional (during catheterization)
  TP2: 14h ±8h after intervention
  TP3: 60h ±12h after acute event
  TP4: 5-8 days after acute event (before discharge)

Three cohort modes:
  'acs_vs_control'  — 2 cohorts: control (TP0) / acs (TP1-4)
  'grouped'         — 3 cohorts: control (TP0) / acs_acute (TP1+TP2) / acs_recovery (TP3+TP4)
  'by_timepoint'    — 5 cohorts: control / acs_tp1 / acs_tp2 / acs_tp3 / acs_tp4

Default SimpleViews (patient-level bulk omics, fully aligned N=103):
  proteo  (490,)  plasma proteomics
  ck      (71,)   cytokine/chemokine 71-plex assay
  neutro  (892,)  neutrophil prime-seq bulk RNA-seq

Optional additional views (cell-composition summaries):
  leiden  (15,)   Leiden cluster proportions from scRNA-seq
  lda_20  (20,)   LDA topic proportions from scRNA-seq
  clinical (4,)   CK, CK-MB, Troponin T, Leukocytes (z-scored)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, NamedTuple

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.views import Views  # noqa: E402

CohortMode = Literal["acs_vs_control", "grouped", "by_timepoint"]

DEFAULT_STORE = _REPO_ROOT / "general_data_store"

# Modalities used as SimpleViews by default (patient-level vectors)
DEFAULT_VIEWS = ("proteo", "ck", "neutro")

# All available patient-level modalities (can be passed via extra_views)
AVAILABLE_VIEWS = ("proteo", "ck", "neutro", "leiden", "lda_20", "clinical")


class GeneralData(NamedTuple):
    views: Views
    cohorts: np.ndarray       # string labels, length N
    timepoints: np.ndarray    # integer timepoint per sample (0-4), length N
    sample_keys: np.ndarray   # HDF5 keys e.g. '10.2', length N


def _cohort_label(timepoint: int, mode: CohortMode) -> str:
    if mode == "acs_vs_control":
        return "control" if timepoint == 0 else "acs"
    if mode == "grouped":
        if timepoint == 0:
            return "control"
        if timepoint in (1, 2):
            return "acs_acute"
        return "acs_recovery"
    if mode == "by_timepoint":
        if timepoint == 0:
            return "control"
        return f"acs_tp{timepoint}"
    raise ValueError(f"Unknown cohort_mode: {mode!r}")


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def load_general_data(
    cohort_mode: CohortMode = "grouped",
    standardize: bool = True,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    store_path: str | os.PathLike[str] | None = None,
) -> GeneralData:
    """Load the Munich ACS/CCS dataset and return a Views object for FACTM training.

    Parameters
    ----------
    cohort_mode : {'acs_vs_control', 'grouped', 'by_timepoint'}
        How to partition samples into cohorts.
    standardize : bool, default True
        Z-score each feature column independently within each view.
    views : tuple of str
        Which patient-level modalities to include as SimpleViews.
        Subset of ('proteo', 'ck', 'neutro', 'leiden', 'lda_20', 'clinical').
        Default: ('proteo', 'ck', 'neutro').
    store_path : path-like, optional
        Path to the HDF5 file. Defaults to <repo_root>/general_data_store.
    """
    for v in views:
        if v not in AVAILABLE_VIEWS:
            raise ValueError(f"Unknown view {v!r}. Choose from {AVAILABLE_VIEWS}.")

    store = Path(store_path) if store_path is not None else DEFAULT_STORE
    if not store.exists():
        raise FileNotFoundError(f"HDF5 store not found: {store}")

    with h5py.File(store, "r") as f:
        # Find samples present in ALL requested views
        key_sets = [set(f[v].keys()) for v in views]
        common_keys = sorted(key_sets[0].intersection(*key_sets[1:]))

        if not common_keys:
            raise RuntimeError("No samples found across all requested views.")

        # Build arrays: one row per sample
        arrays = {v: np.stack([f[v][k][:] for k in common_keys]).astype(np.float32)
                  for v in views}

    timepoints = np.array([int(k.split(".")[1]) for k in common_keys])
    cohorts = np.array([_cohort_label(tp, cohort_mode) for tp in timepoints])

    if standardize:
        arrays = {v: _standardize(arr) for v, arr in arrays.items()}

    view_obj = Views.from_list([arrays[v] for v in views], cohorts=cohorts)

    return GeneralData(
        views=view_obj,
        cohorts=cohorts,
        timepoints=timepoints,
        sample_keys=np.array(common_keys),
    )


def feature_names(
    view: str,
    store_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return feature names for modalities that store them in metadata.

    Available for 'proteo', 'ck', 'neutro'. Returns empty list for others.
    """
    store = Path(store_path) if store_path is not None else DEFAULT_STORE
    name_map = {"proteo": "proteo", "ck": "ck", "neutro": "neutro"}
    if view not in name_map:
        return []
    with h5py.File(store, "r") as f:
        raw = f["metadata"][name_map[view]][:]
    return [n.decode() if isinstance(n, bytes) else n for n in raw]
