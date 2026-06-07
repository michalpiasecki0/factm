"""
Load Immucan IF cell tables for Cohort FACTM.

Manifest-listed TSVs under ``tables/`` (via ``IF*_file_df_MOFA.txt``):

* **Simple views** — sample-level proportions / marker means (MOFA-style aggregates).
* **Structured views** — per-sample spatial niches: k-nearest-cell windows
  (default 75 cells) with cell-type count vectors for CTM (default ``L=10`` topics).

Each TSV is one sample; columns include ``nucleus.x``, ``nucleus.y``, ``celltype``.
Rows are filtered to ``in.ROI.tumor_tissue == TRUE`` (column dropped after filter).

Default root (T7 drive)::

    /Volumes/T7/immucan/results/IF/05_IF_table_extraction
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scipy.spatial import cKDTree  # noqa: E402

from src.cohort_data import build_views_from_long_df  # noqa: E402
from src.views import SimpleView, StructuredView, Views  # noqa: E402

DEFAULT_IMMUCAN_ROOT = Path("/Volumes/T7/immucan/results/IF/05_IF_table_extraction")

COHORT_SHORT_NAMES: dict[str, str] = {
    "IMMU_BC1": "BC1",
    "IMMU_NSCLC": "NSCLC",
    "IMMU_RCC": "RCC",
    "IMMU_SCCHN1": "SCCHN1",
    "SYNG_BC1": "SYG_BC1",
    "UPST_SCCHN3": "SCCHN3",
}


def _manifest_cohort_ids(cohort_names: Sequence[str]) -> frozenset[str]:
    """Resolve short labels (``BC1``) and manifest ids (``IMMU_BC1``) for filtering."""
    manifest_ids = frozenset(COHORT_SHORT_NAMES)
    short_to_manifest = {v: k for k, v in COHORT_SHORT_NAMES.items()}
    out: set[str] = set()
    for name in cohort_names:
        if name in manifest_ids:
            out.add(name)
        elif name in short_to_manifest:
            out.add(short_to_manifest[name])
        else:
            out.add(name)
    return frozenset(out)


CohortMode = Literal["by_type", "grouped", "binary", "per_severity"]

TISSUE_GROUP: dict[str, str] = {
    "BC1": "breast",
    "SYG_BC1": "breast",
    "NSCLC": "lung",
    "RCC": "kidney",
    "SCCHN1": "head_neck",
    "SCCHN3": "head_neck",
}

SYNERGY_TYPES = frozenset({"SYG_BC1"})

GROUPED_ORDINAL: dict[str, float] = {
    "breast": 1.0,
    "lung": 2.0,
    "kidney": 3.0,
    "head_neck": 4.0,
}

BINARY_ORDINAL: dict[str, float] = {
    "immucan": 0.0,
    "synergy": 1.0,
}

PANELS = ("IF1", "IF2", "IF3")

DEFAULT_STRUCTURED_TOPICS = 10
DEFAULT_SPATIAL_WINDOW_SIZE = 75
MIN_SPATIAL_WINDOW_SIZE = 50
MAX_SPATIAL_WINDOW_SIZE = 100

_SKIP_TSV_COLUMNS = frozenset(
    {
        "sample_id",
        "cell.ID",
        "nucleus.x",
        "nucleus.y",
        "cell.area",
        "phenotype",
        "TLS.ID",
        "qc_reception_status",
        "qc_staining_status",
        "qc_scanning_status",
        "qc_scanning_comment",
        "qc_analysis_comment",
    }
)

_BASE_TSV_COLUMNS = [
    "celltype",
    "tissue.type",
    "in.ROI.tumor_tissue",
    "qc_analysis_status",
    "flag_no_cells",
]


class ImmucanData(NamedTuple):
    views: Views
    cohorts: np.ndarray
    severity: np.ndarray
    sample_ids: np.ndarray
    long_df: pd.DataFrame
    celltype_vocab: list[str]
    n_topics: int


def resolve_table_path(root: Path, relative_path: str) -> Path:
    cleaned = relative_path.replace("//", "/").lstrip("/")
    return root / cleaned


def load_panel_manifest(root: Path, panel: str) -> pd.DataFrame:
    if panel not in PANELS:
        raise ValueError(f"panel must be one of {PANELS}, got {panel!r}")
    path = root / f"{panel}_file_df_MOFA.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path, sep="\t")
    required = {"sample_id", "full_path", "cohort", "panel"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return manifest


def _short_cohort_name(cohort: str) -> str:
    if cohort in COHORT_SHORT_NAMES:
        return COHORT_SHORT_NAMES[cohort]
    return cohort.replace("IMMU_", "").replace("UPST_", "")


def _normalize_cohort_mode(mode: CohortMode) -> CohortMode:
    if mode == "per_severity":
        return "by_type"
    return mode


def make_cohort_labels(type_labels: np.ndarray, mode: CohortMode) -> np.ndarray:
    """
    Partition samples into cohorts (Immucan analog of COVID ``cohort_mode``).

    * ``by_type`` / ``per_severity`` — one cohort per cancer type (BC1, NSCLC, …).
    * ``grouped`` — tissue groups: breast / lung / kidney / head_neck.
    * ``binary`` — IMMU-native vs SYNG (synergy trial) samples.
    """
    mode = _normalize_cohort_mode(mode)
    labels = type_labels.astype(str)
    if mode == "by_type":
        return labels
    if mode == "grouped":
        out = np.array([TISSUE_GROUP.get(g, g) for g in labels], dtype=object)
        unknown = sorted(set(out) - set(TISSUE_GROUP.values()))
        if unknown:
            raise ValueError(f"Unknown cancer types for 'grouped' mode: {unknown}")
        return out.astype(str)
    if mode == "binary":
        return np.array(
            ["synergy" if g in SYNERGY_TYPES else "immucan" for g in labels],
            dtype=object,
        ).astype(str)
    raise ValueError(f"Unknown cohort_mode: {mode!r}")


def cohort_ordinal_values(cohort_labels: np.ndarray, mode: CohortMode) -> np.ndarray:
    """Per-sample pseudo-ordinal scores for correlation tests (notebook parity with WHO)."""  # noqa: E501
    mode = _normalize_cohort_mode(mode)
    labels = cohort_labels.astype(str)
    if mode == "grouped":
        return np.array([GROUPED_ORDINAL[c] for c in labels], dtype=float)
    if mode == "binary":
        return np.array([BINARY_ORDINAL[c] for c in labels], dtype=float)
    uniq = sorted(np.unique(labels))
    code_map = {g: float(i) for i, g in enumerate(uniq)}
    return np.array([code_map[c] for c in labels], dtype=float)


def _filter_cells(
    cells: pd.DataFrame,
    *,
    tumor_roi_only: bool,
    qc_pass_only: bool,
) -> pd.DataFrame:
    out = cells
    if tumor_roi_only and "in.ROI.tumor_tissue" in out.columns:
        roi = out["in.ROI.tumor_tissue"]
        out = out[roi.fillna(False).astype(bool)].drop(columns=["in.ROI.tumor_tissue"])
    if qc_pass_only and "qc_analysis_status" in out.columns:
        out = out[out["qc_analysis_status"] == "pass"]
    if "flag_no_cells" in out.columns:
        out = out[~out["flag_no_cells"].fillna(False).astype(bool)]
    return out


def _marker_columns(cells: pd.DataFrame) -> list[str]:
    skip = {
        "celltype",
        "tissue.type",
        "qc_analysis_status",
        "flag_no_cells",
    }
    skip.update(c for c in cells.columns if c.startswith("in.ROI."))
    skip.update(_SKIP_TSV_COLUMNS)
    markers: list[str] = []
    for col in cells.columns:
        if col in skip:
            continue
        uniq = set(cells[col].dropna().unique())
        if (
            cells[col].dtype == bool
            or uniq.issubset({True, False})
            or uniq.issubset({0, 1})
        ):
            markers.append(col)
    return markers


def aggregate_cell_table(
    cells: pd.DataFrame,
    *,
    view_idx: int,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
) -> dict[str, float]:
    filtered = _filter_cells(
        cells, tumor_roi_only=tumor_roi_only, qc_pass_only=qc_pass_only
    )
    if len(filtered) == 0:
        return {}
    suffix = f"_view{view_idx}"
    features: dict[str, float] = {}

    for ct, frac in filtered["celltype"].value_counts(normalize=True).items():
        features[f"celltype_{ct}{suffix}"] = float(frac)
    if "tissue.type" in filtered.columns:
        for tt, frac in filtered["tissue.type"].value_counts(normalize=True).items():
            features[f"tissue_{tt}{suffix}"] = float(frac)
    for marker in _marker_columns(filtered):
        vals = filtered[marker]
        if vals.dtype == bool:
            features[f"marker_{marker}{suffix}"] = float(vals.mean())
        else:
            numeric = pd.to_numeric(vals, errors="coerce")
            if numeric.notna().any():
                features[f"marker_{marker}{suffix}"] = float(numeric.mean())
    return features


def _columns_to_read(table_path: Path) -> list[str]:
    header = pd.read_csv(table_path, sep="\t", nrows=0).columns.tolist()
    usecols = [c for c in _BASE_TSV_COLUMNS if c in header]
    extra = [
        c
        for c in header
        if c not in usecols
        and not c.startswith("in.ROI.")
        and c not in _SKIP_TSV_COLUMNS
    ]
    return usecols + extra


def extract_sample_features(
    table_path: Path,
    *,
    view_idx: int,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
) -> dict[str, float]:
    cols = _columns_to_read(table_path)
    cells = pd.read_csv(table_path, sep="\t", usecols=cols)
    return aggregate_cell_table(
        cells,
        view_idx=view_idx,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
    )


def _spatial_columns_to_read(table_path: Path) -> list[str]:
    header = pd.read_csv(table_path, sep="\t", nrows=0).columns.tolist()
    want = [
        "celltype",
        "nucleus.x",
        "nucleus.y",
        "in.ROI.tumor_tissue",
        "qc_analysis_status",
        "flag_no_cells",
    ]
    missing = [c for c in want if c not in header]
    if missing:
        raise ValueError(f"{table_path} missing columns for spatial view: {missing}")
    return want


def load_cells_for_structure(
    table_path: Path,
    *,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
) -> pd.DataFrame:
    """Load x/y positions and cell types for one sample TSV (tumor ROI only)."""
    cells = pd.read_csv(
        table_path, sep="\t", usecols=_spatial_columns_to_read(table_path)
    )
    cells = _filter_cells(
        cells, tumor_roi_only=tumor_roi_only, qc_pass_only=qc_pass_only
    )
    return cells[["celltype", "nucleus.x", "nucleus.y"]].reset_index(drop=True)


def _validate_window_size(window_size: int) -> int:
    if not MIN_SPATIAL_WINDOW_SIZE <= window_size <= MAX_SPATIAL_WINDOW_SIZE:
        raise ValueError(
            f"spatial_window_size must be in [{MIN_SPATIAL_WINDOW_SIZE}, "
            f"{MAX_SPATIAL_WINDOW_SIZE}], got {window_size}"
        )
    return window_size


def spatial_windows_from_cells(
    cells: pd.DataFrame,
    *,
    celltype_to_idx: dict[str, int],
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    max_windows_per_sample: int | None = 500,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a (n_windows, G) count matrix from cell coordinates.

    Each window is the ``window_size`` nearest cells to a center cell
    (including the center). Centers are subsampled when ``max_windows_per_sample``
    is set and there are more cells than that cap.
    """
    window_size = _validate_window_size(window_size)
    g = len(celltype_to_idx)
    n = len(cells)
    if n == 0:
        return np.zeros((0, g), dtype=float)

    k = min(window_size, n)
    coords = cells[["nucleus.x", "nucleus.y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)

    center_idx = np.arange(n, dtype=int)
    if max_windows_per_sample is not None and n > max_windows_per_sample:
        rng = np.random.default_rng(seed)
        center_idx = rng.choice(n, size=max_windows_per_sample, replace=False)

    unknown = set(cells["celltype"].unique()) - set(celltype_to_idx)
    if unknown:
        raise ValueError(f"Unknown cell types (not in vocabulary): {sorted(unknown)}")

    ct_idx = cells["celltype"].map(celltype_to_idx).to_numpy(dtype=int)
    rows: list[np.ndarray] = []
    for i in center_idx:
        _, nn_idx = tree.query(coords[i], k=k)
        nn_idx = np.atleast_1d(nn_idx)
        counts = np.bincount(ct_idx[nn_idx], minlength=g).astype(float)
        rows.append(counts)
    return np.vstack(rows)


def collect_celltype_vocabulary(
    root: Path,
    manifests: dict[str, pd.DataFrame],
    *,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
) -> list[str]:
    """Sorted global cell-type vocabulary across manifest-listed TSVs."""
    types: set[str] = set()
    for manifest in manifests.values():
        for row in manifest.itertuples(index=False):
            table_path = resolve_table_path(root, row.full_path)
            cells = load_cells_for_structure(
                table_path,
                tumor_roi_only=tumor_roi_only,
                qc_pass_only=qc_pass_only,
            )
            types.update(cells["celltype"].dropna().astype(str).unique())
    return sorted(types)


def build_immucan_structured_views(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    intersect_samples: bool = True,
    seed: int = 0,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    max_windows_per_sample: int | None = 500,
    sample_order: Sequence[str] | None = None,
    progress: bool = True,
) -> tuple[list[list[np.ndarray]], list[str]]:
    """
    One structured view per panel: list of (n_windows, G) matrices per sample.

    Sample order matches ``sample_order`` or sorted unique ids from manifests.
    """
    root_path = Path(root)
    _validate_window_size(spatial_window_size)
    manifests = prepare_immucan_manifests(
        root_path,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
    )
    vocab = collect_celltype_vocabulary(
        root_path,
        manifests,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
    )
    ct_map = {ct: i for i, ct in enumerate(vocab)}

    if sample_order is None:
        all_samples: list[str] = []
        for manifest in manifests.values():
            for sid in manifest["sample_id"]:
                if sid not in all_samples:
                    all_samples.append(sid)
        sample_order = sorted(all_samples)
    else:
        sample_order = list(sample_order)

    path_by_sample_panel: dict[tuple[str, str], Path] = {}
    for panel, manifest in manifests.items():
        for row in manifest.itertuples(index=False):
            path_by_sample_panel[(row.sample_id, panel)] = resolve_table_path(
                root_path, row.full_path
            )

    structured_per_panel: list[list[np.ndarray]] = []
    panel_items: Iterable[str] = panels
    if progress:
        try:
            from tqdm import tqdm

            panel_items = tqdm(list(panel_items), desc="structured panels")
        except ImportError:
            panel_items = list(panel_items)

    for panel in panel_items:
        per_sample: list[np.ndarray] = []
        sample_iter: Iterable[str] = sample_order
        if progress:
            try:
                from tqdm import tqdm

                sample_iter = tqdm(
                    list(sample_iter),
                    desc=f"{panel} spatial",
                    leave=False,
                )
            except ImportError:
                pass
        for sample_id in sample_iter:
            key = (sample_id, panel)
            if key not in path_by_sample_panel:
                raise KeyError(
                    f"No TSV for sample {sample_id!r} panel {panel!r} in manifest"
                )
            cells = load_cells_for_structure(
                path_by_sample_panel[key],
                tumor_roi_only=tumor_roi_only,
                qc_pass_only=qc_pass_only,
            )
            per_sample.append(
                spatial_windows_from_cells(
                    cells,
                    celltype_to_idx=ct_map,
                    window_size=spatial_window_size,
                    max_windows_per_sample=max_windows_per_sample,
                    seed=seed,
                )
            )
        structured_per_panel.append(per_sample)

    return structured_per_panel, vocab


def _subsample_manifest(
    manifest: pd.DataFrame,
    *,
    cohorts: Sequence[str] | None,
    exclude_cohorts: Sequence[str] | None,
    sample_ids: Sequence[str] | None,
    max_samples_per_cohort: int | None,
    first_n_per_cohort: int | None,
    seed: int,
) -> pd.DataFrame:
    m = manifest
    if cohorts is not None:
        m = m[m["cohort"].isin(set(cohorts))]
    if exclude_cohorts is not None:
        excluded = _manifest_cohort_ids(exclude_cohorts)
        m = m[~m["cohort"].isin(excluded)]
    if sample_ids is not None:
        m = m[m["sample_id"].isin(set(sample_ids))]
    if first_n_per_cohort is not None:
        parts = [
            grp.head(first_n_per_cohort) for _, grp in m.groupby("cohort", sort=False)
        ]
        return pd.concat(parts, ignore_index=True)
    if max_samples_per_cohort is None:
        return m
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, grp in m.groupby("cohort", sort=False):
        if len(grp) <= max_samples_per_cohort:
            parts.append(grp)
        else:
            parts.append(
                grp.sample(n=max_samples_per_cohort, random_state=rng, replace=False)
            )
    return pd.concat(parts, ignore_index=True)


def prepare_immucan_manifests(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    intersect_samples: bool = True,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Filter manifests only — no cell TSV I/O."""
    root_path = Path(root)
    panel_list = list(panels)
    manifests: dict[str, pd.DataFrame] = {}
    for panel in panel_list:
        manifests[panel] = _subsample_manifest(
            load_panel_manifest(root_path, panel),
            cohorts=cohorts,
            exclude_cohorts=exclude_cohorts,
            sample_ids=sample_ids,
            max_samples_per_cohort=max_samples_per_cohort,
            first_n_per_cohort=first_n_per_cohort,
            seed=seed,
        )
    if intersect_samples and len(panel_list) > 1:
        common: set[str] | None = None
        for m in manifests.values():
            s = set(m["sample_id"])
            common = s if common is None else common & s
        assert common is not None
        for panel in panel_list:
            manifests[panel] = manifests[panel][
                manifests[panel]["sample_id"].isin(common)
            ].reset_index(drop=True)
    return manifests


def immucan_load_plan(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Summarise how many TSVs would be read (no cell files opened)."""
    manifests = prepare_immucan_manifests(root, **kwargs)
    per_panel = {
        panel: {
            "n_samples": len(m),
            "cohorts": m["cohort"].value_counts().to_dict(),
        }
        for panel, m in manifests.items()
    }
    return {
        "panels": list(manifests.keys()),
        "n_tsv_files_to_read": sum(len(m) for m in manifests.values()),
        "per_panel": per_panel,
        "structured_views": "one CTM view per panel (spatial kNN windows from tables/)",
        "default_topics": DEFAULT_STRUCTURED_TOPICS,
        "default_window_size": DEFAULT_SPATIAL_WINDOW_SIZE,
        "note": "Manifest paths only; does not scan tables/ beyond listed TSVs.",
    }


def _subset_fingerprint_payload(
    *,
    panels: Sequence[str],
    cohorts: Sequence[str] | None,
    exclude_cohorts: Sequence[str] | None,
    sample_ids: Sequence[str] | None,
    max_samples_per_cohort: int | None,
    first_n_per_cohort: int | None,
    intersect_samples: bool,
    seed: int,
    tumor_roi_only: bool,
    qc_pass_only: bool,
    spatial_window_size: int | None = None,
    max_windows_per_sample: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "panels": list(panels),
        "cohorts": sorted(cohorts) if cohorts else None,
        "exclude_cohorts": sorted(exclude_cohorts) if exclude_cohorts else None,
        "sample_ids": sorted(sample_ids) if sample_ids else None,
        "max_samples_per_cohort": max_samples_per_cohort,
        "first_n_per_cohort": first_n_per_cohort,
        "intersect_samples": intersect_samples,
        "seed": seed,
        "tumor_roi_only": tumor_roi_only,
        "qc_pass_only": qc_pass_only,
    }
    if spatial_window_size is not None:
        payload["spatial_window_size"] = spatial_window_size
    if max_windows_per_sample is not None:
        payload["max_windows_per_sample"] = max_windows_per_sample
    return payload


def _config_fingerprint(**kwargs: Any) -> str:
    payload = _subset_fingerprint_payload(**kwargs)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def immucan_cache_path(
    cache_dir: Path | str,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    intersect_samples: bool = True,
    seed: int = 0,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
) -> Path:
    tag = _config_fingerprint(
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
    )
    panel_tag = "-".join(panels)
    return Path(cache_dir) / f"immucan_{panel_tag}_{tag}.csv"


def immucan_structured_cache_path(
    cache_dir: Path | str,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    intersect_samples: bool = True,
    seed: int = 0,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    max_windows_per_sample: int | None = 500,
) -> Path:
    tag = _config_fingerprint(
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        spatial_window_size=spatial_window_size,
        max_windows_per_sample=max_windows_per_sample,
    )
    panel_tag = "-".join(panels)
    return Path(cache_dir) / f"immucan_structured_{panel_tag}_{tag}.pkl"


def load_or_build_structured_views(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    *,
    cache_dir: Path | str = "data",
    rebuild: bool = False,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    intersect_samples: bool = True,
    seed: int = 0,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    max_windows_per_sample: int | None = 500,
    sample_order: Sequence[str] | None = None,
    progress: bool = True,
) -> tuple[list[list[np.ndarray]], list[str]]:
    cache = immucan_structured_cache_path(
        cache_dir,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        spatial_window_size=spatial_window_size,
        max_windows_per_sample=max_windows_per_sample,
    )
    if cache.is_file() and not rebuild:
        if progress:
            print(f"Immucan: loading structured cache {cache}")
        with cache.open("rb") as f:
            payload = pickle.load(f)
        return payload["panel_mats"], payload["vocab"]
    panel_mats, vocab = build_immucan_structured_views(
        root,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        spatial_window_size=spatial_window_size,
        max_windows_per_sample=max_windows_per_sample,
        sample_order=sample_order,
        progress=progress,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump({"panel_mats": panel_mats, "vocab": vocab}, f, protocol=4)
    if progress:
        print(f"Immucan: structured cache saved to {cache}")
    return panel_mats, vocab


def build_immucan_long_df(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    intersect_samples: bool = True,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """Build long-format table (view, group, feature, sample, value).

    Does not walk ``tables/``; only manifest-listed TSVs after filters.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"Immucan root not found: {root_path}. Mount T7 or pass another root."
        )

    manifests = prepare_immucan_manifests(
        root_path,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        intersect_samples=intersect_samples,
        seed=seed,
    )

    n_tsv = sum(len(m) for m in manifests.values())
    if progress:
        per_panel = {p: len(m) for p, m in manifests.items()}
        print(
            f"Immucan: reading {n_tsv} TSV files "
            f"(manifest only, not scanning tables/). Per panel: {per_panel}"
        )

    rows: list[dict[str, object]] = []
    panel_items: Iterable[tuple[str, pd.DataFrame]] = manifests.items()
    if progress:
        try:
            from tqdm import tqdm

            panel_items = tqdm(list(panel_items), desc="panels")
        except ImportError:
            pass

    for view_idx, (panel, manifest) in enumerate(panel_items):
        sample_iter = manifest.itertuples(index=False)
        if progress:
            try:
                from tqdm import tqdm

                sample_iter = tqdm(
                    sample_iter,
                    total=len(manifest),
                    desc=panel,
                    leave=False,
                )
            except ImportError:
                pass

        for row in sample_iter:
            table_path = resolve_table_path(root_path, row.full_path)
            if not table_path.is_file():
                raise FileNotFoundError(f"Missing cell table: {table_path}")
            feats = extract_sample_features(
                table_path,
                view_idx=view_idx,
                tumor_roi_only=tumor_roi_only,
                qc_pass_only=qc_pass_only,
            )
            group = _short_cohort_name(row.cohort)
            for feature, value in feats.items():
                rows.append(
                    {
                        "view": panel,
                        "group": group,
                        "feature": feature,
                        "sample": row.sample_id,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


def load_or_build_long_df(
    cache_path: Path | str | None = None,
    *,
    cache_dir: Path | str = "data",
    rebuild: bool = False,
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = None,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    intersect_samples: bool = True,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    cache = (
        Path(cache_path)
        if cache_path is not None
        else immucan_cache_path(
            cache_dir,
            panels=panels,
            cohorts=cohorts,
            exclude_cohorts=exclude_cohorts,
            sample_ids=sample_ids,
            max_samples_per_cohort=max_samples_per_cohort,
            first_n_per_cohort=first_n_per_cohort,
            intersect_samples=intersect_samples,
            seed=seed,
            tumor_roi_only=tumor_roi_only,
            qc_pass_only=qc_pass_only,
        )
    )
    if cache.is_file() and not rebuild:
        if progress:
            print(f"Immucan: loading cache {cache}")
        return pd.read_csv(cache)
    df = build_immucan_long_df(
        root,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        intersect_samples=intersect_samples,
        seed=seed,
        progress=progress,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    if progress:
        print(f"Immucan: cached to {cache}")
    return df


def _standardize_matrix(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd_safe


def load_immucan(
    root: Path | str = DEFAULT_IMMUCAN_ROOT,
    *,
    panels: Sequence[str] = ("IF1",),
    cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    max_samples_per_cohort: int | None = None,
    first_n_per_cohort: int | None = 8,
    tumor_roi_only: bool = True,
    qc_pass_only: bool = False,
    intersect_samples: bool = True,
    seed: int = 0,
    cache_dir: Path | str = "data",
    rebuild_cache: bool = False,
    standardize: bool = True,
    include_structured: bool = True,
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    max_windows_per_sample: int | None = 500,
    n_topics: int = DEFAULT_STRUCTURED_TOPICS,
    cohort_mode: CohortMode = "by_type",
    progress: bool = True,
) -> ImmucanData:
    """
    Load Immucan IF data into :class:`~views.Views` for Cohort FACTM.

    Simple views: MOFA-style aggregates per panel. Structured views (optional):
    spatial kNN windows over ``tables/`` cell positions (CTM; use ``n_topics`` in
    :class:`~model_config.StructuredViewConfig`, default 10).

    ``cohort_mode`` mirrors COVID ``load_covid(cohort_mode=...)``:
    ``per_severity`` is an alias for ``by_type`` (one cohort per cancer type).
    """
    long_df = load_or_build_long_df(
        cache_path=None,
        cache_dir=cache_dir,
        rebuild=rebuild_cache,
        root=root,
        panels=panels,
        cohorts=cohorts,
        exclude_cohorts=exclude_cohorts,
        sample_ids=sample_ids,
        max_samples_per_cohort=max_samples_per_cohort,
        first_n_per_cohort=first_n_per_cohort,
        tumor_roi_only=tumor_roi_only,
        qc_pass_only=qc_pass_only,
        intersect_samples=intersect_samples,
        seed=seed,
        progress=progress,
    )
    views, type_labels, _, samples, _ = build_views_from_long_df(long_df)
    cohort_labels = make_cohort_labels(type_labels, cohort_mode)
    severity = cohort_ordinal_values(cohort_labels, cohort_mode)
    if standardize:
        simple = [SimpleView(_standardize_matrix(sv.data)) for sv in views.simple]
    else:
        simple = list(views.simple)

    structured: list[StructuredView] = []
    vocab: list[str] = []
    if include_structured:
        panel_mats, vocab = load_or_build_structured_views(
            root,
            cache_dir=cache_dir,
            rebuild=rebuild_cache,
            panels=panels,
            cohorts=cohorts,
            exclude_cohorts=exclude_cohorts,
            sample_ids=sample_ids,
            max_samples_per_cohort=max_samples_per_cohort,
            first_n_per_cohort=first_n_per_cohort,
            intersect_samples=intersect_samples,
            seed=seed,
            tumor_roi_only=tumor_roi_only,
            qc_pass_only=qc_pass_only,
            spatial_window_size=spatial_window_size,
            max_windows_per_sample=max_windows_per_sample,
            sample_order=samples,
            progress=progress,
        )
        structured = [StructuredView(mats) for mats in panel_mats]

    views = Views(simple=simple, structured=structured, cohorts=cohort_labels)
    return ImmucanData(
        views=views,
        cohorts=cohort_labels,
        severity=severity,
        sample_ids=np.array(samples),
        long_df=long_df,
        celltype_vocab=vocab,
        n_topics=n_topics,
    )


__all__ = [
    "CohortMode",
    "ImmucanData",
    "DEFAULT_IMMUCAN_ROOT",
    "DEFAULT_SPATIAL_WINDOW_SIZE",
    "DEFAULT_STRUCTURED_TOPICS",
    "MAX_SPATIAL_WINDOW_SIZE",
    "MIN_SPATIAL_WINDOW_SIZE",
    "PANELS",
    "aggregate_cell_table",
    "build_immucan_long_df",
    "build_immucan_structured_views",
    "cohort_ordinal_values",
    "collect_celltype_vocabulary",
    "immucan_load_plan",
    "make_cohort_labels",
    "immucan_structured_cache_path",
    "load_cells_for_structure",
    "load_immucan",
    "load_or_build_long_df",
    "load_or_build_structured_views",
    "spatial_windows_from_cells",
]
