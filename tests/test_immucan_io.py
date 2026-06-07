from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immucan_data.loader import (
    aggregate_cell_table,
    build_immucan_long_df,
    cohort_ordinal_values,
    immucan_load_plan,
    load_cells_for_structure,
    load_immucan,
    make_cohort_labels,
    spatial_windows_from_cells,
)
from src.cohort_data import build_views_from_long_df, long_df_summary


@pytest.fixture
def mini_extraction(tmp_path: Path) -> Path:
    root = tmp_path / "extraction"
    tsv_dir = root / "tables" / "IMMU_BC1" / "IF1" / "tsv"
    tsv_dir.mkdir(parents=True)

    rows_a = "\n".join(
        [
            "sample_id\tcell.ID\tcelltype\ttissue.type\tnucleus.x\tnucleus.y\t"
            "in.ROI.tumor_tissue\tCD3\tCK\tqc_analysis_status\tflag_no_cells",
            "S1-FIXT-01-IF1-01_1\tS1\tTumor\ttumor\t0\t0\tTRUE\tFALSE\tTRUE\tpass\tFALSE",
            "S1-FIXT-01-IF1-01_2\tS1\tT\ttumor\t1\t0\tTRUE\tTRUE\tFALSE\tpass\tFALSE",
            "S1-FIXT-01-IF1-01_3\tS1\tother\tstroma\t2\t0\tFALSE\tFALSE\tFALSE\tpass\tFALSE",
        ]
    )
    rows_b = "\n".join(
        [
            "sample_id\tcell.ID\tcelltype\ttissue.type\tnucleus.x\tnucleus.y\t"
            "in.ROI.tumor_tissue\tCD3\tCK\tqc_analysis_status\tflag_no_cells",
            "S2-FIXT-01-IF1-01_1\tS2\tTumor\ttumor\t0\t0\tTRUE\tFALSE\tTRUE\tpass\tFALSE",
            "S2-FIXT-01-IF1-01_2\tS2\tTumor\ttumor\t1\t0\tTRUE\tFALSE\tTRUE\tpass\tFALSE",
        ]
    )
    (tsv_dir / "S1.tsv").write_text(rows_a)
    (tsv_dir / "S2.tsv").write_text(rows_b)

    manifest = pd.DataFrame(
        [
            {
                "measurement_id": "S1-IF1-01",
                "sample_id": "S1-FIXT-01",
                "full_path": "tables/IMMU_BC1/IF1/tsv/S1.tsv",
                "patient_id": "S1",
                "cohort": "IMMU_BC1",
                "panel": "IF1",
            },
            {
                "measurement_id": "S2-IF1-01",
                "sample_id": "S2-FIXT-01",
                "full_path": "tables/IMMU_BC1/IF1/tsv/S2.tsv",
                "patient_id": "S2",
                "cohort": "IMMU_BC1",
                "panel": "IF1",
            },
        ]
    )
    manifest.to_csv(root / "IF1_file_df_MOFA.txt", sep="\t", index=False)
    return root


def test_aggregate_cell_table() -> None:
    cells = pd.DataFrame(
        {
            "celltype": ["Tumor", "T", "Tumor"],
            "tissue.type": ["tumor", "tumor", "stroma"],
            "in.ROI.tumor_tissue": [True, True, True],
            "CD3": [False, True, False],
            "CK": [True, False, True],
            "qc_analysis_status": ["pass", "pass", "pass"],
            "flag_no_cells": [False, False, False],
        }
    )
    feats = aggregate_cell_table(cells, view_idx=0, tumor_roi_only=False)
    assert "n_cells_view0" not in feats
    assert pytest.approx(feats["celltype_Tumor_view0"]) == 2 / 3


def test_aggregate_cell_table_qc_pass_only() -> None:
    cells = pd.DataFrame(
        {
            "celltype": ["Tumor", "T", "Tumor"],
            "tissue.type": ["tumor", "tumor", "stroma"],
            "in.ROI.tumor_tissue": [True, True, True],
            "CD3": [False, True, False],
            "CK": [True, False, True],
            "qc_analysis_status": ["pass", "fail", "pass"],
            "flag_no_cells": [False, False, False],
        }
    )
    feats = aggregate_cell_table(
        cells, view_idx=0, tumor_roi_only=False, qc_pass_only=True
    )
    assert pytest.approx(feats["celltype_Tumor_view0"]) == 1.0


def test_build_immucan_long_df(mini_extraction: Path) -> None:
    df = build_immucan_long_df(mini_extraction, panels=("IF1",), progress=False)
    assert set(df.columns) == {"view", "group", "feature", "sample", "value"}
    views, labels, cohort_map, samples, _ = build_views_from_long_df(df)
    assert views.N == 2
    assert len(cohort_map) == 1
    assert long_df_summary(df)["n_samples"] == 2


def test_immucan_load_plan_caps_samples(mini_extraction: Path) -> None:
    plan = immucan_load_plan(mini_extraction, panels=("IF1",), max_samples_per_cohort=1)
    assert plan["n_tsv_files_to_read"] == 1


def test_first_n_per_cohort_manifest_order(mini_extraction: Path) -> None:
    plan = immucan_load_plan(mini_extraction, panels=("IF1",), first_n_per_cohort=1)
    assert plan["n_tsv_files_to_read"] == 1


def test_exclude_cohorts(mini_extraction: Path) -> None:
    plan = immucan_load_plan(
        mini_extraction, panels=("IF1",), exclude_cohorts=("IMMU_BC1",)
    )
    assert plan["n_tsv_files_to_read"] == 0
    plan_short = immucan_load_plan(
        mini_extraction, panels=("IF1",), exclude_cohorts=("BC1",)
    )
    assert plan_short["n_tsv_files_to_read"] == 0


def test_load_cells_for_structure_tumor_roi(mini_extraction: Path) -> None:
    path = mini_extraction / "tables/IMMU_BC1/IF1/tsv/S1.tsv"
    cells = load_cells_for_structure(path)
    assert "in.ROI.tumor_tissue" not in cells.columns
    assert len(cells) == 2


def test_spatial_windows_from_cells() -> None:
    cells = pd.DataFrame(
        {
            "celltype": ["Tumor", "T", "Tumor"],
            "nucleus.x": [0.0, 1.0, 0.1],
            "nucleus.y": [0.0, 0.0, 0.1],
        }
    )
    vocab = {"T": 0, "Tumor": 1}
    mat = spatial_windows_from_cells(
        cells, celltype_to_idx=vocab, window_size=50, max_windows_per_sample=None
    )
    assert mat.shape == (3, 2)
    assert mat.sum(axis=1).min() == 3


def test_load_immucan(mini_extraction: Path) -> None:
    data = load_immucan(
        mini_extraction,
        panels=("IF1",),
        max_samples_per_cohort=None,
        progress=False,
        cache_dir=mini_extraction / "cache",
        rebuild_cache=True,
    )
    assert data.views.N == 2
    assert data.views.cohorts is not None
    assert len(data.cohorts) == 2
    assert len(data.severity) == 2
    assert data.views.num_structured == 1
    assert data.n_topics == 10
    assert data.views.structured[0].G == len(data.celltype_vocab)


def test_cohort_modes() -> None:
    types = np.array(["BC1", "NSCLC", "SYG_BC1", "RCC"])
    assert list(make_cohort_labels(types, "by_type")) == [
        "BC1",
        "NSCLC",
        "SYG_BC1",
        "RCC",
    ]
    assert list(make_cohort_labels(types, "per_severity")) == list(
        make_cohort_labels(types, "by_type")
    )
    grouped = make_cohort_labels(types, "grouped")
    assert list(grouped) == ["breast", "lung", "breast", "kidney"]
    binary = make_cohort_labels(types, "binary")
    assert list(binary) == ["immucan", "immucan", "synergy", "immucan"]
    sev = cohort_ordinal_values(grouped, "grouped")
    assert sev.tolist() == [1.0, 2.0, 1.0, 3.0]
