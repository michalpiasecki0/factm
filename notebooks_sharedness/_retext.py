"""Refresh the question/markdown text of already-executed notebooks 01-04 in place
from _build.py, keeping every code cell and its outputs.

A built notebook and the on-disk notebook have the same cell sequence; only
markdown cells changed. We rebuild each notebook structure fresh, then copy the
executed code outputs across by position, checking the code sources still match.

    uv run --with nbformat python notebooks_sharedness/_retext.py
"""
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _build  # noqa: E402

BUILDERS = {
    "01_easy_synthetic.ipynb": lambda: _build.nb_synthetic("easy", "01", 5),
    "02_harder_synthetic.ipynb": lambda: _build.nb_synthetic("hard", "02", 5),
    "03_real_data.ipynb": _build.nb03,
    "04_improvements.ipynb": _build.nb04,
}


def refresh(path: Path, make):
    disk = nbformat.read(path, as_version=4)
    fresh = make()
    di = [c for c in disk.cells if c.cell_type == "code"]
    fi = [c for c in fresh.cells if c.cell_type == "code"]
    if len(di) != len(fi):
        print(f"  SKIP {path.name}: code-cell count {len(di)} != {len(fi)} (rebuild+rerun needed)")
        return False
    mismatched = [i for i, (a, b) in enumerate(zip(di, fi)) if a.source.strip() != b.source.strip()]
    if mismatched:
        print(f"  SKIP {path.name}: code cell {mismatched[0]} changed (rebuild+rerun needed)")
        return False
    # Copy executed outputs from disk onto the fresh (new-text) code cells.
    for a, b in zip(di, fi):
        b.outputs = a.outputs
        b.execution_count = a.execution_count
    fresh.metadata = disk.metadata
    nbformat.write(fresh, path)
    print(f"  {path.name}: text refreshed, {len(fi)} code cells kept with outputs")
    return True


if __name__ == "__main__":
    nbformat.write(_build.nb00(), HERE / "00_questions_and_data.ipynb")
    print("00_questions_and_data.ipynb: rebuilt (no code outputs to keep)")
    for name, make in BUILDERS.items():
        p = HERE / name
        if p.exists():
            refresh(p, make)
