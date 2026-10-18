"""Execute the sharedness notebooks in place, saving after every cell.

    uv run --with nbformat --with nbclient python notebooks_sharedness/_run.py [00 01 ...]

Because the file is rewritten after each cell, you can open the notebook in
VS Code while it runs and watch outputs appear (do not edit it meanwhile).
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
which = sys.argv[1:] or ["00", "01", "02", "03"]

for nb_path in sorted(HERE.glob("[0-9][0-9]_*.ipynb")):
    if nb_path.name[:2] not in which:
        continue
    t0 = time.time()
    nb = nbformat.read(nb_path, as_version=4)
    for c in nb.cells:  # start from a clean notebook so stale outputs never mix in
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    nbformat.write(nb, nb_path)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    done = 0

    def save(cell, cell_index, execute_reply):
        global done
        done += 1
        nbformat.write(nb, nb_path)
        print(f"{nb_path.name}: cell {done}/{n_code} done "
              f"({time.time() - t0:.0f} s)", flush=True)

    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE.parent)}},
                   on_cell_executed=save).execute()
    nbformat.write(nb, nb_path)
    print(f"{nb_path.name}: FINISHED in {time.time() - t0:.0f} s", flush=True)
