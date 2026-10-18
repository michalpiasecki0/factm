"""Execute the sharedness notebooks in place (outputs saved into the files).

    uv run --with nbformat --with nbclient python notebooks_sharedness/_run.py [00 01 ...]
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
    t = time.time()
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE.parent)}}).execute()
    nbformat.write(nb, nb_path)
    print(f"{nb_path.name}: executed in {time.time() - t:.0f} s", flush=True)
