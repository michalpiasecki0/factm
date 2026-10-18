# Sharedness experiments (`first_ideas_karol`)

Which factors are **shared between views** and which are **private to one view** — and can the
model actually tell? Six simple questions, answered in a fixed order:
**easy synthetic → harder synthetic → real data**.

| Notebook | What it does |
|---|---|
| `00_questions_and_data.ipynb` | States Q1–Q6 (+ Q2b) in plain words and in maths, with the possible outcomes fixed **before** any run; the difficulty ladder; the data inventory |
| `01_easy_synthetic.ipynb` | Answers every question on easy synthetic data |
| `02_harder_synthetic.ipynb` | The same on harder data (noise, unbalanced views, correlated loadings, a weak factor, surplus K) |
| `03_real_data.ipynb` | COVID (2 views) and the four-view cohort table (4 views): no truth, so stability and identification instead of recovery |
| `REPORT.md` | All outcomes in one place, what they mean for the thesis, and where proofs are needed |

Each section ends with an **Observed outcome** line picked by a rule in
`sharedness/experiments.py` (`decide_q*`), so the conclusion cannot be fitted to the plot after the fact.

## Code

| File | Contents |
|---|---|
| `sharedness/data.py` | `make_view_data` — synthetic data with a prescribed view pattern `H` (new; no repo generator could do this); COVID and four-view loaders |
| `sharedness/priors.py` | `ViewSwitchWPrior` (on/off switch per view and factor, fixed / global / per-factor / frozen π); `LearnablePiCohortPrior` (CohortFACTM prior with learned π and an optional paired γ–δ factorisation) |
| `sharedness/models.py` | building and fitting every arm; warm and truth starts; the freeze-Z rotation test |
| `sharedness/metrics.py` | variance share, sharedness entropy, factor matching, pattern AUC, leaks/misses |
| `sharedness/experiments.py`, `realdata.py` | one runner and one decision rule per question |
| `tests/test_sharedness_priors.py` | ELBO never decreases, and every new closed-form update is a stationary maximum |

Nothing in `src/` is changed except the `update_z` switch (first commit on this branch).

## Running

From `factm/`:

```bash
uv run pytest tests/test_sharedness_priors.py            # ~25 s
uv run --with nbformat python notebooks_sharedness/_build.py   # regenerate notebooks from text
uv run --with nbformat --with nbclient python notebooks_sharedness/_run.py 01   # execute one
```

Or open a notebook in VS Code with the `factm` kernel and run all cells. Fits run in a process pool
(`N_WORKERS = cpu_count - 1`). Raw results go to `notebooks_sharedness/results/*.csv`.
