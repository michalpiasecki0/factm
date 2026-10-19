"""Builds the sharedness notebooks from plain text, so the questions and the
pre-registered outcomes live in one reviewable file.

    PYTHONPATH=<dir with nbformat> .venv/Scripts/python.exe notebooks_sharedness/_build.py

Then execute each notebook (see README in this directory).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = r'''import os, sys, warnings
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks_sharedness" else os.getcwd()
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Markdown, display
from sharedness import experiments as ex, metrics as mt
from sharedness.data import make_view_data, H_EASY, pattern_for

pd.set_option("display.precision", 3)
sns.set_theme(style="whitegrid", context="notebook")
RESULTS = os.path.join(ROOT, "notebooks_sharedness", "results")
os.makedirs(RESULTS, exist_ok=True)

def outcome(code_text, title="Observed outcome"):
    code, text = code_text
    display(Markdown(f"> **{title}: {code}** — {text}"))
    return code
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


# ---------------------------------------------------------------------------
# Question definitions. Each entry has:
#   title   - one line
#   why     - why the thesis cares (1-2 sentences)
#   setup   - dict level -> exact experimental setup (markdown, written out)
#   fit     - the models fitted, with their settings
#   measure - each number defined: formula, range, which direction is good
#   outcomes- [(code, name, "if we see", "we conclude", "so we")]
# The same entry is rendered in full in notebook 00 and, shortened, at the top
# of each section of notebooks 01-04.
# ---------------------------------------------------------------------------

Q = {}

Q["Q1"] = dict(
    title="Q1 — Can CohortFACTM learn how often cohorts differ?",
    why="""The cohort switch rate $\\pi$ is fixed at 0.5 in CohortFACTM. If it can be learned, the model
reports how many cohorts actually have their own mean; if the learned value is biased, we need to know why
before trusting any cohort read-out.""",
    setup={
        "synthetic": r"""
- **Samples.** $C=6$ cohorts, $n_c=50$ samples each, $N=300$. (This is $C$ larger and $n_c$ smaller than the
  other questions, because Q1 is about cohorts, not views.)
- **Factors.** $K$ as in this level. Each true factor score is $z_{nk}=\gamma^\ast_{ck}\delta^\ast_{ck}+\varepsilon$:
  a cohort either sits at the common centre or is shifted by $\delta^\ast=\pm2$.
- **The knob.** The true switch rate $\pi^\ast$ — the fraction of $(c,k)$ pairs that are shifted — is set to
  **0.1, 0.3, 0.6** in turn. These are the numbers the learned $\mathbb E[\pi]$ should recover.
- **What counts as a real shift.** The data are centred on the grand mean, so a switch can only see a cohort's
  mean *relative to the grand mean*. The truth we score against is therefore
  $G^{\text{eff}}_{ck}=\mathbb 1\{|\bar z_{ck}-\bar z_{\cdot k}|>1\}$, not the raw $\gamma^\ast$. $G^{\text{eff}}$ can differ
  from $\pi^\ast$ (e.g. when one cohort shifts, the others move a little the other way).
- **Repeats.** 5 seeds per rate.""",
        "real": r"""
- **Data.** The real cohorts of each dataset (COVID severity groups; the 6 cancer types of the four-view table).
- **No truth.** There is no $\pi^\ast$. We therefore ask only whether the learned $\mathbb E[\pi]$ lands clearly
  inside $(0,1)$ with some switches off (as in the synthetic O1 case) or saturates near 1 (as in O2).
- **Repeats.** 3 seeds.""",
    },
    fit=r"""Three cohort priors, all with ARD loadings:
1. **$\pi=0.5$ fixed** — CohortFACTM as it is today.
2. **$\pi$ learned, current factorisation** $q(\gamma)q(\delta)$ — $q(\pi)=\mathrm{Beta}(1+\sum_{c,k}\mathbb E\gamma_{ck},\,1+CK-\sum\mathbb E\gamma_{ck})$.
3. **$\pi$ learned, paired factorisation** $q(\gamma)q(\delta\mid\gamma)$ — the fix proposed in the model document.""",
    measure=r"""
- **$\mathbb E[\pi\mid Y]$** vs the true rate (effective rate on synthetic data). Good = close to the diagonal.
- **$\gamma$ AUC** $=P(\mathbb E\gamma_{ck}>\mathbb E\gamma_{c'k'}\mid G^{\text{eff}}_{ck}=1,\,G^{\text{eff}}_{c'k'}=0)$: does
  $\mathbb E\gamma$ rank the real shifts above the non-shifts? $1$ perfect, $0.5$ chance.
- **null $\mathbb E\gamma$** = mean $\mathbb E\gamma$ where the truth is 0 (should be low); **alt** = where it is 1.""",
    outcomes=[
        ("O1", "Learnable", r"max $|\mathbb E\pi-\pi^\ast|<0.15$ and $\mathbb E\pi$ rises with $\pi^\ast$ (slope $>0.5$)",
         "the data carry the rate", "keep $\\pi$ learnable"),
        ("O2", "Drifts to 1", r"$\mathbb E\pi>0.8$ whatever $\pi^\ast$ is",
         "switches that should be off are not pushed off", "the factorisation, not the formula, is at fault — try the paired one"),
        ("O3", "Prior-dominated", r"$\mathbb E\pi$ moves $<0.2$ across all $\pi^\ast$",
         "too few switches ($CK$) to learn from", "leave $\\pi$ fixed"),
        ("O4", "Partial", "anything else (moves with the truth but biased)",
         "the rate is partly recoverable", "report the bias and prefer the arm with the smallest one"),
    ],
)

Q["Q2"] = dict(
    title="Q2 — Does a plain fit already show which factors are shared?",
    why="""If an ordinary fit already reveals the shared/private pattern, no new model is needed to *see* it —
only (later) to *identify* it. This tells us whether the hard part is reading the pattern or finding the factors.""",
    setup={
        "synthetic": r"""
- **Samples / views / noise / loadings** exactly as this level's data (stated at the top of the notebook: easy
  = 4 views $\times$ 30 features, $N=300$, noise 0.5, independent loadings; hard = 4 views of 80/40/15/10
  features, $N=150$, noise 1.2, correlated loadings, one weak factor, 2 surplus fitted factors).
- **True pattern $H$** (views $\times$ factors) is the matrix drawn at the top of notebook 00. We know it, so we
  can score recovery.""",
        "real": None,
    },
    fit=r"""Two loadings priors: **no prior** ($w\sim\mathcal N(0,1)$) and **ARD**. No cohort structure.""",
    measure=r"""
- **Variance share** $r_{mk}=\lVert\mathbb E z_{\cdot k}\,\mathbb E w^{(m)\top}_{\cdot k}\rVert_F^2/\lVert Y^{(m)}\rVert_F^2$:
  the fraction of view $m$'s variance carried by factor $k$. Normalise per factor to activity $a_{mk}\in[0,1]$.
- **Pattern AUC** $=P(a_{mk}>a_{m'k'}\mid H_{mk}=1,\,H_{m'k'}=0)$, after matching fitted to true factors
  (Hungarian on $|\mathrm{corr}|$). Threshold-free; $1$ perfect, $0.5$ chance. We report it with and without ARD.""",
    outcomes=[
        ("O1", "Yes, any fit", r"AUC $\ge0.95$ with and without ARD",
         "sharedness is a read-out of the fit", "no new model needed to see the pattern"),
        ("O2", "Yes, with ARD", r"AUC $\ge0.95$ only with ARD",
         "ARD is what makes the pattern readable", "use ARD loadings when reading sharedness"),
        ("O3", "Partly", r"best AUC in $[0.75,0.95)$",
         "some cells are read wrongly", "inspect leaks (private read as shared) vs misses"),
        ("O4", "No", r"best AUC $<0.75$",
         "the fit does not expose the pattern", "the bottleneck is elsewhere (see Q2b)"),
    ],
)

Q["Q2b"] = dict(
    title="Q2b — When the fit lands on the wrong factors, is it the optimiser or the model?",
    why="""If the model's objective (ELBO) *prefers* the true factors but the algorithm fails to reach them, the
fix is better optimisation (cheap). If the ELBO itself prefers a rotated answer, the fix is new theory (hard).
Everything downstream depends on which it is.""",
    setup={
        "synthetic": r"""
- **Same data as Q2.** Look only at seeds where the ordinary ARD fit (from the usual PCA start) missed the true
  factors: $\min_k|\mathrm{corr}(\hat z_k,z^\ast_k)|<0.9$.
- For each such seed, **refit a second time starting exactly at the truth** $(Z^\ast,W^\ast)$, and compare the two
  fits' final ELBO.""",
        "real": None,
    },
    fit=r"""ARD loadings from two starts: the usual PCA start, and the true $(Z^\ast,W^\ast)$.""",
    measure=r"""
- **$\Delta_{\text{start}}=\mathrm{ELBO}_{\text{truth start}}-\mathrm{ELBO}_{\text{PCA start}}$** (nats). Positive means the
  model scores the truth higher than the rotated answer it actually found.
- Whether the truth-started fit **keeps** the true axes (does not drift back to the rotated answer).""",
    outcomes=[
        ("O0", "Not needed", "every ARD fit already found the true axes", "no problem to diagnose", "move on"),
        ("O1", "Optimiser", r"$\Delta_{\text{start}}>10$ nats and the truth start keeps the true axes",
         "the ELBO prefers the truth; the algorithm just doesn't reach it", "better initialisation / multi-start fixes it"),
        ("O2", "Model", r"$\Delta_{\text{start}}<-10$, or the truth start drifts away",
         "the ELBO itself prefers a rotation", "an identification result is needed (mean-field artefact regime)"),
        ("O3", "Flat", r"$|\Delta_{\text{start}}|\le10$",
         "the two solutions are indistinguishable to the ELBO", "the rotation is (nearly) unidentified"),
    ],
)

Q["Q3"] = dict(
    title="Q3 — How many views are needed before shared-vs-private is a meaningful question?",
    why="""With 2 views there are only 3 possible patterns; with 4 there are 15. We need to know whether few
views make the pattern merely *coarse* or actually *hard to detect* — this decides whether COVID ($M=2$) is a
usable test bed.""",
    setup={
        "synthetic": r"""
- **A ladder of view counts** $M\in\{2,3,4,6\}$. For each $M$ the true pattern has one fully shared factor, one
  factor in views 1–2 (when $M>2$), and one factor private to each view.
- Feature counts per view: 30 (easy) or spread 80…10 (hard); other settings as this level.
- **Repeats.** 5 seeds per $M$.
- Plus a **no-fitting table**: the posterior sd of a per-factor rate learned from only $M$ switches,
  $\mathrm{Beta}(1+s_k,1+M-s_k)$ — shows how uncertain $\pi_k$ must be when $M$ is small, before any experiment.""",
        "real": r"""No fitting: the datasets are fixed. We only note the pattern count ($2^M-1$ = 3 for COVID, 15 for
the four-view table) and recall the synthetic answer for whether $M=2$ suffices to detect a pattern.""",
    },
    fit=r"""ARD loadings, no cohort structure, one fit per $(M,\text{seed})$.""",
    measure=r"""**AUC of ARD activity against $H$** (as in Q2), averaged over seeds, for each $M$. Good = high and
flat across $M$; a rise with $M$ means more views help detection.""",
    outcomes=[
        ("O1", "M does not limit detection", r"AUC $\ge0.95$ at every $M$",
         "even $M=2$ detects the pattern, just coarsely", "COVID is usable; more views only add resolution"),
        ("O2", "More views help", r"AUC rises by $\ge0.05$ from the smallest to the largest $M$",
         "detection improves with views", "prefer the four-view data over COVID"),
        ("O3", "More views hurt", r"AUC falls by $\ge0.05$",
         "more private factors split the signal", "cap the number of views / factors"),
        ("O4", "Flat but imperfect", "otherwise",
         "detection is limited by something other than $M$", "look at noise / factor strength instead"),
    ],
)

Q["Q4"] = dict(
    title="Q4 — Does an explicit on/off view switch beat ARD's soft dials?",
    why="""ARD turns a factor's loadings in a view *down* but never fully *off*, so any yes/no call needs an
arbitrary cutoff. A real switch $h_{mk}$ would give a direct answer — if it is identifiable.""",
    setup={
        "synthetic": r"""
- **Same data as Q2** (this level). We know $H$, so we can check the switch's calls.
- The switch is $w^{(m)}_{\cdot k}=h_{mk}V^{(m)}_{\cdot k}$, $h_{mk}\sim\mathrm{Bern}(\pi)$, one Bernoulli per (view, factor).""",
        "real": r"""Each dataset, 3 seeds. No truth: we report how many switches stay on and whether seeds agree,
and we compare the ARD slab against a fixed slab.""",
    },
    fit=r"""Four arms: **ARD** (reference, no switch); **switch, cold start**; **switch, warm start** (started from a
converged ARD fit); **switch, warm start + fixed slab** (slab precision $\alpha\equiv1$ instead of learned ARD).""",
    measure=r"""
- **Switch AUC** of $\mathbb E[h]$ against $H$ (as in Q2).
- **Decisiveness** = share of $\mathbb E[h_{mk}]$ outside $(0.1,0.9)$ — are the calls confident?
- **Switches on** = share with $\mathbb E[h]>0.5$ (true value 50% in the synthetic pattern).
- **Leaks** (private called shared) and **misses** (shared called absent).""",
    outcomes=[
        ("O1", "Cleaner", r"best switch AUC $\ge$ ARD$-0.02$, $\ge0.9$, and $\ge90\%$ decisive",
         "the switch gives sharper correct calls", "worth adding as a diagnostic"),
        ("O2", "Switch stays on", r"$\ge95\%$ of switches on in the best arm",
         "$h$ cannot tell itself apart from a shrunk slab ($\\alpha\\to\\infty$ and $h=0$ look the same)",
         "drop the switch or use a fixed slab"),
        ("O3", "Worse than ARD", r"best switch AUC $<$ ARD$-0.05$",
         "the switch harms recovery", "keep ARD"),
        ("O4", "Mixed", "otherwise", "it depends on the slab", "report per arm"),
    ],
)

Q["Q5"] = dict(
    title="Q5 — Do forced zeros pin the factors down better than learned zeros?",
    why="""This is the headline. A *frozen* zero (‘factor 2 never appears in view 1’) should make a
shared↔private rotation impossible; a *learned* zero can be switched back on. If the frozen penalty grows with
$N$ and the learned one does not, only a prescribed pattern identifies the split.""",
    setup={
        "synthetic": r"""
- **The rotation.** Pick the fully shared factor and one view-private factor. Rotate one into the other by an
  angle $\theta\in\{0,10,20,30,45\}^\circ$: set the factor means to $Z^\ast G(\theta)$.
- **Freeze those means** and re-optimise *everything else* (loadings, switches, noise, cohort parameters).
- **Two sample sizes.** $N=150$ ($n_c=50$) and $N=600$ ($n_c=200$) — a $\times4$ increase, to see whether the
  penalty scales with $N$. With $z\sim\mathcal N(0,I)$ the factor-prior part of the ELBO is unchanged by the
  rotation, so any gap comes from the loadings side.
- **Repeats.** 2 seeds (the signal is large).""",
        "real": r"""Same idea, but the reference factors are the ARD fit's own (no truth): rotate its most shared
factor into its most private one. The frozen arm fixes the pattern read off that fit. "Grows with $N$" is checked
on a random quarter of the samples vs all of them.""",
    },
    fit=r"""Three arms, each re-optimised at every angle: **ARD** (no view structure); **learned switch**;
**frozen pattern** (the true $H$ on synthetic data; the read-off pattern on real data).""",
    measure=r"""**$\Delta(\theta)=\mathrm{ELBO}(0)-\mathrm{ELBO}(\theta)$** (nats): how hard the model resists the rotation.
We read its **growth from $N$ to $4N$**: a ratio near $4$ means the penalty scales with $N$ (likelihood-level
identification); a ratio near $1$ means a fixed, prior-level preference.""",
    outcomes=[
        ("O1", "Flat", r"$\Delta<5$ nats at the larger $N$",
         "rotating shared into private is essentially free", "no identification from this mechanism"),
        ("O2", "Grows like N", r"gap ratio $\ge0.6\times$ the sample-size ratio",
         "the split is pinned at the likelihood level", "a prescribed pattern identifies it (the `lem:zeros` regime)"),
        ("O3", "Grows slower than N", "otherwise",
         "there is a real penalty but it does not scale with $N$", "a preference, not identification"),
    ],
)

Q["Q6"] = dict(
    title="Q6 — One sharing rate for the whole model, or one per factor?",
    why="""A single $\\pi$ answers ‘are this dataset's factors mostly shared?’; a per-factor $\\pi_k$ answers ‘is
factor $k$ shared?’ — but $\\pi_k$ is learned from only $M$ switches. We need to know if the finer version is worth
the extra uncertainty.""",
    setup={
        "synthetic": r"""
- **View counts** $M\in\{2,4,8\}$, same pattern family as Q3, this level's settings.
- Fit the view switch twice: with **one global $\pi$** and with **one $\pi_k$ per factor**.
- **Repeats.** 5 seeds per $M$.""",
        "real": r"""Each dataset, 3 seeds: compare the two modes' switch calls and the spread of $\pi_k$.""",
    },
    fit=r"""View switch (fixed slab), global $q(\pi)=\mathrm{Beta}(1+\sum_{m,k}\mathbb E h_{mk},\,1+MK-\sum\mathbb E h_{mk})$
vs per-factor $q(\pi_k)=\mathrm{Beta}(1+\sum_m\mathbb E h_{mk},\,1+M-\sum_m\mathbb E h_{mk})$.""",
    measure=r"""
- **Switch AUC** in both modes (does per-factor recover the pattern any better?).
- **$\mathrm{sd}(\pi_k)$** — how uncertain the per-factor rate is (large when $M$ is small).
- **Separation** = mean $\mathbb E[\pi_k]$ over shared factors minus over private factors.""",
    outcomes=[
        ("O1", "Either works", r"$\mathrm{sd}(\pi_k)<0.15$ at $M\le4$ and no AUC gain",
         "per-factor is certain enough and costs nothing", "choose by interpretability"),
        ("O2", "Use one global $\\pi$", r"$\mathrm{sd}(\pi_k)\ge0.15$ at $M\le4$ and no AUC gain",
         "per-factor rates are noise at realistic $M$", "use a single $\\pi$"),
        ("O3", "Per-factor helps", r"switch AUC gain $\ge0.03$ at some $M$",
         "the finer rate improves recovery there", "keep $\\pi_k$ where it helps"),
    ],
)


def outcomes_table(q):
    rows = ["| Outcome | If we see | It would mean | So we |", "|---|---|---|---|"]
    for c, name, see, means, do in q["outcomes"]:
        rows.append(f"| **{c}** {name} | {see} | {means} | {do} |")
    return "\n".join(rows)


def _setup_for(q, level):
    """The level-specific setup block: 'synthetic' for 01/02/04, 'real' for 03."""
    s = q.get("setup", {})
    if level in ("easy", "hard", "synthetic"):
        return s.get("synthetic")
    return s.get("real", s.get("synthetic"))


def question_cell(key, short=False, level="synthetic"):
    q = Q[key]
    setup = _setup_for(q, level)
    def labelled(label, body):
        body = body.strip("\n")
        sep = "\n\n" if body.lstrip().startswith(("-", "1.", "$$")) else " "
        return f"**{label}**{sep}{body}"

    parts = [f"## {q['title']}", labelled("Why we ask.", q["why"])]
    if setup:
        parts.append(labelled("Setup.", setup))
    if q.get("fit"):
        parts.append(labelled("What we fit.", q["fit"]))
    if q.get("measure"):
        parts.append(labelled("What we measure.", q["measure"]))
    parts.append("**How to read the answer** (the rule the code applies, fixed before the run):\n\n"
                 + outcomes_table(q))
    if short:
        parts.append("*Thresholds live in `sharedness/experiments.py`; the full statement is in notebook 00.*")
    return md("\n\n".join(parts))


# ---------------------------------------------------------------------------
# Notebook 00
# ---------------------------------------------------------------------------


def nb00():
    cells = [md(r"""
# 00 · Questions, outcomes and data

This notebook states the six questions *before* any result. Notebooks 01–03 answer them in a fixed
order — **easy synthetic → harder synthetic → real data** — and each section ends with an
*Observed outcome* line chosen by a rule written in code here, not by reading the plot.

**The setting in one line.** One set of factor scores $Z$ ($N\times K$) drives every view:
$Y^{(m)}\approx Z\,W^{(m)\top}$. Factor $k$ is **shared** if its loading column $W^{(m)}_{\cdot k}$ is non-zero in
every view, **private** to view $m$ if only there. The *pattern* of factor $k$ is the 0/1 vector
$H_{\cdot k}$ over views. "Shared" in these notebooks always means *shared between views*, never
between cohorts.

**Development note (honesty).** Before these notebooks were written, a pilot on easy data showed
that (a) a learned cohort $\pi$ drifts towards 1 and (b) a cold-started view switch stays on.
Both observations shaped two *extra* arms — the paired cohort factorisation (Q1) and the warm-started
switch (Q4) — but the original arms are kept and the outcomes below were fixed before the runs
reported in notebooks 01–03.
"""), code(SETUP)]
    for k in ("Q1", "Q2", "Q2b", "Q3", "Q4", "Q5", "Q6"):
        cells.append(question_cell(k))
    cells += [
        md(r"""
## The difficulty ladder

| | Easy synthetic | Harder synthetic | Real data |
|---|---|---|---|
| views $M$ | 4 × 30 features | 4 views, 80 / 40 / 15 / 10 features (unbalanced) | COVID: 2 · four-view cohort table: 4 |
| samples $N$ | 300 (3 cohorts) | 150 (3 cohorts) | 378 · 1038 |
| noise sd | 0.5 | 1.2 | unknown |
| loadings | independent | correlated, $\rho=0.5$ | unknown |
| factors | 4, all strong | 6, one weak (scale 0.4) | unknown |
| $K$ fitted | = true | true + 2 (surplus factors) | 5 / 6 |
| truth | known | known | none → stability and identification only |

Why these knobs: correlated loadings make the mean-field-preferred axes differ from the truth; unbalanced
views let the big view dominate; a weak private factor is the hardest cell to call; surplus factors test
whether the fit invents patterns.
"""),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (name, H) in zip(axes, [("easy", ex.LEVELS["easy"]["H"]), ("hard", ex.LEVELS["hard"]["H"])]):
    sns.heatmap(H, ax=ax, cmap="Blues", cbar=False, annot=True, linewidths=.5,
                xticklabels=[f"f{k}" for k in range(H.shape[1])],
                yticklabels=[f"view {m+1}" for m in range(H.shape[0])])
    ax.set_title(f"True pattern H — {name}  (1 = factor used by view)")
plt.tight_layout(); plt.show()
'''),
        md("### Q3, the part that needs no fitting\n\nHow sure can a per-factor rate be, even with a perfectly observed pattern?"),
        code(r'''
tab = ex.q3_counting_table()
tab.to_csv(os.path.join(RESULTS, "q3_counting.csv"), index=False)
tab
'''),
        md("""
Reading the table: at $M=2$ a shared factor's $\\pi_k$ has posterior mean $0.75$ with sd $0.19$ and a private one
$0.50$ with sd $0.22$ — the two overlap almost completely. Per-factor rates only separate around $M\\ge6$.

## Data inventory (what the real-data notebook can and cannot ask)
"""),
        code(r'''
from sharedness.data import load_covid_views, load_four_view_cohort
cov = load_covid_views(top=150, cohort_mode="grouped")
fv = load_four_view_cohort(top=100)
inv = pd.DataFrame([
    dict(dataset="COVID", M=cov.M, N=cov.N, cohorts=len(np.unique(cov.labels)),
         features=[Y.shape[1] for Y in cov.Ys], views=cov.view_names, patterns=2**cov.M - 1),
    dict(dataset="four-view cohort table", M=fv.M, N=fv.N, cohorts=len(np.unique(fv.labels)),
         features=[Y.shape[1] for Y in fv.Ys], views=fv.view_names, patterns=2**fv.M - 1),
])
inv
'''),
        md("""
**Caveats carried into notebook 03.** The four-view table is the IMMUCAN-style long table in the repo
root, restricted to the 1038 samples that have all four views (6 cohorts; SCCHN3 has no complete sample).
DNA is binary mutation calls (≈99.6 % zeros): only mutations present in ≥1 % of samples are kept (25 mutations; at 5 % only 3 would survive), and
they are fitted with a Gaussian likelihood, which is an approximation. The IMMUCAN loader in
`immucan_data/` needs a Mac drive path and is not used.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


# ---------------------------------------------------------------------------
# Notebooks 01 and 02: the same questions at two difficulty levels
# ---------------------------------------------------------------------------


def nb_synthetic(level: str, number: str, seeds: int):
    title = "Easy synthetic data" if level == "easy" else "Harder synthetic data"
    cells = [md(f"""
# {number} · {title}

Answers Q1–Q6 on **{level}** data ({seeds} seeds per cell; ladder in notebook 00). Each section:
question and outcomes → run → table/plot → the outcome picked by the pre-registered rule.
All raw results are saved to `notebooks_sharedness/results/{level}_*.csv`.
"""), code(SETUP + f'\nLEVEL = "{level}"\nSEEDS = range({seeds})\n')]

    # Q1
    cells += [question_cell("Q1", short=True, level=level), code(r'''
q1 = ex.run_q1(LEVEL, seeds=SEEDS)
q1.to_csv(os.path.join(RESULTS, f"{LEVEL}_q1.csv"), index=False)
summary = (q1.groupby(["arm", "rate"])
             [["true_rate", "E_pi", "sd_pi", "mean_gamma_null", "mean_gamma_alt", "auc_gamma", "monotone"]]
             .mean())
summary
'''), code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for arm, d in q1.groupby("arm"):
    g = d.groupby("rate")[["true_rate", "E_pi"]].mean()
    axes[0].plot(g["true_rate"], g["E_pi"], "o-", label=arm)
    g2 = d.groupby("rate")["mean_gamma_null"].mean()
    axes[1].plot(g.index, g2, "o-", label=arm)
axes[0].plot([0, 1], [0, 1], "k:", lw=1); axes[0].set_xlabel("true (effective) switch rate")
axes[0].set_ylabel("learned E[π]"); axes[0].set_title("Does the learned rate follow the truth?")
axes[1].set_xlabel("generating rate π*"); axes[1].set_ylabel("mean E[γ] where the truth is 0")
axes[1].set_title("Switches that should be off"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''), code(r'''
for arm in ["learned pi (current q)", "learned pi (paired q)"]:
    outcome(ex.decide_q1(q1, arm), title=f"Observed outcome [{arm}]")
''')]

    # Q2
    cells += [question_cell("Q2", short=True, level=level), code(r'''
q2 = ex.run_q2(LEVEL, seeds=SEEDS)
q2[["arm", "seed", "auc_r", "auc_ard", "leak_r", "miss_r", "exact_r", "min_corr", "surplus_share",
    "monotone", "n_iter"]].to_csv(os.path.join(RESULTS, f"{LEVEL}_q2.csv"), index=False)
q2.groupby("arm")[["auc_r", "auc_ard", "leak_r", "miss_r", "exact_r", "min_corr", "surplus_share"]].mean()
'''), code(r'''
# One fit, drawn: the variance-share matrix next to the truth.
row = q2[(q2.arm == "ARD") & (q2.seed == 0)].iloc[0]
ds0, _ = ex.level_data(LEVEL, 0)
r = row["r"][:, row["match_idx"]]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
sns.heatmap(ds0.H, ax=axes[0], cmap="Blues", cbar=False, annot=True); axes[0].set_title("truth H")
sns.heatmap(r, ax=axes[1], cmap="Blues", annot=True, fmt=".2f"); axes[1].set_title("variance share r (ARD fit)")
S_true = mt.sharedness_entropy(mt.variance_share(ds0.Z, ds0.Ws, ds0.Ys))
S_fit = row["S_fit"][row["match_idx"]]
axes[2].bar(np.arange(len(S_true)) - .2, S_true, .4, label="truth")
axes[2].bar(np.arange(len(S_fit)) + .2, S_fit, .4, label="fit")
axes[2].set_title("sharedness S_k (1 = all views)"); axes[2].set_xlabel("factor"); axes[2].legend()
plt.tight_layout(); plt.show()
'''), code("outcome(ex.decide_q2(q2))")]

    cells += [question_cell("Q2b", short=True, level=level), code(r'''
piv = q2[q2.arm.isin(["ARD", "ARD_truthstart"])].pivot(index="seed", columns="arm",
                                                       values=["elbo", "min_corr"])
piv[("elbo", "truth − PCA start")] = piv[("elbo", "ARD_truthstart")] - piv[("elbo", "ARD")]
piv
'''), code("outcome(ex.decide_q2b(q2))")]

    # Q3
    cells += [question_cell("Q3", short=True, level=level), code(r'''
q3 = ex.run_q3(LEVEL, seeds=SEEDS)
q3[["M", "seed", "auc_ard", "auc_r", "exact_r", "min_corr"]].to_csv(
    os.path.join(RESULTS, f"{LEVEL}_q3.csv"), index=False)
q3.groupby("M")[["auc_ard", "auc_r", "exact_r", "min_corr"]].mean()
'''), code("outcome(ex.decide_q3(q3))")]

    # Q4
    cells += [question_cell("Q4", short=True, level=level), code(r'''
q4 = ex.run_q4(LEVEL, seeds=SEEDS)
q4["on_share"] = q4["E_h"].map(lambda x: np.nan if not isinstance(x, np.ndarray) else float(np.mean(x > .5)))
q4["pi_mean"] = q4["pi"].map(lambda x: np.nan if not isinstance(x, dict) else x["mean"][0])
cols = ["arm", "seed", "auc_ard", "auc_h", "decisive_h", "leak_h", "miss_h", "on_share", "pi_mean",
        "min_corr", "elbo", "monotone", "n_iter"]
q4[[c for c in cols if c in q4]].to_csv(os.path.join(RESULTS, f"{LEVEL}_q4.csv"), index=False)
q4.groupby("arm")[[c for c in cols[2:] if c in q4]].mean()
'''), code(r'''
row = q4[(q4.arm == "switch_warm") & (q4.seed == 0)].iloc[0]
ds0, _ = ex.level_data(LEVEL, 0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
sns.heatmap(ds0.H, ax=axes[0], cmap="Blues", cbar=False, annot=True); axes[0].set_title("truth H")
ard_row = q4[(q4.arm == "ARD") & (q4.seed == 0)].iloc[0]
sns.heatmap(mt.activity(ard_row["ard"])[:, ard_row["match_idx"]], ax=axes[1], cmap="Blues",
            annot=True, fmt=".2f", vmin=0, vmax=1); axes[1].set_title("ARD activity (dial)")
sns.heatmap(row["E_h"][:, row["match_idx"]], ax=axes[2], cmap="Blues", annot=True, fmt=".2f",
            vmin=0, vmax=1); axes[2].set_title("switch E[h] (warm)")
plt.tight_layout(); plt.show()
'''), code("outcome(ex.decide_q4(q4))")]

    # Q5
    cells += [question_cell("Q5", short=True, level=level), code(r'''
q5 = ex.run_q5(LEVEL, seeds=range(2))
q5.to_csv(os.path.join(RESULTS, f"{LEVEL}_q5.csv"), index=False)
print("all freeze fits monotone:", bool(q5.monotone.all()))
q5.groupby(["arm", "N", "theta"])["gap"].mean().unstack("theta")
'''), code(r'''
fig, axes = plt.subplots(1, len(q5.arm.unique()), figsize=(13, 3.6), sharey=False)
for ax, (arm, d) in zip(axes, q5.groupby("arm")):
    for N, dn in d.groupby("N"):
        g = dn.groupby("theta")["gap"].mean()
        ax.plot(g.index, g.values, "o-", label=f"N={N}")
    ax.set_title(arm); ax.set_xlabel("rotation θ (deg)"); ax.axhline(0, color="k", lw=.5)
axes[0].set_ylabel("ELBO(0) − ELBO(θ)  [nats]"); axes[-1].legend()
plt.tight_layout(); plt.show()
'''), code(r'''
for arm, oc in ex.decide_q5(q5).items():
    outcome(oc, title=f"Observed outcome [{arm}]")
''')]

    # Q6
    cells += [question_cell("Q6", short=True, level=level), code(r'''
q6 = ex.run_q6(LEVEL, seeds=SEEDS)
q6.to_csv(os.path.join(RESULTS, f"{LEVEL}_q6.csv"), index=False)
q6.groupby(["M", "mode"])[["auc_h", "decisive_h", "pi_sd", "pi_sep", "min_corr"]].mean()
'''), code("outcome(ex.decide_q6(q6))")]

    cells.append(md("""
## Summary of this notebook

The outcome lines above are collected, with the other levels, in `notebooks_sharedness/REPORT.md`.
"""))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


# ---------------------------------------------------------------------------
# Notebook 03: real data
# ---------------------------------------------------------------------------


def nb03():
    cells = [md(r"""
# 03 · Real data

There is no truth here, so the questions change form:

* (A) *descriptive* — what pattern does the fit report?
* (B) *stability* — does it report the same pattern from different seeds? (replaces "recovery")
* (C) *identification* — does rotating a shared factor into a private one cost ELBO, and does that cost
  grow with $N$? (Q5 on a subsample vs the full data)

Datasets: **COVID** (metabolomic + proteomic, $M=2$) and the **four-view cohort table** (DNA, RNA, IF
densities, IF interactions, $M=4$). Caveats from notebook 00 apply (DNA is binary, fitted as Gaussian).
"""), code(SETUP + r'''
from sharedness.data import load_covid_views, load_four_view_cohort
from sharedness import realdata as rd
DATA = {"COVID (M=2)": load_covid_views(top=150, cohort_mode="grouped"),
        "four-view (M=4)": load_four_view_cohort(top=100)}
K_REAL = {"COVID (M=2)": 5, "four-view (M=4)": 6}
SEEDS = range(3)
''')]

    cells += [question_cell("Q1", short=True, level="real"), md("""
On real data we cannot score $\\mathbb E[\\pi]$ against a truth. We ask instead: does the learned $\\pi$
behave like the synthetic O1 case (a rate clearly inside $(0,1)$, with some switches clearly off) or like
O2 (everything on)?
"""), code(r'''
q1r = rd.run_q1_real(DATA, K_REAL, seeds=SEEDS)
q1r.to_csv(os.path.join(RESULTS, "real_q1.csv"), index=False)
q1r.groupby(["dataset", "arm"])[["E_pi", "sd_pi", "frac_gamma_on", "frac_gamma_off"]].mean()
'''), code(r'''
for name in DATA:
    outcome(rd.decide_q1_real(q1r, name), title=f"Observed outcome [{name}]")
''')]

    cells += [question_cell("Q2", short=True, level="real"), code(r'''
q2r = rd.run_q2_real(DATA, K_REAL, seeds=SEEDS)
fig, axes = plt.subplots(1, len(DATA), figsize=(13, 3.6))
for ax, name in zip(axes, DATA):
    r = q2r[name]["r_ref"]
    sns.heatmap(r, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                yticklabels=DATA[name].view_names, xticklabels=[f"f{k}" for k in range(r.shape[1])])
    ax.set_title(f"{name}: variance share r (ARD, seed 0)")
plt.tight_layout(); plt.show()
pd.DataFrame({name: dict(stability_auc=q2r[name]["stability"], S_k=np.round(q2r[name]["S_ref"], 2).tolist(),
                         matched_corr=np.round(q2r[name]["match_corr"], 2).tolist()) for name in DATA}).T
'''), code(r'''
for name in DATA:
    outcome(rd.decide_q2_real(q2r[name]), title=f"Observed outcome [{name}]")
''')]

    cells += [question_cell("Q3", short=True, level="real"), md("""
Q3 on real data is the inventory itself: COVID has $2^2-1=3$ possible patterns, the four-view table $15$.
The synthetic answer (notebooks 01–02) says whether $M=2$ is enough to *detect* the pattern; here we only
note that every COVID factor must be one of {shared, metabolomic-only, proteomic-only}.
""")]

    cells += [question_cell("Q4", short=True, level="real"), md(r"""
Two switches: with an **ARD slab** (as in the synthetic `switch_warm` arm) and with a **fixed slab**
($\alpha\equiv 1/0.25$; loadings of standardised views are small, so the synthetic value 1 would be too wide).
"""), code(r'''
q4r = rd.run_q4_real(DATA, K_REAL, seeds=SEEDS)
fig, axes = plt.subplots(2, len(DATA), figsize=(13, 6.6))
for j, name in enumerate(DATA):
    for i, slab in enumerate(["ard", "fixed"]):
        sns.heatmap(q4r[name][slab]["E_h_ref"], ax=axes[i, j], cmap="Blues", annot=True, fmt=".2f",
                    vmin=0, vmax=1, yticklabels=DATA[name].view_names)
        axes[i, j].set_title(f"{name}: switch E[h], {slab} slab (seed 0)")
plt.tight_layout(); plt.show()
pd.DataFrame({(n, sl): {k: v for k, v in q4r[n][sl].items() if not isinstance(v, np.ndarray)}
              for n in DATA for sl in q4r[n]}).T
'''), code(r'''
for name in DATA:
    for slab in q4r[name]:
        outcome(rd.decide_q4_real(q4r[name][slab]), title=f"Observed outcome [{name} · {slab} slab]")
''')]

    cells += [question_cell("Q5", short=True, level="real"), md(r"""
Real-data version: the reference is the seed-0 ARD fit of Q2. Rotate its most shared factor (largest
$S_k$) into its most private one (smallest $S_k$). The **frozen** arm fixes the pattern
$\text{mask}_{mk}=\mathbb 1\{a_{mk}\ge 0.1\}$ read off that same fit's variance shares — a prescribed pattern
with real zeros (the learned ARD-slab switch produces none on these data, see Q4). "Grows with $N$" is checked
on a random quarter of the samples against all of them.
"""), code(r'''
q5r = rd.run_q5_real(DATA, K_REAL, q2r)
q5r.to_csv(os.path.join(RESULTS, "real_q5.csv"), index=False)
for name, mask in rd.q5_real_masks(q2r).items():
    print(name, "prescribed pattern (views x factors):\n", mask)
q5r.groupby(["dataset", "arm", "N", "theta"])["gap"].mean().unstack("theta")
'''), code(r'''
for name in DATA:
    for arm, oc in rd.decide_q5_real(q5r[q5r.dataset == name]).items():
        outcome(oc, title=f"Observed outcome [{name} · {arm}]")
''')]

    cells += [question_cell("Q6", short=True, level="real"), code(r'''
q6r = rd.run_q6_real(DATA, K_REAL, seeds=SEEDS)
q6r.to_csv(os.path.join(RESULTS, "real_q6.csv"), index=False)
q6r.groupby(["dataset", "mode"])[["on_share", "pi_mean", "pi_sd", "agreement_with_global"]].mean()
'''), code(r'''
for name in DATA:
    outcome(rd.decide_q6_real(q6r, name), title=f"Observed outcome [{name}]")
''')]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def nb04():
    cells = [md(r"""
# 04 · Improvements: before vs after

Notebooks 01–03 measured the original pipeline. This notebook applies the improvements suggested in
`REPORT.md` and re-measures **the same scores** — no new questions.

| arm | what changes |
|---|---|
| **before** | single PCA start, ARD, 400 iterations (as in 01–03) |
| **multistart-pca** | 8 random rotations of the PCA start, keep the best ELBO, run to relative tolerance $10^{-7}$ |
| **multistart** | the same around a **generalised-CCA** start (shared/private-aware) |
| **two-stage** | *multistart* → learned view pattern $\hat H_{mk}=\mathbb 1\{a_{mk}\ge 0.1\}$ frozen → refit with the corrected element spike-and-slab → flip borderline cells if the ELBO rises |
| **switch, calibrated slab** | view switch $h_{mk}$ with a fixed slab whose variance is read off the *multistart* fit |

Cohort prior: **before** = CohortFACTM's $q(\gamma)q(\delta)$ with learned $\pi$; **after** = the scale-free paired prior
$\delta_{ck}\mid\tau_{ck}\sim\mathcal N(0,(\kappa\tau_{ck})^{-1})$ with $q(\gamma)q(\delta\mid\gamma)$, optionally with a reference cohort.
Definitions of every component: `sharedness_model.pdf` (repo root of `factm/`).
"""), code(SETUP + r'''
from sharedness import improvements as im
from sharedness.data import load_covid_views, load_four_view_cohort
SEEDS = range(5)
''')]
    cells += [md("## 1 · Synthetic data: factor recovery and pattern recovery"), code(r'''
syn = im.run_synthetic(seeds=SEEDS)
syn.to_csv(os.path.join(RESULTS, "improve_synthetic.csv"), index=False)
order = ["before", "multistart-pca", "multistart", "two-stage", "switch, calibrated slab"]
cols = ["mean_corr", "min_corr", "min_corr_strong", "auc", "cell_acc", "exact_cols", "leaks", "misses"]
tab = syn.groupby(["level", "arm"])[cols].mean().reindex(order, level="arm")
tab
'''), code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, col, title in zip(axes, ["mean_corr", "min_corr_strong", "cell_acc"],
                          ["mean |corr| with true factors", "worst strong factor |corr|",
                           "pattern cells correct"]):
    sns.barplot(data=syn, x="arm", y=col, hue="level", order=order, ax=ax, errorbar="sd")
    ax.set_title(title); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.show()
''')]
    cells += [md("## 2 · Cohort π: does the scale-free paired prior remove the bias?"), code(r'''
coh = im.run_cohort(seeds=range(3))
coh.to_csv(os.path.join(RESULTS, "improve_cohort.csv"), index=False)
print("all fits monotone:", bool(coh.monotone.all()))
coh.groupby(["level", "arm"])[["pi_err", "gamma_auc", "null_gamma", "alt_gamma", "min_corr"]].mean()
'''), code(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, lv in zip(axes, ["easy", "hard"]):
    for arm, d in coh[coh.level == lv].groupby("arm"):
        g = d.groupby("rate")[["true_rate", "E_pi"]].mean()
        ax.plot(g["true_rate"], g["E_pi"], "o-", label=arm)
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(f"{lv}: learned π vs true rate")
    ax.set_xlabel("true rate"); ax.set_ylabel("E[π]")
axes[1].legend(fontsize=7); plt.tight_layout(); plt.show()
''')]
    cells += [md("## 3 · Real data: seed stability and the final model's view pattern"), code(r'''
DATA = {"COVID (M=2)": load_covid_views(top=150, cohort_mode="grouped"),
        "four-view (M=4)": load_four_view_cohort(top=100)}
K_REAL = {"COVID (M=2)": 5, "four-view (M=4)": 6}
real, refs = im.run_real(DATA, K_REAL, seeds=range(3), n_starts=4, max_iter=800)
real.to_csv(os.path.join(RESULTS, "improve_real.csv"), index=False)
real
'''), code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, name in zip(axes, DATA):
    H = refs[(name, "two-stage")]["H"]
    sns.heatmap(H, ax=ax, cmap="Blues", cbar=False, annot=True,
                yticklabels=DATA[name].view_names, xticklabels=[f"f{k}" for k in range(H.shape[1])])
    ax.set_title(f"{name}: frozen view pattern of the final model (seed 0)")
plt.tight_layout(); plt.show()
'''), code(r'''
rc = im.run_real_cohort(DATA, K_REAL)
rc.to_csv(os.path.join(RESULTS, "improve_real_cohort.csv"), index=False)
rc
''')]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    import sys

    BUILDERS = {
        "00": (nb00, "00_questions_and_data.ipynb"),
        "01": (lambda: nb_synthetic("easy", "01", 5), "01_easy_synthetic.ipynb"),
        "02": (lambda: nb_synthetic("hard", "02", 5), "02_harder_synthetic.ipynb"),
        "03": (nb03, "03_real_data.ipynb"),
        "04": (nb04, "04_improvements.ipynb"),
    }
    for key in sys.argv[1:] or BUILDERS:  # e.g. `_build.py 03` rebuilds only notebook 03
        make, fname = BUILDERS[key]
        nbf.write(make(), HERE / fname)
        print("written", fname)
