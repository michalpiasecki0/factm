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
# Shared question text. Used verbatim in notebook 00 and repeated, shortened,
# at the top of each section of notebooks 01-03.
# ---------------------------------------------------------------------------

Q = {}

Q["Q1"] = dict(
    title="Q1 — Can CohortFACTM learn how often cohorts differ?",
    plain="""CohortFACTM gives every (cohort, factor) pair a switch $\\gamma_{ck}$: *does cohort $c$ have its
own centre on factor $k$?* Today the switch rate $\\pi$ is a number we pick (0.5). If we let the model
**learn** $\\pi$, does it find the true rate?""",
    math=r"""
$$z_{nk}\mid c \sim \mathcal N(\gamma_{ck}\delta_{ck},\ \tau_{ck}^{-1}),\qquad
\gamma_{ck}\sim\mathrm{Bern}(\pi),\qquad \pi\sim\mathrm{Beta}(1,1).$$
Mean-field update: $q(\pi)=\mathrm{Beta}\big(1+\sum_{c,k}\mathbb E[\gamma_{ck}],\ 1+CK-\sum_{c,k}\mathbb E[\gamma_{ck}]\big)$.

**Data:** cohort switches drawn with rate $\pi^\ast\in\{0.1,0.3,0.6\}$ ($C=6$, shift $\pm2$).
Because the data are centred globally, the switch can only see a cohort's mean *relative to the grand
mean*; the truth we score against is $G^{\text{eff}}_{ck}=\mathbb 1\{|\bar z_{ck}-\bar z_{\cdot k}|>1\}$.

**Measured:** $\mathbb E[\pi\mid Y]$ against the effective true rate, and the AUC of $\mathbb E[\gamma]$ against $G^{\text{eff}}$.

**Arms:** (i) fixed $\pi=0.5$ (current model); (ii) learned $\pi$ with the current factorisation
$q(\gamma)q(\delta)$; (iii) learned $\pi$ with the paired factorisation $q(\gamma)q(\delta\mid\gamma)$.""",
    outcomes=[
        ("O1", "Learnable", r"max $|\mathbb E[\pi]-\pi^\ast| < 0.15$ and slope $>0.5$",
         "the data carry the rate; keep $\\pi$ learnable"),
        ("O2", "Drifts to 1", r"$\mathbb E[\pi] > 0.8$ at every $\pi^\ast$",
         "switches are never pushed off; the factorisation, not the formula, is at fault"),
        ("O3", "Prior-dominated", r"$\mathbb E[\pi]$ moves $<0.2$ across $\pi^\ast$",
         "too few switches ($CK$) to learn from; fix $\\pi$"),
        ("O4", "Partial", "anything else", "moves with the truth but biased; report the bias"),
    ],
)

Q["Q2"] = dict(
    title="Q2 — Can a plain fit, with no new parts, already show which factors are shared?",
    plain="""Fit the model we already have. For each factor, ask each view *how much of you does this factor
explain?* If the answer is ~0 in some views and large in others, the fit has told us the pattern
for free.""",
    math=r"""
Variance share of factor $k$ in view $m$ (non-negative, unlike `FA.variance_explained_per_factor_view`):
$$r_{mk}=\frac{\lVert \mathbb E[z_{\cdot k}]\,\mathbb E[w^{(m)}_{\cdot k}]^{\top}\rVert_F^2}{\lVert Y^{(m)}\rVert_F^2},
\qquad a_{mk}=\frac{r_{mk}}{\max_{m'} r_{m'k}}\in[0,1].$$
Sharedness of factor $k$: $S_k=-\sum_m p_{mk}\log p_{mk}/\log M$, $p_{mk}=r_{mk}/\sum_{m'}r_{m'k}$
($1$ = spread over all views, $0$ = one view).

**Measured:** after matching fitted to true factors (Hungarian on $|\mathrm{corr}|$),
$\mathrm{AUC}(a, H)=P(a_{mk}>a_{m'k'}\mid H_{mk}=1,H_{m'k'}=0)$ — threshold-free, $1$ perfect, $0.5$ chance.

**Arms:** no loadings prior ($w\sim\mathcal N(0,1)$) and ARD.""",
    outcomes=[
        ("O1", "Yes, any fit", "AUC $\\ge 0.95$ with and without ARD",
         "sharedness is a read-out; no new model needed to *see* it"),
        ("O2", "Yes, with ARD", "AUC $\\ge 0.95$ only with ARD",
         "ARD is what makes it readable"),
        ("O3", "Partly", "best AUC in $[0.75, 0.95)$", "some cells wrong; inspect leaks"),
        ("O4", "No", "best AUC $<0.75$", "the fit does not expose the pattern"),
    ],
)

Q["Q2b"] = dict(
    title="Q2b — When the fit misses the true factors, is it the optimiser or the model?",
    plain="""Reading the pattern needs the right factors first. If a fit lands on a *rotated* set of factors,
there are two very different explanations: the model would prefer the truth but the algorithm never
gets there (an **optimiser** problem — fixable), or the model's own objective prefers the rotated
answer (a **model / identification** problem — needs theory).""",
    math=r"""
For every seed where the ARD fit from the usual PCA start has $\min_k|\mathrm{corr}(\hat z_k,z^\ast_k)|<0.9$,
refit starting **at the truth** $(Z^\ast,W^\ast)$ and compare
$$\Delta_{\text{start}}=\mathrm{ELBO}_{\text{truth start}}-\mathrm{ELBO}_{\text{PCA start}}.$$""",
    outcomes=[
        ("O0", "Not needed", "every fit found the true axes", "skip"),
        ("O1", "Optimiser", r"$\Delta_{\text{start}}>10$ nats and the truth start keeps the true axes",
         "better initialisation or annealing would fix it"),
        ("O2", "Model", r"$\Delta_{\text{start}}<-10$ or the truth start drifts away",
         "the ELBO prefers a rotation: an identification result to prove (Lemma C regime)"),
        ("O3", "Flat", r"$|\Delta_{\text{start}}|\le10$", "the rotation is (nearly) unidentified"),
    ],
)

Q["Q3"] = dict(
    title="Q3 — How many views do we need before ‘shared vs private’ is a meaningful question?",
    plain="""With 2 views there are only 3 patterns: shared, private to view 1, private to view 2. With 4 views
there are 15. Does having few views make the pattern harder to *detect*, or just coarser?""",
    math=r"""
Number of non-empty patterns: $2^M-1$. A per-factor rate learned from $M$ switches has posterior
$\mathrm{Beta}(1+s_k,\,1+M-s_k)$ with $s_k=\sum_m h_{mk}$ — so even a *perfectly* observed pattern
leaves $\mathrm{sd}(\pi_k)$ large when $M$ is small (table below, no fitting needed).

**Data:** one shared factor, one factor in views 1–2 (if $M>2$), one private factor per view;
$M\in\{2,3,4,6\}$. **Measured:** AUC of ARD activity against $H$.""",
    outcomes=[
        ("O1", "M does not limit detection", "AUC $\\ge0.95$ at every $M$",
         "M=2 data (COVID) can answer the question, just coarsely"),
        ("O2", "More views help", "AUC rises by $\\ge0.05$ from smallest to largest $M$",
         "prefer data with more views"),
        ("O3", "More views hurt", "AUC falls by $\\ge0.05$",
         "more private factors split the signal"),
        ("O4", "Flat but imperfect", "otherwise", "detection limited by something other than $M$"),
    ],
)

Q["Q4"] = dict(
    title="Q4 — Does an explicit on/off switch per view give a cleaner answer than ARD’s dials?",
    plain="""ARD turns a factor's loadings in a view *down* but never fully *off*. The view-axis twin of
CohortFACTM's $\\gamma$ is a real switch $h_{mk}$: is factor $k$ used by view $m$, yes or no?
Does that switch give sharper, correct answers?""",
    math=r"""
$$W^{(m)}_{\cdot k}=h_{mk}V^{(m)}_{\cdot k},\quad h_{mk}\sim\mathrm{Bern}(\pi),\quad
V^{(m)}_{dk}\sim\mathcal N(0,\alpha_{mk}^{-1}),\quad q(h,V)=q(h)\,q(V\mid h).$$
$$\operatorname{logit}\mathbb E[h_{mk}]=\mathbb E\log\pi-\mathbb E\log(1-\pi)+\sum_{d}\Big[\tfrac12\log\tfrac{\mathbb E[\alpha_{mk}]}{\Lambda_d}+\tfrac{\nu_d^2}{2\Lambda_d}\Big],
\quad \Lambda_d=\textstyle\sum_n\mathbb E[z_{nk}^2]\,\mathbb E[\tau_d]+\mathbb E[\alpha_{mk}].$$
**Measured:** AUC of $\mathbb E[h]$ against $H$; *decisiveness* = share of $\mathbb E[h_{mk}]\notin(0.1,0.9)$;
leaks (private called shared) and misses.

**Arms:** ARD (reference); switch started cold; switch started from a converged ARD fit ("warm");
warm switch with a fixed slab variance ($\alpha\equiv1$) instead of ARD.""",
    outcomes=[
        ("O1", "Cleaner", "best switch AUC $\\ge$ ARD$-0.02$, $\\ge0.9$, decisive $\\ge90\\%$",
         "the switch is worth adding as a diagnostic"),
        ("O2", "Switch stays on", "$\\ge95\\%$ of switches on in the best arm",
         "$h$ is not identifiable next to $\\alpha$; drop the switch or change the slab"),
        ("O3", "Worse than ARD", "best switch AUC $<$ ARD$-0.05$", "the switch harms recovery"),
        ("O4", "Mixed", "otherwise", "report per arm"),
    ],
)

Q["Q5"] = dict(
    title="Q5 — Do forced zeros pin the factors down better than learned zeros?",
    plain="""Mix a shared factor into a private one by rotating them. A *forced* zero (‘factor 2 never appears in
view 1’) should make that rotation impossible. A *learned* zero can simply be switched back on.
How much does the model resist the rotation in each case, and does the resistance grow with the
number of samples?""",
    math=r"""
Fix the factor means at $\bar Z(\theta)=Z^\ast G(\theta)$, where $G(\theta)$ rotates a fully shared and a
view-private factor, and re-optimise **everything else**. With $z\sim\mathcal N(0,I)$ the factor-prior
part of the ELBO is unchanged by the rotation, so the gap
$$\Delta(\theta)=\mathrm{ELBO}(0)-\mathrm{ELBO}(\theta)$$
comes from the loadings side only. Likelihood-level identification means $\Delta=\Theta(N)$;
a prior-level preference means $\Delta=O(1)$ in $N$. We compare $N=150$ and $N=600$ ($\times4$).

**Arms:** ARD (no switch); learned switch; frozen switch fixed at the true pattern.""",
    outcomes=[
        ("O1", "Flat", "$\\Delta < 5$ nats at the larger $N$", "rotation is essentially free"),
        ("O2", "Grows like N", "gap ratio $\\ge 0.6\\times$ the sample ratio",
         "pinned at likelihood level (`lem:zeros` regime)"),
        ("O3", "Grows slower than N", "otherwise", "a real penalty, but not identification"),
    ],
)

Q["Q6"] = dict(
    title="Q6 — One sharing rate for the whole model, or one per factor?",
    plain="""A single $\\pi$ says ‘in this dataset, factors are mostly shared (or mostly private)’. A rate $\\pi_k$
per factor says ‘factor $k$ is shared’. But $\\pi_k$ is learned from only $M$ switches. Is it worth it?""",
    math=r"""
Global: $q(\pi)=\mathrm{Beta}(1+\sum_{m,k}\mathbb E h_{mk},\,1+MK-\sum\mathbb E h_{mk})$.
Per factor: $q(\pi_k)=\mathrm{Beta}(1+\sum_m\mathbb E h_{mk},\,1+M-\sum_m\mathbb E h_{mk})$.

**Measured:** AUC of $\mathbb E[h]$ (both modes); $\mathrm{sd}(\pi_k)$; separation
$\overline{\mathbb E[\pi_k]}_{\text{shared}}-\overline{\mathbb E[\pi_k]}_{\text{private}}$. $M\in\{2,4,8\}$.""",
    outcomes=[
        ("O1", "Either works", "sd$(\\pi_k)<0.15$ at $M\\le4$, no AUC gain",
         "choose by interpretability"),
        ("O2", "Use one global $\\pi$", "sd$(\\pi_k)\\ge0.15$ at $M\\le4$, no AUC gain",
         "per-factor rates are noise at realistic $M$"),
        ("O3", "Per-factor helps", "AUC gain $\\ge0.03$ at some $M$", "keep $\\pi_k$ where it helps"),
    ],
)


def outcomes_table(q):
    rows = ["| Outcome | Name | What we would see | What it would mean |", "|---|---|---|---|"]
    rows += [f"| **{c}** | {n} | {see} | {mean} |" for c, n, see, mean in q["outcomes"]]
    return "\n".join(rows)


def question_cell(key, short=False):
    q = Q[key]
    if short:
        return md(f"## {q['title']}\n\n{q['plain']}\n\n{outcomes_table(q)}\n\n"
                  f"*Full mathematical statement: notebook 00.*")
    return md(f"## {q['title']}\n\n**In plain words.** {q['plain']}\n\n**Precisely.**{q['math']}\n\n"
              f"**Possible outcomes (fixed before running; thresholds live in "
              f"`sharedness/experiments.py`).**\n\n{outcomes_table(q)}")


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
    cells += [question_cell("Q1", short=True), code(r'''
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
    cells += [question_cell("Q2", short=True), code(r'''
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

    cells += [question_cell("Q2b", short=True), code(r'''
piv = q2[q2.arm.isin(["ARD", "ARD_truthstart"])].pivot(index="seed", columns="arm",
                                                       values=["elbo", "min_corr"])
piv[("elbo", "truth − PCA start")] = piv[("elbo", "ARD_truthstart")] - piv[("elbo", "ARD")]
piv
'''), code("outcome(ex.decide_q2b(q2))")]

    # Q3
    cells += [question_cell("Q3", short=True), code(r'''
q3 = ex.run_q3(LEVEL, seeds=SEEDS)
q3[["M", "seed", "auc_ard", "auc_r", "exact_r", "min_corr"]].to_csv(
    os.path.join(RESULTS, f"{LEVEL}_q3.csv"), index=False)
q3.groupby("M")[["auc_ard", "auc_r", "exact_r", "min_corr"]].mean()
'''), code("outcome(ex.decide_q3(q3))")]

    # Q4
    cells += [question_cell("Q4", short=True), code(r'''
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
    cells += [question_cell("Q5", short=True), code(r'''
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
    cells += [question_cell("Q6", short=True), code(r'''
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

    cells += [question_cell("Q1", short=True), md("""
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

    cells += [question_cell("Q2", short=True), code(r'''
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

    cells += [question_cell("Q3", short=True), md("""
Q3 on real data is the inventory itself: COVID has $2^2-1=3$ possible patterns, the four-view table $15$.
The synthetic answer (notebooks 01–02) says whether $M=2$ is enough to *detect* the pattern; here we only
note that every COVID factor must be one of {shared, metabolomic-only, proteomic-only}.
""")]

    cells += [question_cell("Q4", short=True), md(r"""
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

    cells += [question_cell("Q5", short=True), md(r"""
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

    cells += [question_cell("Q6", short=True), code(r'''
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
