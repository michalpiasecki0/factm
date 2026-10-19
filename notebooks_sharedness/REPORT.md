# Sharedness between views — first results

*Branch `first_ideas_karol`, 18 October 2026. Notebooks `00`–`03` in this folder hold every table and plot;
running them also writes raw CSVs to `results/` (gitignored, so local only).*

## The short version

1. **Forced zeros pin the shared/private split; learned zeros do not.** Rotating a shared factor into a
   private one costs an ELBO penalty that grows **like N** when the view pattern is fixed (×4.3 easy, ×3.9
   hard, ×3.8 on the four-view real data, for ×4 samples), but only **×1.3–1.5** with ARD, and a learned switch
   either matches ARD (synthetic) or gives no consistent resistance at all (real data). This is the
   frozen-vs-learned claim from the skills, confirmed at all three levels (Q5).
2. **The pattern is already readable from a plain fit.** The variance share of each factor in each view
   recovers the true pattern with AUC 0.97–1.00 (Q2). With ARD this holds on hard data too (0.97).
3. **When the fit finds the wrong axes, it is the optimiser, not the model.** Started at the truth, the same
   model reaches a higher ELBO (easy: +138 to +728 nats; hard: +0.2 to +41) and keeps the true axes. On hard
   data the PCA start visibly mixes a shared factor with a private one, and the unmixed truth scores better
   (Q2b). Better initialisation is a cheaper fix than new theory here.
4. **CohortFACTM's learned π is biased, and the cause is the factorisation.** With q(γ)q(δ) the γ update has
   no Occam term, so switches that should be off are not pushed off. On real data this drives learned π to
   0.90 (COVID) and 0.97 (four-view) with almost no switch off. A paired q(γ)q(δ|γ) reduces the bias
   (worst error 0.42 → 0.28 easy, 0.24 → 0.17 hard; real 0.66 / 0.82) but does not remove it (Q1).
5. **An on/off switch per view works only with a fixed slab, and only on synthetic data so far.** With an ARD
   slab it stays on (96–100 % of switches) because a column can be silenced by α → ∞ instead of h = 0. With a
   fixed slab it is decisive and nearly correct on hard synthetic data (39 % on vs 50 % true, 1 leak), but on
   real data it still keeps 88–100 % on (Q4; see the slab-variance caveat).
6. **Use one global π.** Per-factor π_k is too uncertain at M ≤ 4 (posterior sd ≈ 0.18) and never improved
   recovery (Q6). Two views are enough to *detect* the pattern on clean data, but more views help when
   the data are noisy (AUC 0.80 at M = 2 → 0.96 at M ≥ 3, hard data) (Q3).

## Outcomes, question by question

Outcomes were fixed in notebook 00 before any run, and each was picked by a rule in
`sharedness/experiments.py`. Where the rule's label disagrees with the substance, both are given.

| | Question | Easy synthetic | Harder synthetic | Real data |
|---|---|---|---|---|
| Q1 | Can a learned π find the true cohort switch rate? | **O4 partial** — tracks the rate, biased (max error 0.42 current, 0.28 paired) | **O4 partial** — 0.24 current, 0.17 paired | current q **drifts to 1** (COVID 0.90, four-view 0.97; 0–2 % of switches off); paired q 0.66 / 0.82 |
| Q2 | Does a plain fit show the pattern? | **O1 yes** — AUC 0.97 without ARD, 1.00 with | **O2 with ARD** — 0.97 vs 0.87 | **partly stable** — read-outs agree across seeds at corr 0.72 (COVID) / 0.90 (four-view); weakest factor matches at only 0.57–0.58 |
| Q2b | Wrong axes: optimiser or model? | **O1 optimiser** — truth start +138…+728 nats, keeps the axes | rule says **O2 model**; substance **O1 optimiser** for the strong factors — see below | — |
| Q3 | How many views are needed? | **O1** — AUC ≥ 0.98 at every M | **O2 more views help** — 0.80 (M=2) → 0.96 (M≥3) | COVID: 3 possible patterns; four-view: 15 |
| Q4 | Is an on/off switch cleaner than ARD? | **O2 stays on** — 96 % on (true 50 %) | rule says **O1**, substance **O2** — ARD-slab switch 100 % on; see below | **stays on** — ARD slab 100 % on in both; fixed slab 100 % (COVID) / 88 % (four-view) |
| Q5 | Do forced zeros pin the split better? | frozen **O2 ×4.3**; ARD and learned switch **O3 ×1.3** | frozen **O2 ×3.9**; ARD and learned **O3 ×1.5** | frozen **grows like N**: four-view ×3.8, COVID ×10.7; ARD ×1.4 (four-view); learned switch **no consistent resistance** (gap turns negative) |
| Q6 | One π or one per factor? | **O2 global** — sd(π_k) 0.18, no AUC gain | **O2 global** — sd 0.19, no gain | **global** — sd(π_k) 0.16–0.20; calls agree with global π in 100 % of cells |

### Real data, in one paragraph

The real-data runs reproduce the two synthetic findings that matter most: CohortFACTM's learned π drifts
towards 1 (Q1), and only a prescribed pattern resists rotation in an N-growing way (Q5: four-view gap
498 → 1,914 nats, ×3.8). They also show where things are weaker on real data: sharedness read-outs are only
partly seed-stable (Q2), and no learned switch — not even the fixed-slab one — turned many columns off (Q4).
The prescribed patterns used for Q5 (threshold 0.1 on the ARD variance shares) were:

    COVID (views x factors)        four-view (DNA, RNA, IF_dens, IF_inter x factors)
    [[1 1 1 0 0]                   [[0 0 1 1 0 0]
     [1 1 1 1 1]]                   [1 1 1 1 1 0]
                                    [0 0 1 1 1 1]
                                    [0 0 1 1 0 1]]

So on the four-view data two factors look RNA-private (f0, f1), two are shared by all four views (f2, f3),
and two are partially shared: f4 by RNA and IF densities, f5 by the two IF views. Every view pattern class
— private, partially shared, fully shared — appears in real data, which is the case for M ≥ 3.

### Notes on individual results

**Q1, the cohort π.** The learned rate under the current factorisation overshoots on easy data
(0.25 / 0.63 / 0.96 for true 0.10 / 0.31 / 0.53). The paired version is closer (0.24 / 0.59 / 0.79) and also
ranks the switches better on hard data (AUC 0.97 / 0.91 / 0.90 vs 0.85 / 0.76 / 0.79). On hard data the
factors themselves are poorly recovered in the Q1 setting (6 cohorts of 25 samples, min |corr| ≈ 0.35), so
part of the Q1-hard error is a factor-recovery error, not a π error.

**Q2b on hard data: the rule was fooled by the weak factor.** The rule uses the *minimum* factor
correlation, and the hard design includes a deliberately weak private factor (scale 0.4) that is lost
from *both* starts (|corr| ≈ 0.6, or 0 when ARD switches it off). Looking factor by factor (seed 1):

| | f0 all views | f1 views 1–2 | f2 views 2–4 | f3 private v1 | f4 private v3 | f5 weak |
|---|---|---|---|---|---|---|
| ARD, PCA start | 0.99 | **0.78** | 0.98 | **0.77** | 0.92 | 0.61 |
| ARD, truth start (+41 nats) | 0.99 | **0.98** | 0.98 | **0.94** | 0.93 | 0.60 |

The PCA start mixes a factor shared by views 1–2 with a factor private to view 1 — exactly the
shared/private confusion this thesis is about — and the truth start, which keeps them apart, has the
*higher* ELBO. In every hard seed the truth start is at least as good (+0.2 to +41 nats). So the
substantive reading on hard data is also **optimiser**, plus a detection limit for the weak factor.

**Q4, why the rule and the substance disagree on hard data.** The pre-registered O1 ("cleaner": AUC ≥ ARD − 0.02
and ≥ 90 % decisive) and O2 ("stays on": ≥ 95 % on) can both hold, and the rule tests O1 first. On hard data the
ARD-slab switch is "decisive" only because every switch sits near 1. The rule is left unchanged, since
changing it after seeing the result would undo the pre-registration; the substantive reading is O2. The
fixed-slab switch is the one that works:

| arm | easy: % on (true 50 %) | easy: leaks / misses | hard: % on (true 50 %) | hard: leaks / misses |
|---|---|---|---|---|
| switch, cold start, ARD slab | 85 % | 5.6 / 0 | 72 % | 7.8 / 0 |
| switch, warm start, ARD slab | 96 % | 7.4 / 0 | 100 % | 12.0 / 0 |
| switch, warm start, **fixed slab** | **65 %** | **2.4 / 0** | **39 %** | **1.0 / 0.4** |

**Q5, the frozen-vs-learned test.** ELBO gap Δ(θ) when a shared factor is rotated into a private one by θ,
everything except the factor means re-optimised (hard data, mean over 2 seeds):

| arm | N | 10° | 20° | 30° | 45° |
|---|---|---|---|---|---|
| ARD | 150 | 43 | 74 | 88 | 91 |
| | 600 | 80 | 115 | 130 | 139 |
| learned switch | 150 | 43 | 72 | 86 | 89 |
| | 600 | 79 | 114 | 130 | 138 |
| **frozen (true pattern)** | 150 | 91 | 313 | 609 | 1087 |
| | 600 | 351 | 1225 | 2388 | 4226 |

The learned switch behaves exactly like ARD: it switches the needed column back on and pays only a
prior-sized cost. That is Lemma A's trap one level up, as predicted.

## Improvements (notebook 04) — before vs after on the same scores

The full model, its priors, variational family, ELBO, updates and fitting order are written out in
`factm/sharedness_model.pdf` (source `sharedness_model.tex`). Changes: multi-start (8 starts, best ELBO) run to a
relative tolerance; a generalised-CCA start; a two-stage fit (learn pattern → freeze → refit with the corrected
element spike-and-slab → ELBO refinement); the paired cohort prior with a learned π; a view switch with a slab
variance calibrated from the stage-1 fit.

**Synthetic, 5 seeds per level**

| level | arm | mean \|corr\| | worst strong factor | pattern cells correct | exact columns |
|---|---|---|---|---|---|
| easy | before | 0.921 | 0.856 | 0.900 | 3.0 / 4 |
| easy | multi-start | **0.996** | **0.991** | **1.000** | **4.0 / 4** |
| easy | two-stage (final) | 0.995 | 0.989 | **1.000** | **4.0 / 4** |
| hard | before | 0.855 | 0.878 | 0.950 | 5.2 / 6 |
| hard | multi-start (PCA start) | **0.869** | **0.912** | **0.983** | **5.6 / 6** |
| hard | two-stage (final) | 0.842 | 0.910 | 0.975 | 5.4 / 6 |

- **Multi-start is the big gain.** On easy data every seed now recovers the true factors and the exact pattern
  (spread across seeds ≈ 0). On hard data it fixes the seed where a shared factor was mixed with a private one
  (0.77 → 0.92) and reaches the truth-start ELBO in **every** seed — the optimiser problem of Q2b is solved.
- **The gCCA start adds nothing over a multi-started PCA start** (easy: identical; hard: pattern cells 0.975 vs
  0.983) and loses the deliberately weak factor more often. Keep the PCA start as default.
- **Two-stage does not improve recovery and is not meant to**: its gain is identification (Q5: the rotation
  penalty grows like N with a frozen pattern). Refinement flipped no cell at either level.
- **A calibrated slab fixes the view switch on synthetic data**: 50.0 % / 47.5 % of switches on (truth 50 %),
  switch AUC 1.00 / 0.99 (easy / hard), versus 96–100 % on with an ARD slab.

**Cohort prior, 3 seeds × true rates 0.1 / 0.3 / 0.6**

| arm | γ AUC easy | γ AUC hard | π error easy | π error hard |
|---|---|---|---|---|
| CohortFACTM, π = 0.5 fixed | 0.841 | 0.883 | 0.208* | 0.244 |
| CohortFACTM, π learned | 0.864 | 0.755 | 0.291 | 0.138 |
| **paired, π learned (final)** | **0.873** | **0.904** | 0.235 | **0.105** |
| scale-free paired (variant) | 0.869 | **0.904** | 0.337 | 0.107 |
| reference cohort (variant) | 0.619 | 0.642 | 0.163 | 0.121 |

\*0.5 happens to be near the average true rate; a fixed π does not learn anything. Learning π *without* pairing
hurts detection on hard data (0.883 → 0.755); with pairing it is the best or tied everywhere. The scale-free shift
prior and reference coding were not adopted (no gain; reference coding detects shifts worse).

**Real data, 3 seeds (4 starts each)** — seed stability, since there is no truth

| data | arm | activity matrices agree across seeds | mean factor match | worst factor match |
|---|---|---|---|---|
| four-view (M=4) | before | 0.898 | 0.785 | 0.580 |
| four-view (M=4) | multi-start | **1.000** | **0.999** | **0.999** |
| four-view (M=4) | two-stage (final) | **1.000** | **1.000** | **1.000** |
| COVID (M=2) | before | 0.724 | 0.738 | 0.572 |
| COVID (M=2) | multi-start | 0.962 | 0.762 | 0.582 |
| COVID (M=2) | two-stage (final) | **0.977** | **0.862** | **0.624** |

- **On the four-view data the final model gives the same factors and the same shared/private pattern from
  every seed** (worst factor match 0.58 → 1.00). Multi-start also reaches a higher ELBO than the single start on
  both datasets.
- **COVID improves but stays partly unstable** (worst factor 0.62): with two views the shared block is large and
  rotations inside it are not pinned by zeros — the M = 2 limitation seen in Q3.

| data | cohort prior | E[π] | switches on (>0.9) | switches off (<0.1) |
|---|---|---|---|---|
| COVID | CohortFACTM, π learned | 0.92 | 100 % | 0 % |
| COVID | **paired, π learned (final)** | **0.52** | 40 % | **47 %** |
| four-view | CohortFACTM, π learned | 0.97 | 100 % | 0 % |
| four-view | **paired, π learned (final)** | 0.89 | 83 % | 0 % |

On COVID the paired prior turns "every cohort shifts on every factor" into a real decision. On the four-view data
most switches stay on; with six different cancer types that may simply be true.

## What this means for the two thesis aims

**Aim (b), the shared/private translation, is viable — with one condition.** The split is identified at
likelihood level only if the pattern is *prescribed*. A learned pattern gives a weak preference that does
not grow with N. Two workable designs follow:
- **Two-stage:** learn the pattern (variance shares or a fixed-slab switch), then freeze it and refit.
  The refit then has the `lem:zeros` protection. The open question is how much the error from stage one
  leaks into stage two.
- **Pattern as a hyperparameter:** treat the pattern as discrete and compare patterns by ELBO, as SLIDE
  does by selection.

**Aim (a), CohortFACTM's identifiability, has a concrete new item.** Learning π in the current model is
biased because of the q(γ)q(δ) factorisation; the paired factorisation is the principled fix and passed
every correctness test here (ELBO monotone; each update a stationary maximum).

**Not shown here:** identification *within* the shared block. Every result above is about the block
partition. Rotations inside the shared block are not tested and, by `lem:zeros`, are not expected to be
pinned by zeros.

## Is proper mathematical research needed for the thesis? Yes, in four places

| # | Open point | Why the experiments cannot settle it | Kind of work |
|---|---|---|---|
| R1 | `lem:zeros` under CohortFACTM's prior | **Proved** (model document, P3): block-diagonality needs only a-priori independent factors (diagonal, cohort-specific covariances), not N(0, I). The earlier claim that it fails under the cohort prior was wrong | Done; write it up |
| R2 | Learned vs frozen as a theorem | Q5 shows ×1.3–1.5 vs ×3.9–4.3 on two settings; a claim for the thesis needs the order of growth in general | A short proof: the frozen gap is a residual sum of squares (Θ(N)); the learned gap is bounded by the switch prior cost (O(MK)) plus an ARD term |
| R3 | Where the learned cohort π settles | The derivation (missing Occam term) is correct, but a simple χ² model of the null cohorts predicts 0.68 at a true rate of 0.10, while the fit gives 0.25. The coupling between S_c and γ through the shrunk E[z] is not captured | A careful fixed-point analysis of the mean-field updates |
| R4 | Identifiability of h next to α | Q4 shows an ARD slab makes h unidentifiable (silencing a column by α → ∞ costs nothing) and a fixed slab restores it. That is an empirical statement about two settings | A lemma: with a learned slab scale the h-likelihood is flat in the limit α → ∞ |

R1 is now proved. R2 (stated with its hypotheses as Conjecture 1 in the model document) is the headline
result still to prove — it needs a third sample size to separate log N from small powers of N. R4 is a
one-paragraph lemma plus a pruning argument. R3 is optional: the paired fix can be adopted without it.
All statements, statuses and proof sketches are in `factm/sharedness_model.pdf`, Section "Proofs needed".

Two conjectures from the skills remain untested: that per-view ARD identifies by joint diagonalisation
(suggestion 8), and that the two switch layers (cohort and view) can be used together (suggestion 7).

## What to do next (suggested)

Done on this branch: paired cohort prior adopted; multi-start and convergence; two-stage design; calibrated
view switch; `ARD_SS` corrected; R1 proved. Remaining:

1. Use **multi-start with the PCA start** as the default fit everywhere (largest single gain).
2. Prove **R2** (frozen vs learned growth) and **R4**; run Q5 at a third sample size.
3. On real data, report the **two-stage frozen pattern** as the shared/private answer, with seed stability.
4. Treat DNA with a genuine Bernoulli likelihood: the repo's `BernoulliLikelihood` is a stub that reuses the
   Gaussian updates, so this needs a Jaakkola–Jordan or Pólya–Gamma bound threaded through the Z, W and noise nodes.

## Caveats

- Synthetic results use 5 seeds (Q5: 2), one generator and two difficulty levels; the thresholds in the
  decision rules are judgement calls, fixed in advance.
- The cohort truth in Q1 is "differs from the grand mean by more than 1": because the data are centred
  globally, CohortFACTM's γ cannot have any other reference. This ambiguity in what γ means is itself worth a
  sentence in the thesis.
- Real data: COVID has only 2 views (3 possible patterns); in the four-view table DNA is 25 binary mutation
  calls fitted with a Gaussian likelihood. Real-data runs use 3 seeds; Q5 compares a random quarter of the
  samples with all of them, which changes the data as well as N (COVID's ×10.7 for ×4 samples is larger than
  linear for that reason, and should not be read as "faster than N").
- The real-data fixed-slab switch used slab variance 0.25 (standardised views have small loadings); a narrower
  slab makes switching off *harder*, and no sensitivity to this value was run. Its "stays on" result may be
  partly this choice.
