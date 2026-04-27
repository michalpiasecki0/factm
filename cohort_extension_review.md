# Cohort FACTM change review

## What was fixed

1. `run_cohort.ipynb` was rebuilt to run top-to-bottom on `raw_data_cohort.csv`:
   - uses `group` as cohort labels,
   - trains baseline (`ZPrior.STD_NORMAL`) and cohort (`ZPrior.COHORT`) variants,
   - writes `cohort_elbo_sequence.csv`,
   - writes `cohort_elbo_plot.png`.
2. Cohort update stability fix in `src/z_priors/Cohort.py`:
   - logistic argument is clipped before `exp` to avoid overflow in `gamma` update.
3. Z-ELBO dispatch fix in `src/z_priors/StdNormal.py`:
   - falls back to standard ELBO if a prior does not provide `compute_elbo_k`.
4. Removed leftover debug print from Z update in `src/z_priors/StdNormal.py`.

## Where cohort formulas are implemented

### 1) Cohort structure and counts
- **File:** `src/z_priors/Cohort.py`
- **Class:** `Cohort` (lines ~19-35)
- Creates:
  - cohort index per sample: `cohort_indices`,
  - cohort sizes: `counts = |N_c|`.

### 2) Expected values used in updates
- **File:** `src/z_priors/Cohort.py`
- **Method:** `_update_expectations` (lines ~71-82)

Definitions in code:
- `E_delta = delta_mu`
- `E_delta_squared = delta_var + delta_mu**2`
- `E_gamma = gamma_prob`
- `E_tau = tau_a / tau_b`
- `E_log_tau = digamma(tau_a) - log(tau_b)`
- `E_lambda = lambda_a / lambda_b`
- `E_log_lambda = digamma(lambda_a) - log(lambda_b)`

### 3) Special cohort `z` update
- **File:** `src/z_priors/Cohort.py`
- **Method:** `update_z_k` (lines ~83-145)

Implemented form:
- $$\sigma_{n,k}^{2} = \left(E[\tau_{c(n),k}] + \sum_{m,d} E[\tau_{n,d}^{m}]E[w_{d,k}^{m2}] + \text{CTM term}\right)^{-1}$$
- $$\mu_{n,k} = \sigma_{n,k}^{2}\left(E[\tau_{c(n),k}]E[\gamma_{c(n),k}]E[\delta_{c(n),k}] + \sum_{m,d}E[\tau_{n,d}^{m}]E[w_{d,k}^{m}]y_{n,d}^{m} + \text{CTM term}\right)$$

Mapping in code:
- prior part: lines ~98-105,
- FA likelihood terms: lines ~115-121,
- CTM quadratic/linear terms: lines ~123-135,
- posterior assign: lines ~137-143.

### 4) Cohort parameter updates
- **File:** `src/z_priors/Cohort.py`
- **Method:** `_update_cohort_params` (lines ~146-185)

#### Delta update
- $$\text{Var}(\delta_{c,k}) = \left(E[\lambda_k] + E[\tau_{c,k}]E[\gamma_{c,k}]|N_c|\right)^{-1}$$
- $$E[\delta_{c,k}] = \text{Var}(\delta_{c,k})E[\tau_{c,k}]E[\gamma_{c,k}]\sum_{n\in N_c}E[z_{n,k}]$$
- Code: lines ~161-164.

#### Gamma update
- $$u_{c,k}=\log\frac{\pi_k}{1-\pi_k}-\frac12E[\tau_{c,k}]\sum_{n\in N_c}\left(E[\delta_{c,k}^{2}]-2E[\delta_{c,k}]E[z_{n,k}]\right),\quad E[\gamma_{c,k}]=\sigma(u_{c,k})$$
- Code: lines ~166-171.

#### Tau update
- $$a^{(\tau)}_{c,k}=a_0^\tau+\frac{|N_c|}{2}$$
- $$b^{(\tau)}_{c,k}=b_0^\tau+\frac12\sum_{n\in N_c}\left(E[z_{n,k}^2]-2E[\gamma_{c,k}]E[\delta_{c,k}]E[z_{n,k}] + E[\gamma_{c,k}]E[\delta_{c,k}^2]\right)$$
- Code: lines ~173-180.

#### Lambda update
- $$a_k^{(\lambda)}=a^\lambda+\frac{C}{2},\quad b_k^{(\lambda)}=b^\lambda+\frac12\sum_cE[\delta_{c,k}^2]$$
- Code: lines ~182-185.

## Where ELBO changes are handled

### Prior-specific ELBO hook
- **File:** `src/z_priors/base.py` (lines ~34-36)
- `compute_elbo_k(...)` is the optional per-factor prior ELBO interface.

### Z-node ELBO dispatch
- **File:** `src/z_priors/StdNormal.py`
- **Method:** `nodeFA_z.ELBO` (lines ~113-132)
- Behavior:
  - if prior returns `compute_elbo_k`, use it,
  - else use standard Normal Z ELBO fallback.

### Cohort ELBO terms
- **File:** `src/z_priors/Cohort.py`
- **Method:** `compute_elbo_k` (lines ~190-273)
- Included terms:
  - `log p(z | gamma, delta, tau) + H[q(z)]`,
  - `log p(delta | lambda) + H[q(delta)]`,
  - `log p(gamma) + H[q(gamma)]`,
  - `log p(tau) + H[q(tau)]`,
  - `log p(lambda) + H[q(lambda)]`.

## Cohort wiring in configuration and data flow

- `src/model_config.py`:
  - `CohortPriorConfig` (hyperparameters) and validation,
  - `ModelConfig.create_z_priors(cohorts=...)`.
- `src/z_priors/factory.py`:
  - `ZPrior.COHORT` instantiation with per-factor `pi_k`.
- `src/views.py`:
  - `Views.cohorts` field + validation (`len == N`).
- `src/FA_model.py`:
  - passes `views.cohorts` to z-prior factory (`create_z_priors(cohorts=views.cohorts)`).

## About `starting_params_fa` parameter

- `FACTModel.__init__` now accepts:
  - `starting_params_fa: dict | None`
  - `starting_params_ctm: list[dict] | None`
- If `views.cohorts` is empty but `starting_params_fa["cohort_labels"]` is provided, the constructor injects these labels into `views.cohorts`.
- This keeps old calling style working:

```python
starting_params_fa = {"cohort_labels": cohort_labels}
model = FACTModel(views, K, model_config, seed=0, starting_params_fa=starting_params_fa)
```

- In `run_cohort.ipynb`, cohort training currently uses the preferred explicit path:
  - `views = Views.from_list(..., cohorts=cohort_codes)`
  - `z_priors=[ZPrior.COHORT] * K`
  - `model_cohort.fit(...)`

## How to run

### Manual (in notebook)
1. Open `run_cohort.ipynb`.
2. Run all cells from top to bottom.
3. Check generated files:
   - `cohort_elbo_plot.png`
   - `cohort_elbo_sequence.csv`

### CLI execution
```powershell
uv run jupyter nbconvert --to notebook --execute run_cohort.ipynb --output run_cohort.executed.ipynb --ExecutePreprocessor.timeout=0
```

## GPU note

Current implementation follows existing project style (`numpy` + `scipy`) and does not introduce a new tensor backend.  
For GPU acceleration, the core update equations in `src/z_priors/Cohort.py` and FA/CTM nodes would need a coordinated PyTorch/JAX backend rewrite (not just notebook-level changes).
