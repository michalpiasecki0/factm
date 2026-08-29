# CohortFACTM

Wariacyjny model czynnikowy (FACTM) z opcjonalnym prior kohortowym na czynnikach utajonych `Z`. Ten README opisuje użycie **z gotowymi danymi w Pythonie** — bez notebooków.

## Wymagania

- Python **≥ 3.13**
- [uv](https://docs.astral.sh/uv/)

## Instalacja

```bash
uv sync
```

Uruchamiaj skrypt z katalogu głównego repo (żeby import `src` działał).

## Kontrakt danych: `Views`

Wszystkie widoki muszą mieć tę samą liczbę próbek `N`.

| Typ | Kształt | Klasa |
|-----|---------|-------|
| prosty (FA) | `(N, D)` — macierz cech | `SimpleView` |
| strukturalny (CTM) | lista `N` macierzy `(n_i, G)` — np. okna spatial per próbka | `StructuredView` |

Etykiety kohort (wymagane dla `ZPrior.COHORT`): wektor długości `N`.

```python
import numpy as np
from src.views import SimpleView, StructuredView, Views

Y_proteom = ..
Y_rna = ...
Y_cells = ...
cohorts = np.array(["A", "AB", "B", ...]) 

views = Views(
    simple=[SimpleView(Y_proteom), SimpleView(Y_rna)],
    structured=[StructuredView(Y_cells)]
    cohorts=cohorts,
)
```

Skrót — `Views.from_list` rozpoznaje typ po elemencie listy (`ndarray` → prosty, `list[ndarray]` → strukturalny):

```python
views = Views.from_list([Y_proteom, Y_rna, Y_cells], cohorts=cohorts)
```

## Konfiguracja: `ModelConfig`
```python
from src.enums import Likelihood, WPrior, ZPrior
from src.model_config import (
    CohortPriorConfig,
    ModelConfig,
    SimpleViewConfig,
    StructuredViewConfig,
)

K = 10

cfg = ModelConfig(
    simple_view_configs=[
        SimpleViewConfig(likelihood=Likelihood.NORMAL, w_prior=WPrior.ARD_SS),
        SimpleViewConfig(likelihood=Likelihood.NORMAL, w_prior=WPrior.ARD_SS),
    ],
    structured_view_configs=[
        StructuredViewConfig(w_prior=WPrior.NONE, L=10),
    ],
    z_priors=[ZPrior.COHORT] * K,
    cohort_prior_config=CohortPriorConfig(pi=0.5),
)
```

Liczba wpisów w `simple_view_configs` / `structured_view_configs` musi się zgadzać z `views`.

## Dopasowanie

```python
from src import FACTModel

model = FACTModel(views=views, K=K, model_config=cfg, seed=0)
model.fit(max_iter=100, pretrain=True, elbo_tres=0.0)
```

- `pretrain=True` — inicjalizacja PCA/sklearn FA (+ krótki pretrain CTM przy widokach strukturalnych)
- `elbo_tres` — wcześniejsze zatrzymanie, gdy przyrost ELBO spadnie poniżej progu (0 = wyłączone)
- `model.elbo_sequence` — historia ELBO po iteracjach

## Wyniki

```python
Z = model.get_latent_factors()
W0 = model.get_loadings(0)
Y_hat = model.get_predictions_FA(0)

priors = model.fa.node_z.z_priors
E_gamma = np.column_stack([p.E_gamma for p in priors])
E_delta = np.column_stack([p.E_delta for p in priors])
```
