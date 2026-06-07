"""ModelConfig helpers for Immucan FACTM (simple + spatial CTM views)."""

from __future__ import annotations

from src.enums import Likelihood, WPrior, ZPrior
from src.model_config import (
    CohortPriorConfig,
    ModelConfig,
    SimpleViewConfig,
    StructuredViewConfig,
)
from src.views import Views

from .loader import DEFAULT_STRUCTURED_TOPICS


def immucan_structured_configs(
    views: Views,
    *,
    n_topics: int = DEFAULT_STRUCTURED_TOPICS,
    w_prior: WPrior = WPrior.NONE,
) -> list[StructuredViewConfig]:
    """One :class:`~model_config.StructuredViewConfig` per structured (CTM) view."""
    return [
        StructuredViewConfig(w_prior=w_prior, L=n_topics)
        for _ in range(views.num_structured)
    ]


def immucan_model_config(
    views: Views,
    K: int,
    *,
    z_prior: ZPrior,
    n_topics: int = DEFAULT_STRUCTURED_TOPICS,
    pi: float = 0.5,
    cohort_prior_config: CohortPriorConfig | None = None,
    structured_w_prior: WPrior = WPrior.NONE,
) -> ModelConfig:
    """
    Build :class:`~model_config.ModelConfig` for Immucan ``Views``.

    Simple views: Normal + ARD_SS. Structured views: CTM with ``L=n_topics``
    (default 10) per spatial panel.
    """
    kwargs: dict = {}
    if z_prior == ZPrior.COHORT:
        kwargs["cohort_prior_config"] = cohort_prior_config or CohortPriorConfig(pi=pi)
    return ModelConfig(
        simple_view_configs=[
            SimpleViewConfig(likelihood=Likelihood.NORMAL, w_prior=WPrior.ARD_SS)
            for _ in range(views.num_simple)
        ],
        structured_view_configs=immucan_structured_configs(
            views, n_topics=n_topics, w_prior=structured_w_prior
        ),
        z_priors=[z_prior] * K,
        **kwargs,
    )
