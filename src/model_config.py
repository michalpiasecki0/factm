"""
Model configuration for FACTM.

ViewConfig is split into SimpleViewConfig (Normal/Bernoulli) and
StructuredViewConfig (CTM) so the type system enforces the distinction
instead of runtime Likelihood checks.

ModelConfig holds the two lists separately, mirroring the Views container.
"""

from dataclasses import dataclass
from typing import List

from .enums import Likelihood, WPrior, ZPrior
from .likelihoods.fa.Bernoulli import BernoulliLikelihood
from .likelihoods.fa.Normal import NormalLikelihood
from .w_priors.factory import create_w_prior
from .z_priors.factory import create_z_priors

# ---------------------------------------------------------------------------
# Per-view configuration objects
# ---------------------------------------------------------------------------


@dataclass
class SimpleViewConfig:
    """
    Configuration for a single Normal or Bernoulli FA view.

    Attributes
    ----------
    likelihood: Likelihood.NORMAL or Likelihood.BERNOULLI
    w_prior:    WPrior enum value
    """

    likelihood: Likelihood
    w_prior: WPrior

    def __post_init__(self) -> None:
        if self.likelihood == Likelihood.CTM:
            raise ValueError(
                "SimpleViewConfig does not support CTM likelihood. "
                "Use StructuredViewConfig instead."
            )

    def create_y_node(self, data_m, m, params):
        """Create the FA y node for this view."""
        if self.likelihood == Likelihood.NORMAL:
            return NormalLikelihood().create_y_node(data_m, m, params)
        elif self.likelihood == Likelihood.BERNOULLI:
            return BernoulliLikelihood().create_y_node(data_m, m, params)
        else:
            raise ValueError(f"Unknown FA likelihood: {self.likelihood}")

    def create_w_prior(self):
        """Return a WPriorBase instance for this view."""
        return create_w_prior(self.w_prior)


@dataclass
class StructuredViewConfig:
    """
    Configuration for a single CTM (structured) view.

    Attributes
    ----------
    w_prior: WPrior enum value
    L:       number of topics / niches
    """

    w_prior: WPrior
    L: int

    @property
    def likelihood(self) -> Likelihood:
        """Always CTM — kept for interface consistency."""
        return Likelihood.CTM

    def create_w_prior(self):
        """Return a WPriorBase instance for this view."""
        return create_w_prior(self.w_prior)


# ---------------------------------------------------------------------------
# Top-level model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """
    Complete model configuration.

    Holds simple-view configs and structured-view configs as separate lists,
    mirroring the :class:`~views.Views` container and eliminating M-indexed
    mixed lists throughout the codebase.

    Attributes
    ----------
    simple_view_configs:     list of SimpleViewConfig (one per simple view)
    structured_view_configs: list of StructuredViewConfig (one per CTM view)
    z_priors:                list of ZPrior (one per latent factor)
    """

    simple_view_configs: List[SimpleViewConfig]
    structured_view_configs: List[StructuredViewConfig]
    z_priors: List[ZPrior]

    def __post_init__(self) -> None:
        if not self.simple_view_configs and not self.structured_view_configs:
            raise ValueError("Must have at least one view configuration.")
        if not self.z_priors:
            raise ValueError("Must have at least one Z prior.")

        self.configs_combined = self.simple_view_configs + self.structured_view_configs

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_simple(self) -> int:
        return len(self.simple_view_configs)

    @property
    def num_structured(self) -> int:
        return len(self.structured_view_configs)

    @property
    def num_factors(self) -> int:
        return len(self.z_priors)

    # Kept for backward-compatibility with FACTM/FA internals that still
    # iterate over a flat list indexed by m.
    @property
    def view_configs(self) -> list:
        """Flat list: simple configs first, then structured configs."""
        return self.configs_combined

    @property
    def likelihoods(self) -> List[Likelihood]:
        return [c.likelihood for c in self.configs_combined]

    @property
    def w_priors(self) -> List[WPrior]:
        return [c.w_prior for c in self.configs_combined]

    @property
    def L(self) -> List[int]:
        """L values for CTM views only."""
        return [c.L for c in self.structured_view_configs]

    def create_z_priors(self):
        """Create Z prior objects for all factors."""
        return create_z_priors(self.z_priors)
