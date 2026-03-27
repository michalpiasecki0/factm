"""
Model configuration for FACTM.

ViewConfig is split into SimpleViewConfig (Normal/Bernoulli) and
StructuredViewConfig (CTM) so the type system enforces the distinction
instead of runtime Likelihood checks.

ModelConfig holds the two lists separately, mirroring the Views container.
"""

from dataclasses import dataclass, field
from typing import List

from .enums import Likelihood, WPrior, ZPrior
from .likelihoods.fa.Bernoulli import BernoulliLikelihood
from .likelihoods.fa.Normal import NormalLikelihood
from .w_priors.factory import create_w_prior
from .z_priors.factory import create_z_priors


@dataclass
class SVIConfig:
    """
    Stochastic Variational Inference (SVI) configuration.

    This project uses Algorithm 3 from the MOFA+ appendix, where:
    - local parameters (Z) are updated on a minibatch of samples,
    - global parameters (W/tau and sparsity/noise hyperparameters) are updated
      using minibatch contributions scaled by `N/S`,
    - and global variational parameters are updated with a rho-mixing schedule.
    """

    enabled: bool = False

    # rho(t) = rho_tau / (1 + rho_kappa * t) ** rho_power
    rho_tau: float = 1.0
    rho_kappa: float = 0.01
    rho_power: float = 0.75
    rho_clip: tuple[float, float] = (0.0, 1.0)

    # Fraction S/N for sampling minibatch indices.
    # If None, falls back to ModelConfig.node_update_factor.
    batch_fraction: float | None = None


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
    node_update_factor:      fraction of view nodes to update per step (0, 1];
    """

    simple_view_configs: List[SimpleViewConfig]
    structured_view_configs: List[StructuredViewConfig]
    z_priors: List[ZPrior]
    svi: SVIConfig = field(default_factory=SVIConfig)
    node_update_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.simple_view_configs and not self.structured_view_configs:
            raise ValueError("Must have at least one view configuration.")
        if not self.z_priors:
            raise ValueError("Must have at least one Z prior.")

        if not (0 < self.node_update_factor <= 1):
            raise ValueError(
                "node_update_factor must be in (0, 1]; got "
                f"{self.node_update_factor!r}."
            )

        if not isinstance(self.svi, SVIConfig):
            raise TypeError("svi must be an instance of SVIConfig")

        if len(self.svi.rho_clip) != 2:
            raise ValueError("svi.rho_clip must be a tuple (min,max)")
        rho_min, rho_max = self.svi.rho_clip
        if not (0.0 <= rho_min <= rho_max <= 1.0):
            raise ValueError(
                f"svi.rho_clip must satisfy 0<=min<=max<=1; got {self.svi.rho_clip!r}"
            )

        if self.svi.batch_fraction is not None and not (
            0 < self.svi.batch_fraction <= 1
        ):
            raise ValueError(
                "svi.batch_fraction must be in (0, 1] or None; got "
                f"{self.svi.batch_fraction!r}."
            )

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
