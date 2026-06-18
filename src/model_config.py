"""
Model configuration for FACTM.

Simple and structured views use separate config types, mirroring :class:`~views.Views`.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .enums import Likelihood, WPrior, ZPrior
from .likelihoods.fa.Bernoulli import BernoulliLikelihood
from .likelihoods.fa.Normal import NormalLikelihood
from .w_priors.base import WPriorBase
from .w_priors.factory import create_w_prior
from .z_priors.base import ZPriorBase
from .z_priors.factory import create_z_priors

if TYPE_CHECKING:
    from .FA_model import FAParams


@dataclass
class SVIConfig:
    """
    Stochastic Variational Inference (SVI) configuration (MOFA+ Algorithm 3).

    Local parameters (Z) update on a minibatch; global FA parameters (W, tau,
    sparsity nodes) use minibatch statistics with optional ``rho`` mixing.

    Full-batch ELBO need not increase every SVI step. If ``rho`` is clipped to 1,
    one noisy minibatch can fully replace globals and cause an ELBO drop — use
    ``rho_tau < 1``, ``warmup_iters``, or a tight ``rho_clip`` upper bound.

    ``stochastic_fa_globals=False`` keeps FA globals on the full-data update path
    while still allowing minibatch Z updates via ``batch_fraction`` /
    ``node_update_factor``.
    """

    enabled: bool = False
    stochastic_fa_globals: bool = True

    # rho(t) = rho_tau / (1 + rho_kappa * t)^rho_power
    rho_tau: float = 0.5
    rho_kappa: float = 0.01
    rho_power: float = 0.75
    rho_clip: tuple[float, float] = (0.0, 1.0)

    batch_size: int | None = None
    batch_fraction: float | None = None
    repeat_minibatch: bool = True

    warmup_iters: int = 0
    elbo_every: int = 10


@dataclass
class SimpleViewConfig:
    """Configuration for a single Normal or Bernoulli FA view."""

    likelihood: Likelihood
    w_prior: WPrior

    def __post_init__(self) -> None:
        if self.likelihood == Likelihood.CTM:
            raise ValueError(
                "SimpleViewConfig does not support CTM likelihood. "
                "Use StructuredViewConfig instead."
            )

    def create_y_node(self, data_m, m, params: "FAParams"):
        """Create the FA y node for this view."""
        match self.likelihood:
            case Likelihood.NORMAL:
                return NormalLikelihood().create_y_node(data_m, m, params)
            case Likelihood.BERNOULLI:
                return BernoulliLikelihood().create_y_node(data_m, m, params)
            case _:
                raise ValueError(f"Unknown FA likelihood: {self.likelihood}")

    def create_w_prior(self) -> WPriorBase:
        return create_w_prior(self.w_prior)


@dataclass
class StructuredViewConfig:
    """Configuration for a single CTM (structured) view."""

    w_prior: WPrior
    L: int

    @property
    def likelihood(self) -> Likelihood:
        return Likelihood.CTM

    def create_w_prior(self) -> WPriorBase:
        return create_w_prior(self.w_prior)


@dataclass
class ModelConfig:
    """
    Complete model configuration.

    Holds simple-view and structured-view configs in separate lists, mirroring
    :class:`~views.Views`.
    """

    simple_view_configs: list[SimpleViewConfig]
    structured_view_configs: list[StructuredViewConfig]
    z_priors: list[ZPrior]
    node_update_factor: float = 1.0
    svi: SVIConfig = field(default_factory=SVIConfig)

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

        if self.svi is None:
            self.svi = SVIConfig()
        if not isinstance(self.svi, SVIConfig):
            raise TypeError("svi must be an instance of SVIConfig")

        self._validate_svi(self.svi)

    @staticmethod
    def _validate_svi(svi: SVIConfig) -> None:
        if len(svi.rho_clip) != 2:
            raise ValueError("svi.rho_clip must be a tuple (min, max)")
        rho_min, rho_max = svi.rho_clip
        if not (0.0 <= rho_min <= rho_max <= 1.0):
            raise ValueError(
                f"svi.rho_clip must satisfy 0<=min<=max<=1; got {svi.rho_clip!r}"
            )

        if svi.batch_fraction is not None and not (0 < svi.batch_fraction <= 1):
            raise ValueError(
                "svi.batch_fraction must be in (0, 1] or None; got "
                f"{svi.batch_fraction!r}."
            )

        if svi.batch_size is not None and svi.batch_size <= 0:
            raise ValueError(
                f"svi.batch_size must be > 0 or None; got {svi.batch_size!r}."
            )

        if svi.rho_tau <= 0:
            raise ValueError(f"svi.rho_tau must be > 0; got {svi.rho_tau!r}.")
        if svi.rho_kappa < 0:
            raise ValueError(f"svi.rho_kappa must be >= 0; got {svi.rho_kappa!r}.")
        if svi.rho_power != 0.75:
            raise ValueError(f"svi.rho_power must be 0.75; got {svi.rho_power!r}.")

        if svi.warmup_iters < 0:
            raise ValueError(
                f"svi.warmup_iters must be >= 0; got {svi.warmup_iters!r}."
            )
        if svi.elbo_every <= 0:
            raise ValueError(f"svi.elbo_every must be > 0; got {svi.elbo_every!r}.")

    @property
    def num_simple(self) -> int:
        return len(self.simple_view_configs)

    @property
    def num_structured(self) -> int:
        return len(self.structured_view_configs)

    @property
    def num_factors(self) -> int:
        return len(self.z_priors)

    def create_z_priors(self) -> list[ZPriorBase]:
        return create_z_priors(self.z_priors)
