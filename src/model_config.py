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

    Full-batch ELBO need not increase every iteration under SVI: globals are
    estimated from a minibatch, while ELBO() sums over all samples. If the
    scheduled ``rho`` is exactly 1, there is no mixing with the previous
    globals—one noisy minibatch fully replaces them, which often causes a sharp
    full-batch ELBO drop. Use ``rho_tau < 1``, a tight ``rho_clip`` upper bound
    below 1, or ``warmup_iters`` to stabilise training.

    Attributes
    ----------
    warmup_iters:
        Number of main-loop iterations (after the first-fit block) that use the
        same global-update rule as non-SVI (full-data W/tau), while still using
        ``batch_fraction`` for Z. Helps avoid an immediate ELBO cliff when
        turning SVI on.

    first_svi_rho_multiplier:
        Applied only on the first SVI step (``_svi_t == 0``) *after* the
        scheduled ``rho`` is computed and *before* ``rho_clip`` is applied.
        Values < 1 make the first global update conservative (more weight on
        previous globals). Set to ``1.0`` to disable.

        **Note:** If ``rho_clip`` has a large *lower* bound (e.g. ``0.5``), the
        clip forces a high learning rate every step and can dominate this
        multiplier unless ``first_svi_rho_multiplier`` is very small. Prefer
        ``rho_clip=(0.0, 1.0)`` so the schedule can use small ``rho``.

    stochastic_fa_globals:
        If True (default), after ``warmup_iters`` the FA **global** parameters
        (W, τ, sparsity nodes) use minibatch statistics scaled by ``N/S``, and
        ``rho`` mixing applies — this is MOFA+ Algorithm 3.

        If False while ``enabled`` is True, training matches **non-SVI** for FA
        globals (full-data W/τ every step, no ρ schedule): only Z may use a
        minibatch via ``batch_fraction`` / ``node_update_factor``. Use this when
        you enable ``SVIConfig`` for other reasons but want the same ELBO path
        as ``enabled=False``.

        **Important:** ``rho_clip=(1.0, 1.0)`` only fixes the mixing
        coefficient to 1; it does **not** turn off minibatch global updates.
        With ``node_update_factor < 1``, the jump at epoch ``warmup_iters+1``
        is from switching W/τ to half-batch (or smaller) updates, not from ρ.
    """

    enabled: bool = False

    stochastic_fa_globals: bool = True

    # Learning rate schedule (MOFA+): rho(t) = tau / (1 + kappa * t)^(3/4)
    rho_tau: float = 0.5
    rho_kappa: float = 0.01
    rho_power: float = 0.75
    rho_clip: tuple[float, float] = (0.0, 1.0)

    # Minibatch size controls:
    # - if batch_size is set, use that
    # - else if batch_fraction is set, use that
    # - else fall back to ModelConfig.node_update_factor
    batch_size: int | None = None
    batch_fraction: float | None = None
    # Instead of scaling minibatch contributions by N/S, repeat the sampled
    # indices until length N (equivalent for sums; slower but simpler).
    repeat_minibatch: bool = True

    warmup_iters: int = 0

    # How often to compute/store ELBO during training.
    elbo_every: int = 10
    # ELBO is computed on full data.


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

        if self.svi.batch_size is not None and self.svi.batch_size <= 0:
            raise ValueError(
                f"svi.batch_size must be > 0 or None; got {self.svi.batch_size!r}."
            )

        if self.svi.rho_tau <= 0:
            raise ValueError(f"svi.rho_tau must be > 0; got {self.svi.rho_tau!r}.")
        if self.svi.rho_kappa < 0:
            raise ValueError(f"svi.rho_kappa must be >= 0; got {self.svi.rho_kappa!r}.")
        if self.svi.rho_power != 0.75:
            raise ValueError(f"svi.rho_power must be 0.75; got {self.svi.rho_power!r}.")

        if self.svi.warmup_iters < 0:
            raise ValueError(
                f"svi.warmup_iters must be >= 0; got {self.svi.warmup_iters!r}."
            )

        if self.svi.elbo_every <= 0:
            raise ValueError(
                f"svi.elbo_every must be > 0; got {self.svi.elbo_every!r}."
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
