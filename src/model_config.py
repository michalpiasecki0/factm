"""
Model configuration enums and types for FACTM.

This module provides type-safe enums for likelihoods and priors,
preventing bugs from typos and invalid values.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .likelihoods.fa.Bernoulli import BernoulliLikelihood
from .likelihoods.fa.Normal import NormalLikelihood


class FA_Pretrain(Enum):
    """Supported FA pretrain types."""

    PCA = "PCA"
    FA = "FA"


class CTM_pretrain(Enum):
    """Supported CTM pretrain types."""

    CTM = "CTM"


class Likelihood(Enum):
    """Supported likelihood types for data views."""

    NORMAL = "normal"
    BERNOULLI = "Bernoulli"
    CTM = "CTM"


class WPrior(Enum):
    """Supported weight prior types."""

    NONE = "None"
    ARD = "ARD"
    ARD_SS = "ARD_SS"
    PATHWAYS = "pathways"


class ZPrior(Enum):
    """Supported latent factor prior types."""

    STD_NORMAL = "stdN"
    INFORMED = "informed"


@dataclass
class ViewConfig:
    """
    Configuration for a single data view.

    Bundles together all view-specific settings to prevent
    errors from separate lists getting out of sync.

    For CTM views, L (number of topics/niches) must be provided.
    For other views, L should be None.
    """

    likelihood: Likelihood
    w_prior: WPrior
    L: int | None = None  # Number of topics for CTM views

    def __post_init__(self):
        """Validate that CTM views have L specified."""
        is_ctm = self.likelihood == Likelihood.CTM
        has_L = self.L is not None

        if is_ctm != has_L:
            raise ValueError(
                f"L must be specified if and only if likelihood is CTM. "
                f"Got likelihood={self.likelihood.value}, L={self.L}"
            )

    def __init__(
        self,
        likelihood: Likelihood,
        w_prior: WPrior,
        L: int | None = None,
    ):
        """
        Initialize ViewConfig.

        Args:
            likelihood: Likelihood
            w_prior: WPrior
            L: Number of topics for CTM views (required if likelihood is CTM)

        Returns:
            None
        """
        self.likelihood = likelihood
        self.w_prior = w_prior
        self.L = L

    def create_y_node(self, data_m, m, params):
        """
        Create a y node for this view's likelihood.

        Args:
            data_m: Data for view m
            m: View index
            params: FAParams object

        Returns:
            nodeFA_y_m instance
        """
        from .nodes import nodeFA_y_m

        if self.likelihood == Likelihood.CTM:
            # CTM views don't use standard y node in FA
            return nodeFA_y_m(None, m, params=params)
        else:
            # For FA likelihoods (Normal, Bernoulli), create y node
            # Normal and Bernoulli are handled the same way in FA
            if self.likelihood == Likelihood.NORMAL:
                likelihood_obj = NormalLikelihood()
                return likelihood_obj.create_y_node(data_m, m, params)
            elif self.likelihood == Likelihood.BERNOULLI:
                likelihood_obj = BernoulliLikelihood()
                return likelihood_obj.create_y_node(data_m, m, params)
            else:
                raise ValueError(f"Unknown FA likelihood: {self.likelihood}")

    def create_w_prior(self):
        """
        Create a W prior object for this view.

        Returns:
            WPriorBase instance
        """
        from .w_priors.factory import create_w_prior

        return create_w_prior(self.w_prior)


@dataclass
class ModelConfig:
    """
    Complete model configuration bundling all view and factor settings.

    This structure ensures consistency and makes the number of views explicit.
    """

    view_configs: List[ViewConfig]
    z_priors: List[ZPrior]

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.view_configs) == 0:
            raise ValueError("Must have at least one view configuration")
        if len(self.z_priors) == 0:
            raise ValueError("Must have at least one Z_prior")

    @property
    def num_views(self) -> int:
        """Number of data views."""
        return len(self.view_configs)

    @property
    def num_factors(self) -> int:
        """Number of latent factors."""
        return len(self.z_priors)

    @property
    def likelihoods(self) -> List[Likelihood]:
        """Extract likelihoods from view configs (for backward compatibility)."""
        return [config.likelihood for config in self.view_configs]

    @property
    def w_priors(self) -> List[WPrior]:
        """Extract W_priors from view configs (for backward compatibility)."""
        return [config.w_prior for config in self.view_configs]

    @property
    def L(self) -> List[int]:
        """Extract L values from CTM view configs."""
        L_values = []
        for config in self.view_configs:
            if config.likelihood == Likelihood.CTM:
                if config.L is None:
                    raise ValueError(
                        "CTM view missing L value. "
                        "This should have been caught during validation."
                    )
                L_values.append(config.L)
        return L_values

    def create_z_priors(self):
        """
        Create Z prior objects for all factors.

        Returns:
            List of ZPriorBase instances
        """
        from .z_priors.factory import create_z_priors

        return create_z_priors(self.z_priors)
