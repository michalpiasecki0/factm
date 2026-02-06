"""
Model configuration enums and types for FACTM.

This module provides type-safe enums for likelihoods and priors,
preventing bugs from typos and invalid values.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class Likelihood(Enum):
    """Supported likelihood types for data views."""

    NORMAL = "normal"
    BERNOULLI = "Bernoulli"
    CTM = "CTM"

    @classmethod
    def from_string(cls, value: str) -> "Likelihood":
        """Convert string to Likelihood enum, with validation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_values = [e.value for e in cls]
            raise ValueError(
                f"Invalid likelihood '{value}'. " f"Must be one of: {valid_values}"
            ) from e

    def __str__(self) -> str:
        """Return string value for backward compatibility."""
        return self.value


class WPrior(Enum):
    """Supported weight prior types."""

    NONE = "None"
    ARD = "ARD"
    ARD_SS = "ARD_SS"
    PATHWAYS = "pathways"

    @classmethod
    def from_string(cls, value: str) -> "WPrior":
        """Convert string to WPrior enum, with validation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_values = [e.value for e in cls]
            raise ValueError(
                f"Invalid W_prior '{value}'. " f"Must be one of: {valid_values}"
            ) from e

    def __str__(self) -> str:
        """Return string value for backward compatibility."""
        return self.value


class ZPrior(Enum):
    """Supported latent factor prior types."""

    STD_NORMAL = "stdN"
    INFORMED = "informed"

    @classmethod
    def from_string(cls, value: str) -> "ZPrior":
        """Convert string to ZPrior enum, with validation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_values = [e.value for e in cls]
            raise ValueError(
                f"Invalid Z_prior '{value}'. " f"Must be one of: {valid_values}"
            ) from e

    def __str__(self) -> str:
        """Return string value for backward compatibility."""
        return self.value


def validate_likelihoods(
    likelihoods: Union[List[str], List[Likelihood]], num_views: int
) -> List[Likelihood]:
    """
    Validate and convert likelihoods to enum list.

    Args:
        likelihoods: List of likelihood strings or enums
        num_views: Expected number of views

    Returns:
        List of Likelihood enums

    Raises:
        ValueError: If length doesn't match or values are invalid
    """
    if len(likelihoods) != num_views:
        raise ValueError(
            f"Number of likelihoods ({len(likelihoods)}) "
            f"must match number of views ({num_views})"
        )

    result = []
    for i, lik in enumerate(likelihoods):
        if isinstance(lik, Likelihood):
            result.append(lik)
        else:
            try:
                result.append(Likelihood.from_string(lik))
            except ValueError as e:
                raise ValueError(f"View {i}: {e}") from e

    return result


def validate_w_priors(
    w_priors: Union[List[str], List[WPrior], None],
    likelihoods: List[Likelihood],
    num_views: int,
) -> List[WPrior]:
    """
    Validate and convert W_priors to enum list with defaults.

    Args:
        w_priors: List of W_prior strings/enums or None for defaults
        likelihoods: List of likelihoods for each view
        num_views: Number of views

    Returns:
        List of WPrior enums
    """
    if w_priors is None:
        # Default: ARD for all views
        return [WPrior.ARD] * num_views

    if len(w_priors) != num_views:
        raise ValueError(
            f"Number of W_priors ({len(w_priors)}) "
            f"must match number of views ({num_views})"
        )

    result = []
    for i, prior in enumerate(w_priors):
        if isinstance(prior, WPrior):
            result.append(prior)
        else:
            try:
                result.append(WPrior.from_string(prior))
            except ValueError as e:
                raise ValueError(f"View {i}: {e}") from e

    return result


def validate_z_priors(
    z_priors: Union[List[str], List[ZPrior], None], num_factors: int
) -> List[ZPrior]:
    """
    Validate and convert Z_priors to enum list with defaults.

    Args:
        z_priors: List of Z_prior strings/enums or None for defaults
        num_factors: Number of latent factors

    Returns:
        List of ZPrior enums
    """
    if z_priors is None:
        # Default: stdN for all factors
        return [ZPrior.STD_NORMAL] * num_factors

    if len(z_priors) != num_factors:
        raise ValueError(
            f"Number of Z_priors ({len(z_priors)}) "
            f"must match number of factors ({num_factors})"
        )

    result = []
    for i, prior in enumerate(z_priors):
        if isinstance(prior, ZPrior):
            result.append(prior)
        else:
            try:
                result.append(ZPrior.from_string(prior))
            except ValueError as e:
                raise ValueError(f"Factor {i}: {e}") from e

    return result


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
    L: Optional[int] = None  # Number of topics for CTM views

    def __post_init__(self):
        """Validate that CTM views have L specified."""
        if self.likelihood == Likelihood.CTM and self.L is None:
            raise ValueError(
                f"CTM views must specify L (number of topics). "
                f"Got likelihood={self.likelihood.value}, L={self.L}"
            )
        if self.likelihood != Likelihood.CTM and self.L is not None:
            raise ValueError(
                f"Non-CTM views should not specify L. "
                f"Got likelihood={self.likelihood.value}, L={self.L}"
            )

    @classmethod
    def from_strings(
        cls,
        likelihood: Union[str, Likelihood],
        w_prior: Union[str, WPrior],
        L: Optional[int] = None,
    ) -> "ViewConfig":
        """
        Create ViewConfig from strings (convenience method).

        Args:
            likelihood: Likelihood string or enum
            w_prior: W_prior string or enum
            L: Number of topics for CTM views (required if likelihood is CTM)

        Returns:
            ViewConfig object
        """
        lik = (
            likelihood
            if isinstance(likelihood, Likelihood)
            else Likelihood.from_string(likelihood)
        )
        prior = w_prior if isinstance(w_prior, WPrior) else WPrior.from_string(w_prior)
        return cls(likelihood=lik, w_prior=prior, L=L)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "likelihood": self.likelihood.value,
            "w_prior": self.w_prior.value,
        }
        if self.L is not None:
            result["L"] = self.L
        return result


def create_view_configs(
    *configs: Union[
        ViewConfig,
        tuple[Union[str, Likelihood], Union[str, WPrior], Optional[int]],
    ],
) -> List[ViewConfig]:
    """
    Create multiple ViewConfig objects concisely.

    Args:
        *configs: Variable number of ViewConfig objects or tuples.
                  Tuples can be (likelihood, w_prior) or (likelihood, w_prior, L).
                  L is optional and only needed for CTM views.

    Returns:
        List of ViewConfig objects

    Examples:
        # Create multiple configs
        configs = create_view_configs(
            ('normal', 'ARD'),  # Normal view
            ('normal', 'ARD'),  # Another normal view
            ('CTM', 'None', 10),  # CTM view with L=10
            ('CTM', 'None', 10),  # Another CTM view with L=10
        )

        # Or mix ViewConfig objects and tuples
        configs = create_view_configs(
            ViewConfig(Likelihood.NORMAL, WPrior.ARD),
            ('CTM', 'None', 10),
        )
    """
    result = []
    for config in configs:
        if isinstance(config, ViewConfig):
            result.append(config)
        elif isinstance(config, tuple):
            if len(config) == 2:
                likelihood, w_prior = config
                result.append(ViewConfig.from_strings(likelihood, w_prior))
            elif len(config) == 3:
                likelihood, w_prior, L = config
                result.append(ViewConfig.from_strings(likelihood, w_prior, L))
            else:
                raise ValueError(
                    f"Tuple config must have 2 (likelihood, w_prior) or "
                    f"3 (likelihood, w_prior, L) elements. Got {len(config)}"
                )
        else:
            raise ValueError(f"Config must be ViewConfig or tuple. Got {type(config)}")
    return result


def create_view_configs_from_lists(
    likelihoods: List[Union[str, Likelihood]],
    w_priors: List[Union[str, WPrior]],
    L: Optional[List[int]] = None,
) -> List[ViewConfig]:
    """
    Create list of ViewConfig from separate lists (for backward compatibility).

    Args:
        likelihoods: List of likelihood strings or enums
        w_priors: List of W_prior strings or enums
        L: List of number of topics for CTM views (required for CTM views)

    Returns:
        List of ViewConfig objects

    Raises:
        ValueError: If lists have different lengths or CTM views missing L
    """
    if len(likelihoods) != len(w_priors):
        raise ValueError(
            f"Number of likelihoods ({len(likelihoods)}) "
            f"must match number of W_priors ({len(w_priors)})"
        )

    # Convert to enums for checking
    likelihoods_enum = [
        lik if isinstance(lik, Likelihood) else Likelihood.from_string(lik)
        for lik in likelihoods
    ]

    # Validate L for CTM views
    if L is not None:
        if len(L) != len(likelihoods):
            raise ValueError(
                f"Number of L values ({len(L)}) "
                f"must match number of views ({len(likelihoods)})"
            )
        for i, (lik, l_val) in enumerate(zip(likelihoods_enum, L, strict=True)):
            if lik == Likelihood.CTM and l_val is None:
                raise ValueError(
                    f"View {i}: CTM views must have L (number of topics) specified"
                )
    else:
        # Check if any CTM views need L
        ctm_indices = [
            i for i, lik in enumerate(likelihoods_enum) if lik == Likelihood.CTM
        ]
        if ctm_indices:
            raise ValueError(
                f"CTM views at indices {ctm_indices} require L values. "
                f"Please provide L parameter."
            )

    return [
        ViewConfig.from_strings(lik, prior, L[i] if L is not None else None)
        for i, (lik, prior) in enumerate(zip(likelihoods, w_priors, strict=True))
    ]


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

    @classmethod
    def from_legacy_format(
        cls,
        likelihoods: List[Union[str, Likelihood]],
        w_priors: List[Union[str, WPrior]],
        z_priors: Optional[List[Union[str, ZPrior]]] = None,
        num_factors: Optional[int] = None,
        L: Optional[List[int]] = None,
    ) -> "ModelConfig":
        """
        Create ModelConfig from legacy separate lists format.

        Args:
            likelihoods: List of likelihood strings or enums
            w_priors: List of W_prior strings or enums
            z_priors: List of Z_prior strings or enums, or None for defaults
            num_factors: Number of factors (required if z_priors is None)
            L: List of number of topics for CTM views (required for CTM views)

        Returns:
            ModelConfig object

        Raises:
            ValueError: If validation fails
        """
        view_configs = create_view_configs_from_lists(likelihoods, w_priors, L)

        if z_priors is None:
            if num_factors is None:
                raise ValueError("Must provide either z_priors or num_factors")
            z_priors_enum = [ZPrior.STD_NORMAL] * num_factors
        else:
            z_priors_enum = validate_z_priors(z_priors, len(z_priors))

        return cls(view_configs=view_configs, z_priors=z_priors_enum)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "view_configs": [config.to_dict() for config in self.view_configs],
            "z_priors": [prior.value for prior in self.z_priors],
            "num_views": self.num_views,
            "num_factors": self.num_factors,
        }
