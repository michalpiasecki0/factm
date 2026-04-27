"""
Enum definitions for FACTM model configuration.

This module contains all enum types used throughout the codebase.
Separated from model_config to avoid circular import dependencies.
"""

from enum import Enum


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
    COHORT = "cohort"
