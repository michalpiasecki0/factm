"""
Enum definitions for FACTM model configuration.

Kept separate from ``model_config`` to avoid circular imports.
"""
from enum import Enum


class FA_Pretrain(Enum):
    """FA weight initialisation strategies used in ``FACTModel.pretrain``."""

    PCA = "PCA"
    FA = "FA"


class Likelihood(Enum):
    """Likelihood type for a simple (FA) view."""

    NORMAL = "normal"
    BERNOULLI = "Bernoulli"
    CTM = "CTM"


class WPrior(Enum):
    """Weight prior for a view (simple or structured)."""

    NONE = "None"
    ARD = "ARD"
    ARD_SS = "ARD_SS"
    PATHWAYS = "pathways"


class ZPrior(Enum):
    """Latent factor prior, one entry per factor."""

    STD_NORMAL = "stdN"
    INFORMED = "informed"
