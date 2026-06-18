"""
Factory functions to create Z prior objects from enums.
"""
from ..enums import ZPrior
from .base import ZPriorBase
from .Informed import InformedZPrior
from .StdNormal import StdNormalZPrior


def create_z_prior(prior_enum: ZPrior) -> ZPriorBase:
    """Create a Z prior object from an enum."""
    match prior_enum:
        case ZPrior.STD_NORMAL:
            return StdNormalZPrior()
        case ZPrior.INFORMED:
            return InformedZPrior()
        case _:
            raise ValueError(f"Unknown Z prior: {prior_enum}")


def create_z_priors(prior_enums: list[ZPrior]) -> list[ZPriorBase]:
    """Create a list of Z prior objects from enum list."""
    return [create_z_prior(prior_enum) for prior_enum in prior_enums]
