"""
Factory functions to create Z prior objects from enums.
"""
from model_config import ZPrior
from z_priors.Informed import InformedZPrior
from z_priors.StdNormal import StdNormalZPrior


def create_z_prior(prior_enum: ZPrior):
    """Create a Z prior object from an enum."""
    if prior_enum == ZPrior.STD_NORMAL:
        return StdNormalZPrior()
    elif prior_enum == ZPrior.INFORMED:
        return InformedZPrior()
    else:
        raise ValueError(f"Unknown Z prior: {prior_enum}")


def create_z_priors(prior_enums):
    """Create a list of Z prior objects from enum list."""
    return [create_z_prior(pe) for pe in prior_enums]
