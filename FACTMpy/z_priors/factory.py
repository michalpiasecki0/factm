"""
Factory functions to create Z prior objects from enums.
"""
from FACTMpy.model_config import ZPrior
from FACTMpy.z_priors.Informed import InformedZPrior
from FACTMpy.z_priors.StdNormal import StdNormalZPrior


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
