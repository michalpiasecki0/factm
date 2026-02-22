"""
Factory functions to create prior objects from enums.
"""
from ..enums import WPrior
from .ARD import ARDWPrior
from .ARD_SS import ARD_SSWPrior
from .NoneWPrior import NoneWPrior
from .Pathways import PathwaysWPrior


def create_w_prior(prior_enum: WPrior):
    """Create a W prior object from an enum."""
    if prior_enum == WPrior.NONE:
        return NoneWPrior()
    elif prior_enum == WPrior.ARD:
        return ARDWPrior()
    elif prior_enum == WPrior.ARD_SS:
        return ARD_SSWPrior()
    elif prior_enum == WPrior.PATHWAYS:
        return PathwaysWPrior()
    else:
        raise ValueError(f"Unknown W prior: {prior_enum}")


def create_w_priors(prior_enums):
    """Create a list of W prior objects from enum list."""
    return [create_w_prior(pe) for pe in prior_enums]
