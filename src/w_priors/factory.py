"""
Factory functions to create weight prior objects from enums.
"""
from ..enums import WPrior
from .ARD import ARDWPrior
from .ARD_SS import ARD_SSWPrior
from .base import WPriorBase
from .NoneWPrior import NoneWPrior


def create_w_prior(prior_enum: WPrior) -> WPriorBase:
    """Create a W prior object from an enum."""
    match prior_enum:
        case WPrior.NONE:
            return NoneWPrior()
        case WPrior.ARD:
            return ARDWPrior()
        case WPrior.ARD_SS:
            return ARD_SSWPrior()
        case WPrior.PATHWAYS:
            raise NotImplementedError("Pathways prior not yet implemented")
        case _:
            raise ValueError(f"Unknown W prior: {prior_enum}")
