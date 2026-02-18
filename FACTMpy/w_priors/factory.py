"""
Factory functions to create prior objects from enums.
"""
import importlib

from model_config import WPrior
from w_priors.ARD import ARDWPrior
from w_priors.ARD_SS import ARD_SSWPrior
from w_priors.Pathways import PathwaysWPrior

# Import None prior using importlib since "None" is a Python keyword
_none_prior_module = importlib.import_module("w_priors.None")
NoneWPrior = _none_prior_module.NoneWPrior


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
