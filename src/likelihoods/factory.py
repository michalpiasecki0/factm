"""
Factory functions to create likelihood objects from enums.
"""
from ..model_config import Likelihood
from .ctm.CTM import CTMLikelihood
from .fa.Bernoulli import BernoulliLikelihood
from .fa.Normal import NormalLikelihood


def create_likelihood(likelihood_enum: Likelihood):
    """Create a likelihood object from an enum."""
    if likelihood_enum == Likelihood.NORMAL:
        return NormalLikelihood()
    elif likelihood_enum == Likelihood.BERNOULLI:
        return BernoulliLikelihood()
    elif likelihood_enum == Likelihood.CTM:
        return CTMLikelihood()
    else:
        raise ValueError(f"Unknown likelihood: {likelihood_enum}")


def create_likelihoods(likelihood_enums):
    """Create a list of likelihood objects from enum list."""
    return [create_likelihood(le) for le in likelihood_enums]
