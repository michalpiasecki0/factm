"""
Factory functions to create Z prior objects from enums.
"""
from ..enums import ZPrior
from .Cohort import CohortZPrior
from .Informed import InformedZPrior
from .StdNormal import StdNormalZPrior


def _resolve_pi_k(pi, k: int, K: int) -> float:
    if isinstance(pi, list):
        if len(pi) == 1:
            return float(pi[0])
        if len(pi) != K:
            raise ValueError(
                f"Cohort prior pi list must have length 1 or K={K}, got {len(pi)}."
            )
        return float(pi[k])
    return float(pi)


def create_z_prior(
    prior_enum: ZPrior,
    k: int | None = None,
    K: int | None = None,
    cohort_prior_config=None,
    cohorts=None,
):
    """Create a Z prior object from an enum."""
    if prior_enum == ZPrior.STD_NORMAL:
        return StdNormalZPrior()
    elif prior_enum == ZPrior.INFORMED:
        return InformedZPrior()
    elif prior_enum == ZPrior.COHORT:
        if cohorts is None:
            raise ValueError(
                "Cohort Z prior requires cohort assignments in Views.cohorts."
            )
        if cohort_prior_config is None:
            raise ValueError("Cohort Z prior requires cohort_prior_config.")
        if k is None or K is None:
            raise ValueError("Cohort Z prior requires factor index metadata.")
        pi_k = _resolve_pi_k(cohort_prior_config.pi, k, K)
        return CohortZPrior(
            cohorts=cohorts,
            pi=pi_k,
            a_lambda=cohort_prior_config.a_lambda,
            b_lambda=cohort_prior_config.b_lambda,
            a0_tau=cohort_prior_config.a0_tau,
            b0_tau=cohort_prior_config.b0_tau,
        )
    else:
        raise ValueError(f"Unknown Z prior: {prior_enum}")


def create_z_priors(prior_enums, cohort_prior_config=None, cohorts=None):
    """Create a list of Z prior objects from enum list."""
    K = len(prior_enums)
    return [
        create_z_prior(
            pe,
            k=k,
            K=K,
            cohort_prior_config=cohort_prior_config,
            cohorts=cohorts,
        )
        for k, pe in enumerate(prior_enums)
    ]
