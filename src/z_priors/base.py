"""
Base class for Z prior implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..FA_model import nodeFA_z


class ZPriorBase(ABC):
    """Base class for all Z prior implementations."""

    @abstractmethod
    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list,
        tau_nodes: list,
        y_nodes: list,
        indices: np.ndarray | None = None,
    ):
        """Update Z node for factor k given this prior."""
        pass

    @abstractmethod
    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Whether this prior should update factor k."""
        pass

    def compute_elbo_k(self, z_node: "nodeFA_z", k: int) -> float | None:
        """Optional prior-specific ELBO term for one latent factor."""
        return None
