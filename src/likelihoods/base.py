"""
Base class for likelihood implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...FA_model import (
        FAParams,
        nodeFA_tau_m,
        nodeFA_w_m,
        nodeFA_y_m,
        nodeFA_z,
    )


class LikelihoodBase(ABC):
    """Base class for all likelihood implementations."""

    @abstractmethod
    def create_y_node(self, data_m, m, params: "FAParams") -> "nodeFA_y_m":
        """Create a y node for this likelihood."""
        pass

    @abstractmethod
    def create_tau_node(self, a0, b0, m, params: "FAParams") -> "nodeFA_tau_m":
        """Create a tau node for this likelihood."""
        pass

    @abstractmethod
    def update_w_k_for_likelihood(
        self, w_node: "nodeFA_w_m", k: int, z_node: "nodeFA_z", tau_node: "nodeFA_tau_m"
    ):
        """Update W node for factor k given this likelihood."""
        pass

    @abstractmethod
    def update_z_k_for_likelihood(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_node: "nodeFA_w_m",
        tau_node: "nodeFA_tau_m",
        y_node: "nodeFA_y_m",
    ):
        """Update Z node for factor k given this likelihood."""
        pass

    @abstractmethod
    def should_update_tau_after_w(self) -> bool:
        """Whether tau should be updated after W updates for this likelihood."""
        pass
