"""
Base class for Z prior implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FACTMpy.FA_model import nodeFA_z


class ZPriorBase(ABC):
    """Base class for all Z prior implementations."""

    @abstractmethod
    def update_z_k(
        self, z_node: "nodeFA_z", k: int, w_nodes: list, tau_nodes: list, y_nodes: list
    ):
        """Update Z node for factor k given this prior."""
        pass

    @abstractmethod
    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Whether this prior should update factor k."""
        pass
