"""
Base class for Z prior implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..FA_model import nodeFA_z


class ZPriorBase(ABC):
    """Base class for all Z prior implementations."""

    @abstractmethod
    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list[Any],
        tau_nodes: list[Any],
        y_nodes: list[Any],
        indices: "np.ndarray | None" = None,
    ) -> None:
        """Update variational parameters of Z for factor k."""
        ...

    @abstractmethod
    def should_update_for_factor(self, k: int, z_priors: list[Any]) -> bool:
        """Return whether factor k should be updated under this prior."""
        ...
