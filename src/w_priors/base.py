"""
Base class for weight prior implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..FA_model import FAParams, nodeFA_w_m


class WPriorBase(ABC):
    """Base class for all weight prior implementations."""

    @abstractmethod
    def create_nodes(
        self, m: int, params: "FAParams", starting_params: dict, D: int, K: int
    ) -> dict:
        """
        Create all nodes needed for this prior.

        Returns:
            dict with keys: w_m_node_not_sparse, hat_w_m_node,
            s_m_node, alpha_m_node, theta_m_node
            (None for nodes not used by this prior)
        """
        pass

    @abstractmethod
    def update_w_k(
        self,
        w_node: "nodeFA_w_m",
        k: int,
        z_node: Any,
        y_node: Any,
        tau_node: Any,
        indices: Any = None,
        data_scale: float = 1.0,
    ):
        """Update W node for factor k given this prior."""
        pass

    @abstractmethod
    def update_params(self, w_node: "nodeFA_w_m"):
        """Update E_w and E_w_squared based on this prior."""
        pass

    @abstractmethod
    def compute_elbo(self, w_node: "nodeFA_w_m") -> float:
        """Compute ELBO contribution for this prior."""
        pass

    @abstractmethod
    def get_additional_nodes_to_update(self) -> list:
        """Return list of additional node types
        that need updating (e.g., ['alpha', 'theta'])."""
        pass
