"""
Base class for weight prior implementations.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..FA_model import FAParams, nodeFA_w_m


class WPriorBase(ABC):
    """Base class for all weight prior implementations."""

    @abstractmethod
    def create_nodes(
        self, m: int, params: "FAParams", starting_params: dict, D: int, K: int
    ) -> dict:
        """
        Create all nodes needed for this prior.

        Returns a dict with keys ``w_m_node_not_sparse``, ``hat_w_m_node``,
        ``s_m_node``, ``alpha_m_node``, and ``theta_m_node`` (``None`` when unused).
        """
        ...

    @abstractmethod
    def update_w_k(
        self,
        w_node: "nodeFA_w_m",
        k: int,
        z_node: Any,
        y_node: Any,
        tau_node: Any,
    ) -> None:
        """Update W node for factor k given this prior."""
        ...

    @abstractmethod
    def svi_target_w_k(
        self,
        w_node: "nodeFA_w_m",
        k: int,
        z_node: Any,
        y_node: Any,
        tau_node: Any,
        indices: "np.ndarray | None",
    ) -> dict:
        """
        Compute minibatch targets for factor k used in stochastic VI.

        Returned keys depend on the prior (e.g. ``mu``/``var`` or
        ``mu_hat``/``var_hat``/``lambda`` for spike-and-slab).
        """
        ...

    @abstractmethod
    def update_params(self, w_node: "nodeFA_w_m") -> None:
        """Update E_w and E_w_squared based on this prior."""
        ...

    @abstractmethod
    def compute_elbo(self, w_node: "nodeFA_w_m") -> float:
        """Compute ELBO contribution for this prior."""
        ...

    @abstractmethod
    def get_additional_nodes_to_update(self) -> list[str]:
        """Return node types that need updating beyond W (e.g. ``alpha``, ``theta``)."""
        ...
