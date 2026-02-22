"""
Informed prior for latent factors Z (not yet implemented).
"""
from typing import TYPE_CHECKING

from .base import ZPriorBase

if TYPE_CHECKING:
    from ..FA_model import nodeFA_z


class InformedZPrior(ZPriorBase):
    """Informed prior for latent factors Z (placeholder for future implementation)."""

    def update_z_k(
        self, z_node: "nodeFA_z", k: int, w_nodes: list, tau_nodes: list, y_nodes: list
    ):
        """Update Z node for factor k for informed prior (not yet implemented)."""
        # TODO: Implement Informed prior
        raise NotImplementedError("Informed Z prior not yet implemented")

    def should_update_for_factor(self, k: int, z_priors: list) -> bool:
        """Informed prior should update informed factors."""
        # For now, informed factors are not updated (TBD)
        return False
