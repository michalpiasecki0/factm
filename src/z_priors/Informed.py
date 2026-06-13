"""
Informed prior for latent factors Z (not yet implemented).
"""
from typing import TYPE_CHECKING, Any

from .base import ZPriorBase

if TYPE_CHECKING:
    import numpy as np

    from ..FA_model import nodeFA_z


class InformedZPrior(ZPriorBase):
    """
    Placeholder for an informed Z prior.

    Informed factors are held fixed during the standard Z update loop; use
    ``nodeFA_z.update_informed`` once that path is implemented.
    """

    def update_z_k(
        self,
        z_node: "nodeFA_z",
        k: int,
        w_nodes: list[Any],
        tau_nodes: list[Any],
        y_nodes: list[Any],
        indices: "np.ndarray | None" = None,
    ) -> None:
        raise NotImplementedError("Informed Z prior not yet implemented")

    def should_update_for_factor(self, k: int, z_priors: list[Any]) -> bool:
        """Informed factors are excluded from the default Z update loop."""
        return False
