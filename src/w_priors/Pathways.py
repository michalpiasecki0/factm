"""
Pathways prior for weights (not yet implemented).
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..FA_model import FAParams


class PathwaysWPrior:
    """Pathways prior for weights (placeholder for future implementation)."""

    @staticmethod
    def create_w_node(m, params: "FAParams"):
        """
        Create a W node for Pathways prior (placeholder).

        Args:
            m: View index
            params: FAParams object

        Returns:
            None (not yet implemented)
        """
        # TODO: Implement Pathways prior
        raise NotImplementedError("Pathways prior not yet implemented")
