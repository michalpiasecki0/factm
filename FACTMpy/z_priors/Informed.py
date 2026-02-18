"""
Informed prior for latent factors Z (not yet implemented).
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FACTMpy.FA_model import FAParams


class InformedZPrior:
    """Informed prior for latent factors Z (placeholder for future implementation)."""

    @staticmethod
    def create_z_node(vi_mu, vi_var, params: "FAParams"):
        """
        Create a Z node with informed prior (placeholder).

        Args:
            vi_mu: Initial mean for Z
            vi_var: Initial variance for Z
            params: FAParams object

        Returns:
            None (not yet implemented)
        """
        # TODO: Implement Informed prior
        raise NotImplementedError("Informed Z prior not yet implemented")
