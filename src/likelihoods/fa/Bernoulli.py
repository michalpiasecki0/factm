"""
Bernoulli likelihood implementation for Factor Analysis.
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...FA_model import FAParams

from ...nodes import nodeFA_y_m


class BernoulliLikelihood:
    """Bernoulli likelihood for FA views."""

    def create_y_node(self, data_m, m, params: "FAParams") -> "nodeFA_y_m":
        """Create a y node for Bernoulli likelihood."""
        data_m = np.array(data_m)
        data_m = np.ma.array(data_m, mask=np.isnan(data_m))
        D = data_m.shape[1]
        node_y_m = nodeFA_y_m(data_m, m, params=params, D=D)
        node_y_m.data_mean = None
        return node_y_m
