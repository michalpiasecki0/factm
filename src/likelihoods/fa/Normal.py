"""
Normal likelihood implementation for Factor Analysis.
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...FA_model import FAParams

from ...nodes import nodeFA_y_m


class NormalLikelihood:
    """Normal likelihood for FA views."""

    def create_y_node(self, data_m, m, params: "FAParams") -> "nodeFA_y_m":
        """Create a y node for Normal likelihood."""
        data_m = np.array(data_m)
        data_m = np.ma.array(data_m, mask=np.isnan(data_m))
        feature_mean_m = np.ma.mean(data_m, axis=0)
        D = data_m.shape[1]
        node_y_m = nodeFA_y_m(data_m - feature_mean_m, m, params=params, D=D)
        node_y_m.data_mean = feature_mean_m
        return node_y_m
