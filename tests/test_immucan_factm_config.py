import numpy as np

from immucan_data.factm_config import immucan_model_config, immucan_structured_configs
from src.enums import ZPrior
from src.views import SimpleView, StructuredView, Views


def test_immucan_model_config_includes_ctm() -> None:
    views = Views(
        simple=[SimpleView(np.zeros((3, 2)))],
        structured=[
            StructuredView([np.ones((4, 5)), np.ones((6, 5)), np.ones((2, 5))])
        ],
    )
    cfg = immucan_model_config(views, K=3, z_prior=ZPrior.COHORT, n_topics=10)
    assert len(cfg.simple_view_configs) == 1
    assert len(cfg.structured_view_configs) == 1
    assert cfg.structured_view_configs[0].L == 10
    assert len(cfg.z_priors) == 3


def test_immucan_structured_configs_empty() -> None:
    views = Views(simple=[SimpleView(np.zeros((2, 1)))])
    assert immucan_structured_configs(views) == []
