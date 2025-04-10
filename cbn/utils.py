from typing import Dict

import torch

from cbn.base.parameter_learning import BaseParameterLearningEstimator
from cbn.inference import INFERENCE_OBJS
from cbn.parameter_learning import ESTIMATORS


def get_distribution_parameters(dist: torch.distributions.Distribution):
    """
    Retrieve the parameters of a PyTorch distribution.

    Args:
        dist (torch.distributions.Distribution): A PyTorch distribution object.

    Returns:
        dict: A dictionary of distribution parameters.
    """
    return {key: getattr(dist, key) for key in vars(dist) if not key.startswith("_")}


def choose_probability_estimator(
    estimator_name: str, config: Dict, **kwargs
) -> BaseParameterLearningEstimator:

    if estimator_name in ESTIMATORS.keys():
        estimator_class = ESTIMATORS[estimator_name](config, **kwargs)
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

    return estimator_class


def choose_inference_obj(inference_name: str, config: Dict, **kwargs):
    # TODO
    if inference_name in INFERENCE_OBJS.keys():
        pass
