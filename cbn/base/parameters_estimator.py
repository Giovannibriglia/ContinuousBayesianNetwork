from abc import ABC, abstractmethod
from typing import Dict

import torch

from cbn.base import initial_uncertainty
from cbn.base.probability_estimator import BaseProbabilityEstimator
from cbn.probability_estimators.parametric_estimator import (
    ParametricProbabilityEstimator,
)


class BaseParameterLearning(ABC):
    """
    Base class for parametric and non-parametric estimators.
    """

    def __init__(
        self, config_probability_estimator: Dict = None, device: str = "cpu", **kwargs
    ):
        self.device = device
        self.probability_estimator = self._setup_probability_estimator(
            config_probability_estimator
        )

    def _setup_probability_estimator(
        self, config: Dict = None
    ) -> BaseProbabilityEstimator:

        if config is None:
            output_distribution = "normal"
            return ParametricProbabilityEstimator(output_distribution, self.device)
        # TODO
        output_distribution = "normal"
        return ParametricProbabilityEstimator(output_distribution, self.device)

    def get_cpd(
        self,
        target_node_index: int,
        evidence: torch.Tensor,
        data: torch.Tensor,
        uncertainty: float = initial_uncertainty,
    ):
        if evidence is not None:
            batch_size = evidence.shape[0]
            expanded_data = data.unsqueeze(0).expand(batch_size, -1, -1)

            evidence = evidence.to(data.device)

            selected_data = self._return_data(
                expanded_data, target_node_index, evidence, uncertainty
            )
            self._check_selected_data(expanded_data, selected_data)
        else:
            selected_data = data

        selected_data = selected_data.to(self.device)

        probabilities = self.probability_estimator.compute_probability(selected_data)

        return probabilities

    @abstractmethod
    def _return_data(
        self,
        data: torch.Tensor,
        target_node_index: int,
        evidence: torch.Tensor = None,
        uncertainty: float = initial_uncertainty,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _check_selected_data(self, data: torch.Tensor, selected_data: torch.Tensor):
        raise NotImplementedError
