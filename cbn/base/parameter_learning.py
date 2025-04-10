from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseParameterLearningEstimator(ABC):
    def __init__(self, config: Dict, **kwargs):
        self.estimator_name = config.get("estimator_name")
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.if_log = kwargs.get("log", False)

    @abstractmethod
    def _setup_model(self, config: Dict, **kwargs):
        raise NotImplementedError

    def fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        """
        :param node_data: [n_samples]
        :param parents_data: [n_parents_features, n_samples]
        """
        self._fit(node_data, parents_data)

    @abstractmethod
    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        raise NotImplementedError

    def get_prob(
        self,
        point_to_evaluate: torch.Tensor,
        query: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        :param point_to_evaluate: [n_queries, domain_node_feature]
        :param query: [n_queries, n_features, 1]
        :return: pdf [n_queries, domain_node_feature]
        """
        return self._get_prob(point_to_evaluate, query)

    @abstractmethod
    def _get_prob(
        self,
        point_to_evaluate: torch.Tensor,
        query: torch.Tensor = None,
    ):
        raise NotImplementedError

    def sample(self, N: int, **kwargs) -> torch.Tensor:
        return self._sample(N, **kwargs)

    @abstractmethod
    def _sample(self, N: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def save_model(self, path: str):
        raise NotImplementedError

    def load_model(self, path: str):
        raise NotImplementedError
