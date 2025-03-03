from abc import ABC, abstractmethod

import torch


class BaseParameterEstimator(ABC):
    """
    Base class for parametric and non-parametric estimators.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def return_data(
        self,
        node_data: torch.Tensor,
        parents_data: torch.Tensor = None,
        query_node: torch.Tensor = None,
        query_parents: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
