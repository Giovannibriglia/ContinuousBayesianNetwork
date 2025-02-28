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
        self, parents_data: torch.Tensor, node_data: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
