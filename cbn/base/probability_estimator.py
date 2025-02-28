from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseProbabilityEstimator(ABC):
    """
    Base class for parametric and non-parametric estimators.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def compute_probability(
        self,
        data: torch.Tensor,
    ) -> Tuple[torch.distributions.Distribution, dict]:
        """
        Compute the parameters of the distribution based on the input data.

        Args:
            data (torch.Tensor): The data for which to compute the parameters. Shape: [n_features, n_samples_data]

        Returns:
            Tuple[torch.distributions.Distribution, dict]:
                - The initialized distribution object.
                - A dictionary of computed parameters for the distribution.
        """
        raise NotImplementedError
