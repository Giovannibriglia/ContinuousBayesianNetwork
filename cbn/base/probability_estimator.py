from abc import ABC, abstractmethod

import torch


class BaseProbabilityEstimator(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_probability(
        self,
        data: torch.Tensor,
    ) -> torch.distributions.Distribution:

        self.check_input(data)

        probabilities = self._compute_probability(data)

        batch_size = data.shape[0]
        self.check_output(data, batch_size)

        return probabilities

    @abstractmethod
    def _compute_probability(
        self,
        data: torch.Tensor,
    ) -> torch.distributions.Distribution:
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

    @abstractmethod
    def check_output(self, prob, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def check_input(self, data: torch.Tensor):
        raise NotImplementedError
