import torch

from cbn.base.probability_estimator import BaseProbabilityEstimator
from cbn.probability_estimators import distribution_mapping


class ParametricProbabilityEstimator(BaseProbabilityEstimator):
    """Parametric estimator for specific distributions."""

    def __init__(self, distribution: str, device: str = "cpu", **kwargs):
        super().__init__(device)
        self.distribution_name = distribution.lower()
        self.parameters = {}

        self.min_tolerance = kwargs.get("min_tolerance", 1e-10)

        # Map supported torch distributions
        self.distribution_mapping = distribution_mapping

        if self.distribution_name not in self.distribution_mapping:
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

        # Store the distribution class
        self.distribution_class = self.distribution_mapping[self.distribution_name]

    def compute_probability(
        self,
        data: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """
        Compute the parameters for the specified parametric distribution.

        Args:
            data (torch.Tensor): The data for which to compute the parameters.
                                 Shape: [n_features, n_samples]

        Returns:
            torch.distributions.Distribution: The initialized distribution object.
        """
        # Compute distribution-specific parameters
        if self.distribution_name == "normal":
            self.parameters["loc"] = data.mean(dim=1).to(self.device)
            self.parameters["scale"] = torch.clamp(
                data.std(dim=1, unbiased=False).to(self.device), min=self.min_tolerance
            )

        elif self.distribution_name == "beta":
            mean = data.mean(dim=1)
            variance = data.var(dim=1, unbiased=False)
            if mean == 1 and variance == 0:
                alpha = torch.tensor(
                    [1e6], device=self.device
                )  # Approximation for "infinity"
                beta = torch.tensor([self.min_tolerance], device=self.device)
            else:
                alpha = mean * ((mean * (1 - mean)) / variance - 1)
                beta = (1 - mean) * ((mean * (1 - mean)) / variance - 1)

            self.parameters["concentration1"] = torch.clamp(
                alpha.to(self.device), min=self.min_tolerance
            )
            self.parameters["concentration0"] = torch.clamp(
                beta.to(self.device), min=self.min_tolerance
            )

        elif self.distribution_name == "categorical":
            unique, counts = torch.unique(data, return_counts=True, dim=1)
            probs = counts.float() / counts.sum()
            self.parameters["probs"] = probs.to(self.device)
            # self.parameters["categories"] = unique.to(self.device)

        elif self.distribution_name == "uniform":
            low = data.min(dim=1).values.to(self.device)
            high = data.max(dim=1).values.to(self.device)

            self.parameters["low"] = low
            self.parameters["high"] = high

        elif self.distribution_name == "gamma":
            mean = data.mean(dim=1)
            variance = torch.clamp(
                data.var(dim=1, unbiased=False), min=self.min_tolerance
            )
            concentration = mean**2 / variance
            rate = mean / variance
            self.parameters["concentration"] = torch.clamp(
                concentration.to(self.device), min=self.min_tolerance
            )
            self.parameters["rate"] = torch.clamp(
                rate.to(self.device), min=self.min_tolerance
            )

        elif self.distribution_name == "poisson":
            self.parameters["rate"] = torch.clamp(
                data.mean(dim=1).to(self.device), min=self.min_tolerance
            )

        elif self.distribution_name == "bernoulli":
            self.parameters["probs"] = torch.clamp(
                data.mean(dim=1).to(self.device), min=self.min_tolerance, max=1.0
            )

        elif self.distribution_name == "exponential":
            self.parameters["rate"] = torch.clamp(
                1 / data.mean(dim=1).to(self.device), min=self.min_tolerance
            )

        elif self.distribution_name == "dirichlet":
            self.parameters["concentration"] = torch.clamp(
                data.mean(dim=1).to(self.device), min=self.min_tolerance
            )

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

        # Initialize the distribution object
        distribution = self.distribution_class(**self.parameters)

        return distribution
