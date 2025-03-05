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

    def _compute_probability(
        self,
        data: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """
        Compute the parameters for the specified parametric distribution.

        Args:
            data (torch.Tensor): The data for which to compute the parameters.
                                 Shape: [batch_size, n_samples]

        Returns:
            torch.distributions.Distribution: The initialized distribution object.
        """
        device = data.device  # Use the same device as the input data

        # Compute distribution-specific parameters
        if self.distribution_name == "normal":
            self.parameters["loc"] = data.mean(dim=1).to(device)  # [batch_size]
            self.parameters["scale"] = torch.clamp(
                data.std(dim=1, unbiased=False).to(device), min=self.min_tolerance
            )  # [batch_size]

        elif self.distribution_name == "beta":
            mean = data.mean(dim=1)
            variance = torch.clamp(
                data.var(dim=1, unbiased=False), min=self.min_tolerance
            )
            common_factor = mean * (1 - mean) / variance - 1

            alpha = torch.clamp(mean * common_factor, min=self.min_tolerance)
            beta = torch.clamp((1 - mean) * common_factor, min=self.min_tolerance)

            self.parameters["concentration1"] = alpha.to(device)
            self.parameters["concentration0"] = beta.to(device)

        elif self.distribution_name == "categorical":
            unique, counts = torch.unique(data, return_counts=True, dim=1)
            probs = counts.float() / counts.sum(dim=1, keepdim=True)
            self.parameters["probs"] = probs.to(device)  # [batch_size, num_categories]

        elif self.distribution_name == "uniform":
            self.parameters["low"] = data.min(dim=1).values.to(device)  # [batch_size]
            self.parameters["high"] = data.max(dim=1).values.to(device)  # [batch_size]

        elif self.distribution_name == "gamma":
            mean = data.mean(dim=1)
            variance = torch.clamp(
                data.var(dim=1, unbiased=False), min=self.min_tolerance
            )

            concentration = torch.clamp(mean**2 / variance, min=self.min_tolerance)
            rate = torch.clamp(mean / variance, min=self.min_tolerance)

            self.parameters["concentration"] = concentration.to(device)
            self.parameters["rate"] = rate.to(device)

        elif self.distribution_name == "poisson":
            self.parameters["rate"] = torch.clamp(
                data.mean(dim=1).to(device), min=self.min_tolerance
            )

        elif self.distribution_name == "bernoulli":
            self.parameters["probs"] = torch.clamp(
                data.mean(dim=1).to(device), min=self.min_tolerance, max=1.0
            )

        elif self.distribution_name == "exponential":
            self.parameters["rate"] = torch.clamp(
                1 / data.mean(dim=1).to(device), min=self.min_tolerance
            )

        elif self.distribution_name == "dirichlet":
            self.parameters["concentration"] = torch.clamp(
                data.mean(dim=1).to(device), min=self.min_tolerance
            )

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

        # Initialize the distribution object
        distribution = self.distribution_class(**self.parameters)

        return distribution

    def check_output(self, prob, batch_size: int):
        """
        Check if the output parameters match the expected batch size.

        Args:
            prob (torch.distributions.Distribution): The probability distribution to check.
            batch_size (int): Expected batch size.
        """
        for param_name, param_value in self.parameters.items():
            assert (
                param_value.shape[0] == batch_size
            ), f"Mismatch in batch size for {param_name}: {param_value.shape}"

    def check_input(self, data: torch.Tensor):
        assert data.dim() == 2, f"data has {data.dim()} dimensions, instead 2"
