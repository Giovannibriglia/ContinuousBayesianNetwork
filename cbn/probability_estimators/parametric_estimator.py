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
        self, data: torch.Tensor
    ) -> torch.distributions.Distribution:
        """
        Compute the parameters for the specified parametric distribution.

        Args:
            data (torch.Tensor): The data for which to compute the parameters.
                                 Shape: [n_queries, n_samples]

        Returns:
            A single `torch.distributions.Distribution` object with parameters
            of shape [n_queries, 1].
        """
        device = data.device  # Use the same device as the input data
        n_queries, _ = data.shape

        if self.distribution_name == "normal":
            loc = data.mean(dim=1, keepdim=True).to(device)  # [n_queries, 1]
            scale = torch.clamp(
                data.std(dim=1, unbiased=False, keepdim=True).to(device),
                min=self.min_tolerance,
            )
            dist = torch.distributions.Normal(loc, scale)

        elif self.distribution_name == "beta":
            mean = data.mean(dim=1, keepdim=True)  # [n_queries, 1]
            variance = torch.clamp(
                data.var(dim=1, unbiased=False, keepdim=True), min=self.min_tolerance
            )
            common_factor = mean * (1 - mean) / variance - 1
            alpha = torch.clamp(mean * common_factor, min=self.min_tolerance)
            beta = torch.clamp((1 - mean) * common_factor, min=self.min_tolerance)
            dist = torch.distributions.Beta(alpha.to(device), beta.to(device))

        elif self.distribution_name == "uniform":
            low = data.min(dim=1, keepdim=True).values.to(device)  # [n_queries, 1]
            high = data.max(dim=1, keepdim=True).values.to(device)  # [n_queries, 1]
            dist = torch.distributions.Uniform(low, high)

        elif self.distribution_name == "gamma":
            mean = data.mean(dim=1, keepdim=True)
            variance = torch.clamp(
                data.var(dim=1, unbiased=False, keepdim=True), min=self.min_tolerance
            )
            concentration = torch.clamp(mean**2 / variance, min=self.min_tolerance)
            rate = torch.clamp(mean / variance, min=self.min_tolerance)
            dist = torch.distributions.Gamma(concentration.to(device), rate.to(device))

        elif self.distribution_name == "poisson":
            rate = torch.clamp(
                data.mean(dim=1, keepdim=True), min=self.min_tolerance
            ).to(device)
            dist = torch.distributions.Poisson(rate)

        elif self.distribution_name == "bernoulli":
            p = torch.clamp(
                data.mean(dim=1, keepdim=True), min=self.min_tolerance, max=1.0
            ).to(device)
            dist = torch.distributions.Bernoulli(probs=p)

        elif self.distribution_name == "exponential":
            rate = torch.clamp(
                1.0 / data.mean(dim=1, keepdim=True), min=self.min_tolerance
            ).to(device)
            dist = torch.distributions.Exponential(rate)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

        return dist

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
