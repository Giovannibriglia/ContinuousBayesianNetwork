from typing import List

import torch


def get_distribution_parameters(dist: torch.distributions.Distribution):
    """
    Retrieve the parameters of a PyTorch distribution.

    Args:
        dist (torch.distributions.Distribution): A PyTorch distribution object.

    Returns:
        dict: A dictionary of distribution parameters.
    """
    return {key: getattr(dist, key) for key in vars(dist) if not key.startswith("_")}


def multiply_distributions(
    distributions: List[torch.distributions.Distribution],
) -> torch.distributions.Distribution:
    """
    Multiplies a list of PyTorch distributions and returns the resulting distribution.
    Supports Normal, Beta, Bernoulli, and Gamma distributions.
    """
    if not distributions:
        raise ValueError("The list of distributions cannot be empty.")

    dist_types = {type(d) for d in distributions}

    if len(dist_types) > 1:
        raise ValueError(
            "All distributions must be of the same type for multiplication."
        )

    dist_type = distributions[0].__class__

    if dist_type == torch.distributions.Normal:
        # Multiply normal distributions
        precisions = torch.tensor([1 / d.variance for d in distributions])
        means = torch.tensor([d.mean for d in distributions])

        precision_C = precisions.sum()
        sigma_C2 = 1 / precision_C
        mu_C = sigma_C2 * (precisions * means).sum()

        return torch.distributions.Normal(mu_C, torch.sqrt(sigma_C2))

    elif dist_type == torch.distributions.Beta:
        # Multiply Beta distributions
        alpha_C = sum(d.concentration1 for d in distributions) - len(distributions) + 1
        beta_C = sum(d.concentration0 for d in distributions) - len(distributions) + 1
        return torch.distributions.Beta(alpha_C, beta_C)

    elif dist_type == torch.distributions.Bernoulli:
        # Multiply Bernoulli distributions
        probs = torch.tensor([d.probs for d in distributions])
        p_C = torch.prod(probs) / (torch.prod(probs) + torch.prod(1 - probs))
        return torch.distributions.Bernoulli(p_C)

    elif dist_type == torch.distributions.Gamma:
        # Multiply Gamma distributions
        alpha_C = sum(d.concentration for d in distributions)
        beta_C = sum(d.rate for d in distributions)
        return torch.distributions.Gamma(alpha_C, beta_C)

    else:
        raise NotImplementedError(f"Multiplication not implemented for {dist_type}")


def sum_distributions(
    distributions: List[torch.distributions.Distribution],
) -> torch.distributions.Distribution:
    """
    Sums a list of PyTorch distributions and returns the resulting distribution.
    Supports Normal, Gamma, Poisson, and Bernoulli (Binomial) distributions.
    """
    if not distributions:
        raise ValueError("The list of distributions cannot be empty.")

    dist_types = {type(d) for d in distributions}

    if len(dist_types) > 1:
        raise ValueError("All distributions must be of the same type for summation.")

    dist_type = distributions[0].__class__

    if dist_type == torch.distributions.Normal:
        # Sum of Normal distributions
        mu_C = sum(d.mean for d in distributions)
        sigma_C2 = sum(d.variance for d in distributions)
        return torch.distributions.Normal(mu_C, torch.sqrt(sigma_C2))

    elif dist_type == torch.distributions.Gamma:
        # Sum of Gamma distributions (assuming same rate parameter)
        rate = distributions[0].rate
        alpha_C = sum(d.concentration for d in distributions)
        return torch.distributions.Gamma(alpha_C, rate)

    elif dist_type == torch.distributions.Poisson:
        # Sum of Poisson distributions
        lambda_C = sum(d.rate for d in distributions)
        return torch.distributions.Poisson(lambda_C)

    elif dist_type == torch.distributions.Bernoulli:
        # Sum of Bernoulli distributions results in a Binomial
        p = distributions[0].probs
        n = len(distributions)
        return torch.distributions.Binomial(n, p)

    else:
        raise NotImplementedError(f"Summation not implemented for {dist_type}")
