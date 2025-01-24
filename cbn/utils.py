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
