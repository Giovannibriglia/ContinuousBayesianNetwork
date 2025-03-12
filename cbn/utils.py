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


def uniform_sample_tensor(
    input_tensor: torch.Tensor, n_samples: int = None
) -> torch.Tensor:
    """
    Uniformly selects n_samples from each row of input_tensor.

    Args:
        input_tensor (torch.Tensor): Tensor of shape [n_queries, n_values], sorted along dim=1.
        n_samples (int): Number of samples to uniformly cover the tensor domain.

    Returns:
        torch.Tensor: Tensor of shape [n_queries, n_samples] with uniformly sampled values,
                      or the original tensor if n_samples > n_values.
    """
    n_queries, n_values = input_tensor.shape

    if n_samples is None:
        return input_tensor

    if n_samples >= n_values:
        return input_tensor

    indices = torch.linspace(
        0, n_values - 1, steps=n_samples, device=input_tensor.device
    )
    indices = indices.round().long().clamp(max=n_values - 1)
    sampled_tensor = input_tensor[:, indices]

    return sampled_tensor
