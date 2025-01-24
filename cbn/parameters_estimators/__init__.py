import torch

distribution_mapping = {
    "normal": torch.distributions.Normal,
    "beta": torch.distributions.Beta,
    "categorical": torch.distributions.Categorical,
    "uniform": torch.distributions.Uniform,
    "gamma": torch.distributions.Gamma,
    "poisson": torch.distributions.Poisson,
    "bernoulli": torch.distributions.Bernoulli,
    "exponential": torch.distributions.Exponential,
    "dirichlet": torch.distributions.Dirichlet,
}
