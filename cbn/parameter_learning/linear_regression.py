from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from cbn.base.parameter_learning import BaseParameterLearningEstimator
from cbn.parameter_learning.utils import config_torch_optimizer


class LinearRegression(BaseParameterLearningEstimator):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_model(config, **kwargs)

        self.linear_model = None
        self.log_sigma = None

    def _setup_model(self, config: Dict, **kwargs):
        self.config_optimizer = config.get("optimizer")
        config_train = config.get("train", {})

        self.n_epochs = config_train.get("n_epochs", 1000)

    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        device = self.device

        # Determine the number of input features.
        # If parents_data is provided, use its first dimension as the input feature count.
        if parents_data is not None:
            # parents_data is assumed to be shaped [n_parents, n_samples]
            input_dim = parents_data.shape[0]
            # Transpose to get [n_samples, n_parents] for our training
            queries = parents_data.transpose(0, 1).to(device).to(torch.float32)
        else:
            # No parents provided implies the node is standalone.
            # We use a one-dimensional input derived from node_data.
            input_dim = 1
            queries = node_data.unsqueeze(1).to(device).to(torch.float32)

        # Setup the linear layer and the learnable log_sigma parameter if not already done.
        if self.linear_model is None:
            self.linear_model = nn.Linear(input_dim, 1).to(device).to(torch.float32)
            # log_sigma is initialized to log(1.0)
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

        # The target values are the node data; reshape to [n_samples, 1]
        targets = node_data.to(device).unsqueeze(1)

        # Set up an optimizer over both the linear layer parameters and log_sigma.
        optimizer = config_torch_optimizer(self.linear_model, self.config_optimizer)

        bar = (
            tqdm(range(self.n_epochs), desc="training linear regression...")
            if self.if_log
            else range(self.n_epochs)
        )

        # Training loop: we minimize the negative log likelihood for a Gaussian.
        for _ in bar:
            optimizer.zero_grad()
            # Predicted means from the query data.
            mu = self.linear_model(queries)  # shape: [n_samples, 1]
            sigma = torch.exp(self.log_sigma)
            # Gaussian negative log likelihood per sample:
            # nll = 0.5*log(2π) + log(sigma) + 0.5*((target - mu)/sigma)^2
            loss = (
                0.5 * torch.log(torch.tensor(2 * torch.pi, device=device))
                + self.log_sigma
                + 0.5 * ((targets - mu) / sigma) ** 2
            )
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            if self.if_log:
                bar.set_postfix(loss=f"{loss.item():.4f}")

    def _get_prob(self, point_to_evaluate: torch.Tensor, query: torch.Tensor = None):
        device = self.device
        # If a query is provided, use it to compute the conditional mean.
        if query is not None:
            # Expecting query shape [n_queries, n_features, 1] so squeeze the last dimension.
            q = query.squeeze(-1).to(device)
            mu = self.linear_model(q)  # shape: [n_queries, 1]
        else:
            # No query given; use a global mean from the bias.
            n_queries = point_to_evaluate.shape[0]
            mu = self.linear_model.weight.new_zeros(
                (n_queries, 1)
            ) + self.linear_model.bias.to(device)
        sigma = torch.exp(self.log_sigma)
        x = point_to_evaluate.to(device)
        pdf = (
            1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi, device=device)))
        ) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return pdf.detach()

    def _sample(self, N: int, **kwargs) -> torch.Tensor:
        device = self.device
        query = kwargs.get("query")
        if query is not None:
            q = query.squeeze(-1).to(device)
            mu = self.linear_model(q)  # shape: [n_queries, 1]
            sigma = torch.exp(self.log_sigma)
            # For each query, sample N points from Normal(mu, sigma)
            n_queries = mu.shape[0]
            samples = torch.normal(mu.expand(n_queries, N), sigma)
        else:
            # If no query is given, use the global bias.
            mu = self.linear_model.weight.new_zeros((1,)) + self.linear_model.bias.to(
                device
            )
            sigma = torch.exp(self.log_sigma)
            samples = torch.normal(mu.expand(N), sigma)
        return samples

    def save_model(self, path: str):
        torch.save(
            {
                "linear_state_dict": self.linear_model.state_dict(),
                "log_sigma": self.log_sigma.data,
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if self.linear_model is None:
            # In an ideal setup, you'd initialize the model beforehand
            raise RuntimeError(
                "Model not initialized. Train or initialize the model before loading."
            )
        self.linear_model.load_state_dict(checkpoint["linear_state_dict"])
        self.log_sigma.data = checkpoint["log_sigma"]
