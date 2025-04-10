from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from cbn.base.parameter_learning import BaseParameterLearningEstimator
from cbn.parameter_learning.utils import config_torch_optimizer


class LogisticRegression(BaseParameterLearningEstimator):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_model(config, **kwargs)
        self.linear_model = None  # Will be set up in the _fit method.
        self.log_scale = None  # Learnable parameter to determine the logistic scale.

    def _setup_model(self, config: Dict, **kwargs):
        self.config_optimizer = config.get("optimizer", {})
        config_train = config.get("train", {})
        self.n_epochs = config_train.get("n_epochs", 1000)

    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        device = self.device

        # Determine the input dimensionality based on parents_data if available.
        if parents_data is not None:
            # Expected shape: [n_parents, n_samples]; transpose to [n_samples, n_parents].
            input_dim = parents_data.shape[0]
            queries = parents_data.transpose(0, 1).to(device)
        else:
            # For an intercept-only model (no parent features), use a dummy vector of ones.
            input_dim = 1
            queries = torch.ones((node_data.shape[0], 1), device=device)

        # Initialize the linear model and the logistic scale parameter if not already done.
        if self.linear_model is None:
            self.linear_model = nn.Linear(input_dim, 1).to(device)
            # Initialize log_scale to log(1)=0 so that scale = exp(0) = 1.
            self.log_scale = nn.Parameter(torch.tensor(0.0, device=device))

        # The targets are assumed to be binary labels of shape [n_samples, 1].
        targets = node_data.to(device).unsqueeze(1).float()

        # Set up the optimizer (optimizer configuration provided through config).
        optimizer = config_torch_optimizer(self.linear_model, self.config_optimizer)
        loss_fn = nn.BCEWithLogitsLoss()

        # Optionally use a progress bar if logging is enabled.
        bar = (
            tqdm(range(self.n_epochs), desc="training logistic regression...")
            if self.if_log
            else range(self.n_epochs)
        )

        for _ in bar:
            optimizer.zero_grad()
            logits = self.linear_model(queries)  # Shape: [n_samples, 1]
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            if self.if_log:
                bar.set_postfix(loss=f"{loss.item():.4f}")

    def _get_prob(self, point_to_evaluate: torch.Tensor, query: torch.Tensor = None):
        """
        Returns the logistic density evaluated at point_to_evaluate.
        This is interpreted as the probability density function of a logistic distribution
        with location parameter determined by the model and a learnable scale.

        :param point_to_evaluate: A tensor of shape [n_queries, n_points_to_evaluate] at which to evaluate the density.
        :param query: Tensor of shape [n_queries, n_features, 1]. If provided, the model computes µ as:
                      linear_model(query.squeeze(-1)). If None, an intercept-only model is used.
        :return: A tensor of shape [n_queries, n_points_to_evaluate] representing the logistic density.
        """
        device = self.device
        if query is not None:
            # Remove the trailing dimension to obtain shape [n_queries, n_features].
            q = query.squeeze(-1).to(device)
            mu = self.linear_model(q)  # Shape: [n_queries, 1]
        else:
            n_queries = point_to_evaluate.shape[0]
            dummy_input = torch.ones((n_queries, 1), device=device)
            mu = self.linear_model(dummy_input)
        # The scale is obtained as the exponential of log_scale (default is 1 if log_scale==0).
        scale = torch.exp(self.log_scale)
        # point_to_evaluate is expected to have shape: [n_queries, n_points_to_evaluate].
        x = point_to_evaluate.to(device)
        # Evaluate the logistic probability density function:
        # f(x; mu, scale) = exp(-(x-mu)/scale) / (scale * (1 + exp(-(x-mu)/scale))^2)
        diff = (
            x - mu
        ) / scale  # Broadcasting: mu shape [n_queries,1] -> diff shape [n_queries, n_points_to_evaluate].
        exp_term = torch.exp(-diff)
        pdf = exp_term / (scale * (1 + exp_term) ** 2)
        return pdf.detach()

    def _sample(self, N: int, **kwargs) -> torch.Tensor:
        """
        Samples class labels (0 or 1) using a Bernoulli distribution defined by the predicted probability
        from the logistic regression model.

        :param N: Number of samples to draw per query.
        :param kwargs: Optionally can include "query" to condition on.
        :return: A tensor with the sampled labels.
        """
        device = self.device
        query = kwargs.get("query")
        if query is not None:
            q = query.squeeze(-1).to(device)
            logits = self.linear_model(q)
            prob = torch.sigmoid(logits)
            n_queries = prob.shape[0]
            prob_expanded = prob.expand(n_queries, N)
            samples = torch.bernoulli(prob_expanded)
        else:
            dummy_input = torch.ones((1, 1), device=device)
            logits = self.linear_model(dummy_input)
            prob = torch.sigmoid(logits)
            samples = torch.bernoulli(prob.expand(N))
        return samples

    def save_model(self, path: str):
        torch.save(
            {
                "linear_state_dict": self.linear_model.state_dict(),
                "log_scale": self.log_scale.data,
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if self.linear_model is None:
            raise RuntimeError(
                "Model not initialized. Train or initialize the model before loading."
            )
        self.linear_model.load_state_dict(checkpoint["linear_state_dict"])
        self.log_scale.data = checkpoint["log_scale"]
