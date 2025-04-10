from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from cbn.base.parameter_learning import BaseParameterLearningEstimator
from cbn.parameter_learning.utils import config_torch_optimizer

# Define a mapping of activation names to their classes
activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    # Add more as needed
}


class NeuralNetwork(BaseParameterLearningEstimator):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_model(config, **kwargs)

        self.nn_model = None
        self.log_scale = None

    def _setup_model(self, config: Dict, **kwargs):
        self.config_optimizer = config.get("optimizer", {})
        config_train = config.get("train", {})
        config_model = config.get("model", {})

        self.n_epochs = config_train.get("n_epochs", 1000)

        self.hidden_dims = config_model.get("hidden_dims", [32])
        activation_name = config_model.get("activation", "tanh").lower()
        if activation_name is None:
            raise ValueError(f"Unsupported activation: {activation_name}")
        self.activation = activation_map[activation_name]()

    def _build_nn(self, input_dim: int) -> nn.Module:
        # Build a sequential network: input_dim -> hidden layers -> output layer producing one value.
        layers = []
        current_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        device = self.device

        # Determine input dimension from parents_data (expected shape: [n_parents, n_samples]).
        if parents_data is not None:
            input_dim = parents_data.shape[0]
            # Transpose to shape [n_samples, n_parents].
            queries = parents_data.transpose(0, 1).to(device).to(torch.float32)
        else:
            input_dim = 1
            # For an intercept-only model, use a dummy input of ones.
            queries = torch.ones((node_data.shape[0], 1), device=device).to(
                torch.float32
            )

        # Build the neural network if it hasn't been created yet.
        if self.nn_model is None:
            self.nn_model = self._build_nn(input_dim).to(device).to(torch.float32)
            # Initialize log_scale to 0 so that scale = exp(0) = 1.
            self.log_scale = nn.Parameter(torch.tensor(0.0, device=device))

        # Targets are expected as binary labels with shape [n_samples, 1].
        targets = node_data.to(device).unsqueeze(1).float()
        optimizer = config_torch_optimizer(self.nn_model, self.config_optimizer)
        loss_fn = nn.BCEWithLogitsLoss()

        bar = (
            tqdm(range(self.n_epochs), desc="training neural network estimator...")
            if self.if_log
            else range(self.n_epochs)
        )

        for _ in bar:
            optimizer.zero_grad()
            logits = self.nn_model(queries)  # Shape: [n_samples, 1]
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            if self.if_log:
                bar.set_postfix(loss=f"{loss.item():.4f}")

    def _get_prob(self, point_to_evaluate: torch.Tensor, query: torch.Tensor = None):
        """
        Evaluates the logistic density at the provided points.
        :param point_to_evaluate: Tensor of shape [n_queries, n_points_to_evaluate]
                                  at which the density will be evaluated.
        :param query: Tensor of shape [n_queries, n_features, 1]. When provided, the network computes
                      the location parameter µ = nn_model(query.squeeze(-1)).
                      If None, a dummy input of ones is used.
        :return: A tensor of shape [n_queries, n_points_to_evaluate] representing the logistic density.
        """
        device = self.device
        if query is not None:
            # Squeeze the last dimension to obtain input shape [n_queries, n_features].
            q = query.squeeze(-1).to(device)
            mu = self.nn_model(q)  # Shape: [n_queries, 1]
        else:
            n_queries = point_to_evaluate.shape[0]
            dummy_input = torch.ones((n_queries, 1), device=device)
            mu = self.nn_model(dummy_input)
        # Compute scale from log_scale.
        scale = torch.exp(self.log_scale)
        # Evaluate the logistic density for each point:
        # f(x; µ, s) = exp(-(x-µ)/s) / [s * (1 + exp(-(x-µ)/s))^2]
        x = point_to_evaluate.to(device)  # Shape: [n_queries, n_points_to_evaluate]
        diff = (
            x - mu
        ) / scale  # Broadcasting mu from [n_queries,1] to [n_queries, n_points_to_evaluate]
        exp_term = torch.exp(-diff)
        pdf = exp_term / (scale * (1 + exp_term) ** 2)
        return pdf.detach()

    def _sample(self, N: int, **kwargs) -> torch.Tensor:
        """
        Samples class labels (0 or 1) from a Bernoulli distribution defined by the predicted probability.
        :param N: Number of samples per query.
        :param kwargs: Optional 'query' argument to condition on.
        :return: A tensor containing sampled labels.
        """
        device = self.device
        query = kwargs.get("query")
        if query is not None:
            q = query.squeeze(-1).to(device)
            logits = self.nn_model(q)
            prob = torch.sigmoid(logits)
            n_queries = prob.shape[0]
            prob_expanded = prob.expand(n_queries, N)
            samples = torch.bernoulli(prob_expanded)
        else:
            dummy_input = torch.ones((1, 1), device=device)
            logits = self.nn_model(dummy_input)
            prob = torch.sigmoid(logits)
            samples = torch.bernoulli(prob.expand(N))
        return samples

    def save_model(self, path: str):
        torch.save(
            {
                "nn_state_dict": self.nn_model.state_dict(),
                "log_scale": self.log_scale.data,
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if self.nn_model is None:
            raise RuntimeError(
                "Model not initialized. Train or initialize the model before loading."
            )
        self.nn_model.load_state_dict(checkpoint["nn_state_dict"])
        self.log_scale.data = checkpoint["log_scale"]
