from typing import Dict

import gpytorch

import torch

from tqdm import tqdm

from cbn.base.parameter_learning import BaseParameterLearningEstimator
from cbn.parameter_learning.utils import config_torch_optimizer


class SimpleGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, mean_cls, kernel_cls, scale_kernel, device
    ):
        super().__init__(train_x, train_y, likelihood)

        self.device = device

        self.mean_module = mean_cls().to(self.device)
        kernel = kernel_cls().to(self.device)
        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                kernel,
            ).to(self.device)
            if scale_kernel
            else kernel
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_gpytorch(BaseParameterLearningEstimator):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_model(config, **kwargs)
        self.model = None

    def _setup_model(self, config: Dict, **kwargs):
        config_model = config.get("model", {})
        self.config_optimizer = config.get("optimizer")
        config_train = config.get("train")

        model_key = config_model.get("model", "ExactGP")
        self.model_class = getattr(gpytorch.models, model_key)

        likelihood_key = config_model.get("likelihood", "GaussianLikelihood")
        likelihood_class = getattr(gpytorch.likelihoods, likelihood_key)
        self.likelihood = likelihood_class().to(self.device)

        mll_key = config_model.get("mll", "ExactMarginalLogLikelihood")
        self.mll_class = getattr(gpytorch.mlls, mll_key)

        mean_key = config_model.get("mean", "ZeroMean")
        self.mean_class = getattr(gpytorch.means, mean_key)

        kernel_key = config_model.get("kernel", "RBFKernel")
        self.kernel_class = getattr(gpytorch.kernels, kernel_key)

        self.scale_kernel = config_model.get("scale_kernel", True)

        self.n_epochs = config_train.get("n_epochs", 1000)

    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        """
        Train the GP model. If parents_data is provided, treat it as X (inputs).
        If not, we create a dummy X as an index (0..N-1).
        """
        y = node_data

        if parents_data is not None:
            # [n_parents_features, n_samples] -> transpose to [n_samples, n_parents_features]
            X = parents_data.T.to(self.device)
        else:
            # If no features, use indices as X
            X = torch.arange(len(y), dtype=torch.float32, device=self.device).unsqueeze(
                -1
            )

        # Create the GP model
        self.model = SimpleGPModel(
            X,
            y,
            self.likelihood,
            self.mean_class,
            self.kernel_class,
            self.scale_kernel,
            self.device,
        ).to(self.device)

        # Train mode
        self.model.train()
        self.likelihood.train()

        optimizer = config_torch_optimizer(self.model, self.config_optimizer)
        mll = self.mll_class(self.likelihood, self.model)

        bar = (
            tqdm(range(self.n_epochs), desc="training gp...")
            if self.if_log
            else range(self.n_epochs)
        )

        for _ in bar:
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            if self.if_log:
                bar.set_postfix(loss=f"{loss.item():.4f}")

    def _get_prob(self, point_to_evaluate: torch.Tensor, query: torch.Tensor = None):
        """
        Given a query (X*), return an approximate PDF and domain grid for each query point.
        Here, we construct a 1D domain around the mean ± 3 std.
        The return shapes are:
          pdfs   -> [n_queries, num_points_in_domain]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() before get_prob().")

        n_queries, n_parents, n_values = query.shape

        # Eval mode
        self.model.eval()
        self.likelihood.eval()

        pdfs = torch.empty((n_queries, point_to_evaluate.shape[1]), device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(n_queries):
                pred_dist = self.likelihood(self.model(query[i].T))
                means = pred_dist.mean
                variances = pred_dist.variance

                for mu, var in zip(means, variances):
                    std = torch.sqrt(var)
                    pdf = (
                        1.0
                        / (
                            std
                            * torch.sqrt(
                                torch.tensor(2.0 * torch.pi, device=self.device)
                            )
                        )
                    ) * torch.exp(-0.5 * ((point_to_evaluate[i] - mu) / std) ** 2)
                    if torch.max(pdf) != 0:
                        pdfs[i] = pdf / torch.max(pdf)
                    else:
                        pdfs[i] = pdf

        return pdfs

    def _sample(self, N: int, **kwargs) -> torch.Tensor:
        """
        Return samples of shape [N, n_parents + 1],
        where each row is [p_1, ..., p_n, node_value].

        Since a GP only models p(node | parents), we must choose
        some way to sample parents. Below, we do it by sampling
        each parent dimension uniformly in [0, 1], then drawing the
        node from the GP's posterior.

        You can replace the uniform distribution with any
        other distribution that makes sense for your problem.
        """

        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() before _sample().")

        # Put model in eval mode to get posterior draws
        self.model.eval()
        self.likelihood.eval()

        # ----------------------------------------------------------------
        # 1) Figure out how many parent dimensions we have.
        #    We do this by looking at the model's defined input dimension.
        #    (Alternatively, you could store n_parents in __init__ or after _fit.)
        # ----------------------------------------------------------------
        # If you do NOT want to rely on self.model.train_inputs,
        # you must store n_parents yourself somewhere. For example:
        #   self.n_parents = ...
        # in _fit(), and just use that here.
        #
        # But if you truly cannot store anything, you need some known value
        # of n_parents. Below we retrieve it from the model's input shape:
        try:
            # This typically works for ExactGP:
            # self.model.train_inputs[0] => [num_training_points, n_parents]
            n_parents = self.model.train_inputs[0].shape[1]
        except Exception:
            raise RuntimeError(
                "Could not determine the parent dimension. Please store or specify it."
            )

        parents_data = kwargs["parents_data"]
        assert parents_data.shape[1] == n_parents, ValueError("n_parents must match.")

        n_queries, n_parents, n_values = parents_data.shape

        samples = torch.empty(n_queries, n_parents + 1, device=self.device)
        samples[:, :-1] = parents_data
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(n_queries):
                pred_dist = self.model(parents_data[i].T)
                samples[i, -1] = torch.exp(pred_dist.sample(torch.Size((N,))))

        return samples
