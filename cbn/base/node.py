from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cbn.base import min_tolerance
from cbn.base.parameters_estimator import BaseParameterEstimator
from cbn.base.probability_estimator import BaseProbabilityEstimator
from cbn.parameters_estimators.mle import MaximumLikelihoodEstimator
from cbn.probability_estimators.parametric_estimator import (
    ParametricProbabilityEstimator,
)


class NodeVariable:
    def __init__(
        self,
        node_name: int | str,
        parents: List,
        device: str = "cpu",
        parameters_estimator_config: Dict = None,
        probability_estimator_config: Dict = None,
    ):
        self.node_name = node_name
        self.global_index = -1
        self.parents = parents
        self.n_features_tot = len(parents) + 1
        self.node_data = torch.zeros((1, 0))  # Initially no samples
        self.parents_data = torch.zeros((len(self.parents), 0))

        self.device = device

        self.parameters_estimator = self._setup_parameters_estimator(
            parameters_estimator_config
        )
        self.probability_estimator = self._setup_probability_estimator(
            probability_estimator_config
        )

    def set_global_index(self, index):
        self.global_index = index

    def _setup_parameters_estimator(
        self, estimator_config: Dict
    ) -> BaseParameterEstimator:
        if estimator_config is None:
            return MaximumLikelihoodEstimator(self.device)
        else:
            estimator_name = estimator_config["name"]
            kwargs = estimator_config["kwargs"]

            if estimator_name == "mle":
                return MaximumLikelihoodEstimator(self.device, **kwargs)
            elif estimator_name == "bayesian_estimator":
                raise NotImplementedError
            else:
                raise ValueError(f"estimator type {estimator_name} is not defined")

    def _setup_probability_estimator(
        self, estimator_config
    ) -> BaseProbabilityEstimator:
        if estimator_config is None:
            kwargs = {"min_tolerance": min_tolerance}
            return ParametricProbabilityEstimator("normal", self.device, **kwargs)
        else:
            estimator_type = estimator_config["type"]
            output_distribution = estimator_config["distribution"]
            kwargs = estimator_config["kwargs"]

            if estimator_type == "parametric":
                return ParametricProbabilityEstimator(
                    output_distribution, self.device, **kwargs
                )
            elif estimator_type == "non_parametric":
                raise NotImplementedError
            else:
                raise ValueError(f"estimator type {estimator_type} is not defined")

    def get_domain(self) -> torch.Tensor:
        return self.node_data.unique().to(self.device)

    def get_prior(self) -> torch.distributions.Distribution:
        return self.probability_estimator.compute_probability(self.node_data)

    def set_data(
        self,
        node_data: Optional[torch.Tensor] = None,
        parents_data: Optional[torch.Tensor] = None,
    ):
        if node_data is not None:
            if (
                isinstance(node_data, torch.Tensor)
                and node_data.dim() == 2
                and node_data.size(1) > 0
            ):
                self.node_data = node_data
            else:
                raise ValueError(
                    f"node_data must be a 2D tensor with shape (1, n_samples), got {node_data.shape if isinstance(node_data, torch.Tensor) else type(node_data)}"
                )

        if parents_data is not None:
            if len(self.parents) > 0:
                if isinstance(parents_data, torch.Tensor) and parents_data.shape == (
                    len(self.parents),
                    node_data.size(1),
                ):
                    self.parents_data = parents_data
                else:
                    raise ValueError(
                        f"parents_data must be a 2D tensor with shape ({len(self.parents)}, {node_data.size(1)}), got {parents_data.shape if isinstance(parents_data, torch.Tensor) else type(parents_data)}"
                    )
            else:
                # No parents, but parents_data was provided
                if parents_data.size(0) > 0:
                    raise ValueError(
                        f"Node {self.node_name} has no parents but received parents_data."
                    )

    def add_data(
        self,
        node_data: Optional[torch.Tensor] = None,
        parents_data: Optional[torch.Tensor] = None,
    ):
        if node_data is not None:
            if isinstance(node_data, torch.Tensor) and node_data.size(
                0
            ) == self.node_data.size(0):
                self.node_data = torch.cat((self.node_data, node_data), dim=1)
            else:
                raise ValueError(
                    f"node_data must have shape (1, n_samples), got {node_data.shape if isinstance(node_data, torch.Tensor) else type(node_data)}"
                )

        if parents_data is not None:
            if len(self.parents) > 0:
                if isinstance(parents_data, torch.Tensor) and parents_data.size(
                    0
                ) == self.parents_data.size(0):
                    self.parents_data = torch.cat(
                        (self.parents_data, parents_data), dim=1
                    )
                else:
                    raise ValueError(
                        f"parents_data must have shape ({len(self.parents)}, n_samples), got {parents_data.shape if isinstance(parents_data, torch.Tensor) else type(parents_data)}"
                    )
            else:
                raise ValueError(f"{self.node_name} has no parents")

    def get_cpd(
        self,
        evidence: Dict[int, Tuple[any, torch.Tensor]] = None,
        default_parents_uncertainty: float = 0.1,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Get the CPDs (mean and covariance) of the node based on the parents' evidence.

        Args:
            evidence (Dict[int, Tuple[float, Optional[float]]]): A dictionary mapping parent indices
                to their evidence values and optional uncertainties as (mean, uncertainty). If uncertainty is
                not provided, the default value will be used.
            default_parents_uncertainty (float): Default uncertainty to use if not specified for a parent.

        Returns:
            Tuple[torch.distributions.Distribution, torch.Tensor]: The initialized distribution object and node data.
        """

        if evidence is not None:
            evidence = evidence.copy()  # Avoid modifying the input dictionary

            # Extract node evidence (if any)
            target_node_evidence = evidence.pop(self.global_index, None)

            # Extract parent indices and corresponding evidence
            parent_indices = list(evidence.keys())

            # Extract means and uncertainties for parents
            means = torch.tensor(
                [evidence[i][0] for i in parent_indices],
                dtype=torch.float32,
                device=self.device,
            )
            uncertainties = torch.tensor(
                [
                    (
                        evidence[i][1]
                        if len(evidence[i]) > 1
                        else default_parents_uncertainty
                    )
                    for i in parent_indices
                ],
                dtype=torch.float32,
                device=self.device,
            )

            # Stack means and uncertainties for parents into a 2D tensor
            query_parents = {
                key: (mean, unc)
                for key, mean, unc in zip(evidence.keys(), means, uncertainties)
            }

            # Handle target node evidence if provided
            if target_node_evidence is not None:
                target_mean = torch.tensor(
                    [target_node_evidence[0]], dtype=torch.float32, device=self.device
                )
                target_uncertainty = torch.tensor(
                    (
                        [target_node_evidence[1]]
                        if len(target_node_evidence) > 1
                        else [default_parents_uncertainty]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                target_query = torch.stack([target_mean, target_uncertainty], dim=1)
            else:
                target_query = None

            # Select data based on evidence
            selected_data_by_estimator = self.parameters_estimator.return_data(
                self.node_data, self.parents_data, target_query, query_parents
            )
        else:
            # If no evidence is provided, return all data
            selected_data_by_estimator = self.parameters_estimator.return_data(
                self.node_data
            )

        return self.probability_estimator.compute_probability(
            selected_data_by_estimator
        ), self.node_data.unique().to(self.device)


if __name__ == "__main__":
    df = pd.read_pickle("../../frozen_lake.pkl")
    df.columns = ["obs_0", "action", "reward"]

    node_reward = NodeVariable("reward", ["obs_0", "action"])

    tensor_obs_0 = torch.from_numpy(df["obs_0"].values).float().unsqueeze(0)
    tensor_action = torch.from_numpy(df["action"].values).float().unsqueeze(0)
    tensor_reward = torch.from_numpy(df["reward"].values).float().unsqueeze(0)

    tensor_parents = torch.cat((tensor_obs_0, tensor_action), dim=0)

    node_reward.set_data(node_data=tensor_reward, parents_data=tensor_parents)

    """initialized_distribution = node_reward.get_cpds({0: (14, 0), 1: (2, 0)})

    print("PDF: ", initialized_distribution)
    print(get_distribution_parameters(initialized_distribution))

    if isinstance(initialized_distribution, torch.distributions.Categorical):
        y_values = torch.unique(tensor_reward)
    else:
        y_values = torch.linspace(
            torch.min(tensor_reward), torch.max(tensor_reward), 1000
        )

    pdf = initialized_distribution.log_prob(y_values)"""

    # plt.plot(y_values.cpu().numpy(), pdf.cpu().numpy())
    # plt.show()
    target_values, _ = torch.sort(torch.unique(tensor_reward))
    t_obs_0, _ = torch.sort(torch.unique(tensor_obs_0))
    t_act, _ = torch.sort(torch.unique(tensor_action))

    pdfs = torch.zeros(
        t_obs_0.shape[0],
        t_act.shape[0],
        target_values.shape[0],
    )
    for i, obs_0_unique in enumerate(torch.unique(tensor_obs_0)):
        for j, action_unique in enumerate(torch.unique(tensor_action)):

            cpd, target_values = node_reward.get_cpd(
                {0: (obs_0_unique, 0), 1: (action_unique, 0)}
            )
            pdfs[i, j, :] = cpd.log_prob(target_values)

    " *************************************************************************************************************** "

    # Create a 3D plot of the PDF without reducing along the z-axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create a grid for X, Y, and Z
    X, Y, Z = np.meshgrid(
        t_obs_0.cpu().numpy(),
        t_act.cpu().numpy(),
        target_values.cpu().numpy(),
        indexing="ij",
    )

    # Flatten the grids and PDF values for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    pdf_flat = pdfs.cpu().numpy().flatten()

    # Plot the PDF as scatter points in 3D
    sc = ax.scatter(X_flat, Y_flat, Z_flat, c=pdf_flat, cmap="berlin", s=100)

    # Add color bar and labels
    colorbar = plt.colorbar(sc, ax=ax)
    colorbar.set_label("PDF Value")
    ax.set_title("3D PDF of reward over observation-action")
    ax.set_xlabel("observation")
    ax.set_ylabel("action")
    ax.set_zlabel("reward")

    plt.show()

    # Create a 3D surface plot using the max over Z-dimension
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create a grid for X, Y
    X, Y = np.meshgrid(t_obs_0.cpu().numpy(), t_act.cpu().numpy(), indexing="ij")

    z_indices = np.argmax(
        pdfs.cpu().numpy(), axis=2
    )  # Indices of max PDF along the Z-dimension
    z_max_values = target_values[z_indices]  # Corresponding Z values

    # Plot the surface
    surf = ax.plot_surface(X, Y, z_max_values, cmap="berlin", edgecolor="none")

    # Add color bar and labels
    colorbar = plt.colorbar(surf, ax=ax)
    colorbar.set_label("PDF Value over reward)")
    ax.set_title("3D PDF of reward over obs-action")
    ax.set_xlabel("observation")
    ax.set_ylabel("action")
    ax.set_zlabel("reward")

    plt.show()
