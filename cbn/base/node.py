from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cbn.base.learning_parameters import BaseEstimator
from cbn.parameters_estimators.parametric_estimator import ParametricEstimator
from cbn.utils import get_distribution_parameters


class NodeVariable:
    def __init__(
        self,
        node_name: int | str,
        parents: List,
        device: str = "cpu",
        estimator_config: Dict = None,
    ):
        self.name = node_name
        self.parents = parents
        self.node_data = torch.zeros((1, 0))  # Initially no samples
        self.parents_data = torch.zeros((len(self.parents), 0))

        self.device = device

        self.estimator = self._setup_parameters_estimator(estimator_config)

    def _setup_parameters_estimator(self, estimator_config: Dict) -> BaseEstimator:
        if estimator_config is None:
            return ParametricEstimator("normal", self.device)
        else:
            estimator_type = estimator_config["type"]
            estimator_name = estimator_config["name"]
            kwargs = estimator_config["kwargs"]

            if estimator_type == "parametric":
                self.estimator = ParametricEstimator(
                    estimator_name, self.device, **kwargs
                )
            elif estimator_type == "non_parametric":
                raise NotImplementedError
            else:
                raise ValueError(f"estimator type {estimator_type} is not defined")

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
                        f"Node {self.name} has no parents but received parents_data."
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
                raise ValueError(f"{self.name} has no parents")

    def get_cpds(
        self,
        parents_evidence: Dict[int, Tuple[float, Optional[float]]],
        default_parents_uncertainty: float = 0.1,
    ) -> torch.distributions.Distribution:
        """
        Get the CPDs (mean and covariance) of the node based on the parents' evidence.

        Args:
            parents_evidence (Dict[int, Tuple[float, Optional[float]]]): A dictionary mapping parent indices
                to their evidence values and optional uncertainties as (mean, uncertainty). If uncertainty is
                not provided, the default value will be used.
            default_parents_uncertainty (float): Default uncertainty to use if not specified for a parent.

        Returns:
            torch.distributions.Distribution: The initialized distribution object.
        """
        if not parents_evidence:
            parents_evidence = {
                i: (torch.mean(self.parents_data[i]), default_parents_uncertainty)
                for i in range(len(self.parents))
            }

        if max(parents_evidence.keys()) >= len(self.parents):
            raise ValueError("Evidence keys must align with parent indices")

        # Extract data for relevant parent features
        parent_indices = list(parents_evidence.keys())
        parents_data_subset = self.parents_data[
            parent_indices, :
        ]  # Shape: (len(evidence), n_samples)

        # Extract means and uncertainties for parents
        means = torch.tensor(
            [parents_evidence[i][0] for i in parent_indices],
            device=self.parents_data.device,
        )
        uncertainties = torch.tensor(
            [
                (
                    parents_evidence[i][1]
                    if len(parents_evidence[i]) > 1
                    else default_parents_uncertainty
                )
                for i in parent_indices
            ],
            device=self.parents_data.device,
        )

        """# Debugging: Print information
        print(f"\nProcessing Node: {self.name}")
        print(f"Parents Data Subset (for {self.name}): {parents_data_subset}")
        print(f"Parents Means (for {self.name}): {means}")
        print(f"Parents Uncertainties (for {self.name}): {uncertainties}")"""

        # Create conditions for filtering data based on parent evidence
        conditions = [
            (parents_data_subset[i, :] >= mean - unc)
            & (parents_data_subset[i, :] <= mean + unc)
            for i, (mean, unc) in enumerate(zip(means, uncertainties))
        ]

        # Combine conditions across features
        combined_condition = torch.stack(conditions, dim=0).all(dim=0)

        # Filter data based on the combined condition
        selected_indices = combined_condition.nonzero(as_tuple=True)[0]

        if selected_indices.numel() == 0:
            raise ValueError(f"No matches found for Node: {self.name}. Returning NaNs.")

        # Select node data for this node
        selected_data_node = self.node_data[:, selected_indices]

        if isinstance(self.estimator, ParametricEstimator):
            return self.estimator.compute_parameters(selected_data_node)
        elif isinstance(self.estimator, None):
            all_data = self.parents_data[:, selected_indices]

            # Create points (means and uncertainties) for non-parametric estimator
            points = torch.stack(
                [means, uncertainties], dim=1
            )  # Shape: [len(parents), 2]

            # Use the non-parametric estimator to compute the PDF for the specific feature
            pdf, values, parameters = self.estimator.compute_parameters(
                data=all_data, points=points
            )

            raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_pickle("../../frozen_lake.pkl")
    df.columns = ["obs_0", "action", "reward"]

    node_reward = NodeVariable("reward", ["obs_0", "action"])

    tensor_obs_0 = torch.from_numpy(df["obs_0"].values).float().unsqueeze(0)
    tensor_action = torch.from_numpy(df["action"].values).float().unsqueeze(0)
    tensor_reward = torch.from_numpy(df["reward"].values).float().unsqueeze(0)

    tensor_parents = torch.cat((tensor_obs_0, tensor_action), dim=0)

    node_reward.set_data(node_data=tensor_reward, parents_data=tensor_parents)

    initialized_distribution = node_reward.get_cpds({0: (14, 0), 1: (2, 0)})

    print("PDF: ", initialized_distribution)
    print(get_distribution_parameters(initialized_distribution))

    if isinstance(initialized_distribution, torch.distributions.Categorical):
        y_values = torch.unique(tensor_reward)
    else:
        y_values = torch.linspace(
            torch.min(tensor_reward), torch.max(tensor_reward), 1000
        )

    pdf = initialized_distribution.log_prob(y_values)

    plt.plot(y_values.cpu().numpy(), pdf.cpu().numpy())
    plt.show()
