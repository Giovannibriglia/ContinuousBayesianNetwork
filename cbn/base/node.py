from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cbn.base.learning_parameters import BaseEstimator
from cbn.parameters_estimators.parametric_estimator import ParametricEstimator


class NodeVariable:
    def __init__(
        self,
        node_name: int | str,
        parents: List,
        device: str = "cpu",
        estimator_config: Dict = None,
    ):
        self.node_name = node_name
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
        parents_evidence: Dict[int, Tuple[any, torch.Tensor]],
        default_parents_uncertainty: float = 0.1,
    ) -> [torch.distributions.Distribution, torch.Tensor]:
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

        """# Create conditions for filtering data based on parent evidence
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
            raise ValueError(
                f"No matches found for Node: {self.node_name}. Returning NaNs. Try to increase uncertainty."
            )"""

        def find_min_uncertainty(
            parents_data_subset,
            means,
            uncertainties,
            min_factor=0.1,
            max_factor=100.0,
            tol=0.1,
        ):
            """
            Finds the minimum possible uncertainty that still results in at least one match.

            Parameters:
            - parents_data_subset: Tensor of shape [n_features, n_samples]
            - means: List or Tensor of means for each feature
            - uncertainties: List or Tensor of initial uncertainties
            - min_factor: Minimum scale factor for uncertainty
            - max_factor: Maximum scale factor for uncertainty
            - tol: Convergence tolerance for binary search

            Returns:
            - Optimized uncertainties that minimize `unc` while still selecting at least one sample.
            """
            min_unc = torch.tensor(
                [unc * min_factor for unc in uncertainties],
                device=parents_data_subset.device,
            )
            max_unc = torch.tensor(
                [unc * max_factor for unc in uncertainties],
                device=parents_data_subset.device,
            )

            while (max_unc - min_unc).max() > tol:
                mid_unc = (min_unc + max_unc) / 2  # Binary search midpoint

                # Compute conditions with the current mid_unc
                conditions = [
                    (parents_data_subset[i, :] >= mean - unc)
                    & (parents_data_subset[i, :] <= mean + unc)
                    for i, (mean, unc) in enumerate(zip(means, mid_unc))
                ]

                combined_condition = torch.stack(conditions, dim=0).all(dim=0)
                selected_indices = combined_condition.nonzero(as_tuple=True)[0]

                if selected_indices.numel() > 0:
                    max_unc = mid_unc  # Decrease uncertainty to minimize it
                else:
                    min_unc = (
                        mid_unc  # Increase uncertainty to allow at least one match
                    )

            return max_unc  # Smallest working uncertainty

        # Find minimal uncertainty
        optimized_uncertainties = find_min_uncertainty(
            parents_data_subset, means, uncertainties, default_parents_uncertainty
        )

        # Use the optimized uncertainty to filter data
        conditions = [
            (parents_data_subset[i, :] >= mean - unc)
            & (parents_data_subset[i, :] <= mean + unc)
            for i, (mean, unc) in enumerate(zip(means, optimized_uncertainties))
        ]

        combined_condition = torch.stack(conditions, dim=0).all(dim=0)
        selected_indices = combined_condition.nonzero(as_tuple=True)[0]

        if selected_indices.numel() == 0:
            raise ValueError(
                f"No matches found for Node: {self.node_name}. Unexpected failure in binary search."
            )

        # Select node data for this node
        selected_data_node = self.node_data[:, selected_indices]

        if isinstance(self.estimator, ParametricEstimator):
            return (
                self.estimator.compute_parameters(selected_data_node),
                self.node_data.unique().to(self.device),
            )
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

    def get_all_cpds_and_pdf(
        self, parents: list, uncertainty: torch.Tensor, max_values: int = 10
    ):
        unique_values_start = [
            torch.unique(feature_values)
            for feature_values in self.parents_data[parents]
        ]

        unique_values_parents = self.reduce_unique_values_list(
            unique_values_start, max_values
        )

        # Compute all possible combinations of unique parent values
        combinations = torch.cartesian_prod(*unique_values_parents)

        # Get unique values of the target feature (store separately)
        unique_values_target_feature = self.node_data.unique(sorted=True)

        # Dictionary to store CPDs and parent unique values
        cpds_dict = {
            key: torch.zeros((combinations.shape[0],), device=self.device)
            for n, key in enumerate(parents)  # Store unique values for each parent
        }
        cpds_dict[self.node_name] = torch.tensor(
            unique_values_target_feature.unsqueeze(0).expand(combinations.shape[0], -1),
            device=self.device,
        )  # Store unique target feature values

        # **Predefine a list for CPDs** (since distributions are objects)
        cpds_dict["cpds"] = [None] * combinations.shape[0]  # Pre-allocate list

        # **Predefine a list for CPDs** (since distributions are objects)
        cpds_dict["pdfs"] = torch.zeros(
            (combinations.shape[0], len(unique_values_target_feature)),
            device=self.device,
        )  # Pre-allocate list

        # cpds_dict["combinations"] = combinations

        # Iterate over each parent value combination
        for i, comb in tqdm(
            enumerate(combinations),
            total=len(combinations),
            desc="computing all cpds and pdfs...",
        ):
            parents_evidence = {}
            for n, par_value in enumerate(comb):
                parents_evidence[parents[n]] = (par_value, uncertainty)

                cpds_dict[parents[n]][i] = par_value

            # Get the conditional probability distribution (cpd) for the given parent evidence
            cpd, _ = self.get_cpd(parents_evidence)

            # Store CPD in the pre-allocated list
            cpds_dict["cpds"][i] = cpd

            cpds_dict["pdfs"][i, :] = cpd.log_prob(unique_values_target_feature)

        return cpds_dict

    @staticmethod
    def reduce_unique_values(unique_values_dict, N):
        """
        Reduces the number of unique values per feature to at most N while ensuring
        the selected values cover the full range.

        Args:
            unique_values_dict (dict): A dictionary where keys are features and
                                       values are torch.Tensors of unique values.
            N (int): Maximum number of unique values per feature.

        Returns:
            dict: A dictionary with at most N values per feature, covering the range.
        """
        reduced_dict = {}

        for feature, unique_values in unique_values_dict.items():
            unique_values = torch.sort(unique_values).values  # Ensure sorted order
            num_values = unique_values.numel()

            if num_values <= N:
                reduced_dict[feature] = unique_values
            else:
                # Select N well-spread values across the range
                indices = torch.linspace(0, num_values - 1, steps=N).long()
                reduced_dict[feature] = unique_values[indices]

        return reduced_dict

    @staticmethod
    def reduce_unique_values_list(unique_values_list, N):
        """
        Reduces the number of unique values in each tensor to at most N, ensuring
        well-distributed selection across the range.

        Args:
            unique_values_list (list of torch.Tensor): List of tensors, each containing unique values.
            N (int): Maximum number of unique values per tensor.

        Returns:
            list of torch.Tensor: Reduced list with at most N values per tensor.
        """
        reduced_list = []

        for unique_values in unique_values_list:
            unique_values = torch.sort(unique_values).values  # Ensure sorted order
            num_values = unique_values.numel()

            if num_values <= N:
                reduced_list.append(unique_values)
            else:
                indices = torch.linspace(
                    0, num_values - 1, steps=N
                ).long()  # Evenly spaced indices
                reduced_list.append(unique_values[indices])

        return reduced_list


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
