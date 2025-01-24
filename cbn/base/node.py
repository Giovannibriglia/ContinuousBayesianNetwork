from typing import Dict, List, Optional

import pandas as pd
import torch


class NodeVariable:
    def __init__(self, node_name: int | str, parents: List):
        self.name = node_name
        self.parents = parents
        self.node_data = torch.zeros((1, 0))  # Initially no samples
        self.parents_data = torch.zeros((len(self.parents), 0))

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

    def get_cpds(self, parents_evidence: Dict[int, float], uncertainty: float = 0.1):
        """
        Get the CPDs (mean and covariance) of the node based on the parents' evidence.

        Args:
            parents_evidence (Dict[int, float]): A dictionary mapping parent indices to their evidence values.
            uncertainty (float): The range of uncertainty for matching the evidence.

        Returns:
            torch.Tensor, torch.Tensor: The mean and covariance of the node given the evidence.
        """
        if not parents_evidence:
            parents_evidence = {
                i: torch.mean(self.parents_data[i]) for i in range(len(self.parents))
            }

        if max(parents_evidence.keys()) >= len(self.parents):
            raise ValueError("Evidence keys must align with parent indices")

        # Extract data for relevant parent features
        parent_indices = list(parents_evidence.keys())
        parents_data_subset = self.parents_data[
            parent_indices, :
        ]  # Shape: (len(evidence), n_samples)

        """print(f"\nProcessing Node: {self.name}")
        print(f"Parents Data Subset (for {self.name}): {parents_data_subset}")
        print(f"Parents Evidence (for {self.name}): {parents_evidence}")"""

        # Create conditions
        conditions = [
            (parents_data_subset[i, :] >= value - uncertainty)
            & (parents_data_subset[i, :] <= value + uncertainty)
            for i, value in enumerate(parents_evidence.values())
        ]
        for i, cond in enumerate(conditions):
            print(f"Condition {i} (for parent {self.parents[i]}): {cond}")

        # Combine conditions across features
        combined_condition = torch.stack(conditions, dim=0).all(dim=0)
        # print(f"Combined Condition (for {self.name}): {combined_condition}")

        # Filter data based on the combined condition
        selected_indices = combined_condition.nonzero(as_tuple=True)[0]
        # print(f"Selected Indices (for {self.name}): {selected_indices}")

        if selected_indices.numel() == 0:
            print(f"No matches found for Node: {self.name}. Returning NaNs.")
            return torch.tensor([float("nan")]), torch.tensor([float("nan")])

        # Select node data for this node
        selected_node_data = self.node_data[:, selected_indices]
        # print(f"Selected Node Data (for {self.name}): {selected_node_data}")

        # Compute mean and covariance for the selected data
        mean = torch.mean(selected_node_data, dim=1)
        covariance = (
            torch.cov(selected_node_data)
            if selected_node_data.size(1) > 1
            else torch.tensor([[0.0]])
        )
        # print(f"Mean (for {self.name}): {mean}")
        # print(f"Covariance (for {self.name}): {covariance}")

        return mean, covariance


if __name__ == "__main__":
    df = pd.read_pickle("../../frozen_lake.pkl")
    df.columns = ["obs_0", "action", "reward"]

    node_reward = NodeVariable("reward", ["obs_0", "action"])

    tensor_obs_0 = torch.from_numpy(df["obs_0"].values).float().unsqueeze(0)
    tensor_action = torch.from_numpy(df["action"].values).float().unsqueeze(0)
    tensor_reward = torch.from_numpy(df["reward"].values).float().unsqueeze(0)

    tensor_parents = torch.cat((tensor_obs_0, tensor_action), dim=0)

    node_reward.set_data(node_data=tensor_reward, parents_data=tensor_parents)

    mean, uncertainty = node_reward.get_cpds({0: 14, 1: 2}, 1)

    print("Mean: ", mean)
    print("Uncertainty: ", uncertainty)
    print("******")
    mean, uncertainty = node_reward.get_cpds({0: 14, 1: 2}, 0.1)

    print("Mean: ", mean)
    print("Uncertainty: ", uncertainty)
