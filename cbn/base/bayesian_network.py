from __future__ import annotations

from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import torch

from cbn.base.node import NodeVariable


class BayesianNetwork:
    def __init__(
        self, dag: nx.DiGraph, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ):
        """
        Initializes a Bayesian Network.

        Args:
            dag (nx.DiGraph): A directed acyclic graph representing the network structure.
            data (pd.DataFrame, np.ndarray, or torch.Tensor): A dataset containing samples for all variables.
        """
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(
                "The provided graph is not a directed acyclic graph (DAG)."
            )

        self.dag = dag
        self.nodes = {}

        # Convert data to tensor and create column mapping
        self.data, self.column_mapping = self._convert_data_to_tensor(data)

        # Create NodeVariable objects for each variable in the DAG
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            self.nodes[node] = NodeVariable(node_name=node, parents=parents)
            print(node, self.nodes[node].name)

        # Assign data to nodes
        self._assign_data_to_nodes()

    @staticmethod
    def _convert_data_to_tensor(data):
        """Converts data to a PyTorch tensor and creates a node-to-column mapping."""
        if isinstance(data, pd.DataFrame):
            tensor_data = torch.tensor(data.values, dtype=torch.float32)
            column_mapping = {
                col_name: idx for idx, col_name in enumerate(data.columns)
            }
        elif isinstance(data, np.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32)
            column_mapping = {f"{i}": i for i in range(data.shape[1])}
        elif isinstance(data, torch.Tensor):
            tensor_data = data
            column_mapping = {f"{i}": i for i in range(data.shape[1])}
        else:
            raise ValueError(
                "Data must be a pd.DataFrame, np.ndarray, or torch.Tensor."
            )

        # Debugging: Print the mapping
        # print("Column Mapping:", column_mapping)

        return tensor_data, column_mapping

    def _assign_data_to_nodes(self):
        """Splits the dataset and assigns node and parent data to each NodeVariable."""
        for node_name, node in self.nodes.items():
            # Get the column index for this node
            node_idx = self.column_mapping[node_name]

            # Extract the node's data
            node_data = self.data[:, node_idx].unsqueeze(0)  # Shape: (1, n_samples)

            # Extract parents' data
            if node.parents:
                parent_indices = [
                    self.column_mapping[parent] for parent in node.parents
                ]
                parents_data = self.data[
                    :, parent_indices
                ].T  # Shape: (len(parents), n_samples)
            else:
                parents_data = torch.zeros((0, self.data.size(0)), dtype=torch.float32)

            # Debugging: Print node and parent data
            # print(f"\nAssigning data for Node: {node_name}")
            # print(f"Node Data (for {node_name}): {node_data}")
            # print(f"Parents Data (for {node_name}): {parents_data}")

            # Assign the node and parent data
            node.set_data(node_data=node_data, parents_data=parents_data)

    def infer_cpds(
        self, node_name: str, parents_evidence: dict, uncertainty: float = 0.1
    ):
        """
        Infers CPDs for a given node given evidence for its parents.

        Args:
            node_name (str): The name of the node.
            parents_evidence (dict): Evidence for the parents as {parent_name: value}.
            uncertainty (float): Uncertainty interval for matching evidence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and covariance of the node's CPDs.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist.")

        # Convert evidence keys from column names to indices if needed
        translated_evidence = {
            self.column_mapping.get(key, key): (value, uncertainty)
            for key, value in parents_evidence.items()
        }

        # print(
        # "Translated Evidence:", translated_evidence
        # )  # Debug: Print the translated evidence

        # Pass the translated evidence to the node's CPD computation
        return self.nodes[node_name].get_cpds(translated_evidence)

    def print_structure(self):
        """Prints the structure of the Bayesian Network."""
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            print(f"Node: {node}, Parents: {parents}")


if __name__ == "__main__":
    # Create a dataset
    data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [0.5, 1.5, 2.5], "C": [0.2, 0.8, 1.2]}
    )
    # Create a DAG
    dag = nx.DiGraph()
    dag.add_edges_from([("A", "C"), ("B", "C")])

    # Initialize the Bayesian Network
    bn = BayesianNetwork(dag=dag, data=data)

    # Print the structure
    bn.print_structure()

    # Infer CPDs for node C given evidence for A and B
    evidence = {"A": 1.5, "B": 1.0}
    initialized_distribution = bn.infer_cpds("C", evidence, 1)
    print("Mean:", initialized_distribution)
