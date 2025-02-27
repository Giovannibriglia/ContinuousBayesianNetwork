from __future__ import annotations

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert data to tensor and create column mapping
        self.data, self.column_mapping = self._convert_data_to_tensor(data)

        # Create NodeVariable objects for each variable in the DAG
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            self.nodes[node] = NodeVariable(node_name=node, parents=parents)
            # print(node, self.nodes[node].node_name)

        # Assign data to nodes
        self._assign_data_to_nodes()

    def _convert_data_to_tensor(self, data):
        """Converts data to a PyTorch tensor and creates a node-to-column mapping."""
        if isinstance(data, pd.DataFrame):
            tensor_data = torch.tensor(
                data.values, dtype=torch.float32, device=self.device
            )
            column_mapping = {
                col_name: idx for idx, col_name in enumerate(data.columns)
            }
        elif isinstance(data, np.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32, device=self.device)
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

    def get_cpd_and_pdf(
        self, node_name: str, parents_evidence: Dict, uncertainty: float = 0.1
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
            self.column_mapping.get(key, key): (
                value,
                torch.tensor([uncertainty], device=self.device),
            )
            for key, value in parents_evidence.items()
        }

        # print(
        # "Translated Evidence:", translated_evidence
        # )  # Debug: Print the translated evidence

        # Pass the translated evidence to the node's CPD computation. It returns node_cpds and target_values
        node_cpd, target_values = self.nodes[node_name].get_cpd(translated_evidence)
        # Compute pdf
        node_pdf = node_cpd.log_prob(target_values)

        return node_cpd, node_pdf, target_values

    def get_all_cpds_and_pdfs(
        self, node_name: str, parents: List, uncertainty: float = 0.1
    ):

        parents_as_numbers = [self.column_mapping[parent] for parent in parents]
        uncertainty_as_tensor = torch.tensor([uncertainty])

        all_cpds = self.nodes[node_name].get_all_cpds_and_pdf(
            parents_as_numbers, uncertainty_as_tensor
        )

        renamed_dict = {
            col_name if key in self.column_mapping.values() else key: value
            for key, value in all_cpds.items()
            for col_name, index in self.column_mapping.items()
            if index == key or key not in self.column_mapping.values()
        }

        return renamed_dict

    def print_bn_structure(self):
        """Prints the structure of the Bayesian Network."""
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            print(f"Node: {node}, Parents: {parents}")

    @staticmethod
    def plot_cpds(distribution: Distribution, target_values: torch.Tensor):
        """
        Plots the probability mass function (PMF) for discrete distributions
        or the probability density function (PDF) for continuous distributions.

        Args:
            distribution (torch.distributions.Distribution): A PyTorch distribution instance.
            target_values (torch.Tensor): The values where the probability should be evaluated.
        """

        # Compute probabilities
        log_probs = distribution.log_prob(target_values)
        probs = torch.exp(log_probs)  # Convert log probabilities to probabilities
        # Plot
        plt.figure(dpi=500)
        plt.xlabel("Target values")
        if isinstance(distribution, torch.distributions.Categorical):
            # Categorical distribution -> PMF (bar plot)
            plt.bar(target_values.numpy(), probs.numpy(), width=0.01)
            plt.ylabel("Probability")
            plt.title(f"PMF of {distribution.__class__.__name__}")
            plt.ylim(ymin=-0.00, ymax=1.00)
        else:
            # Continuous distributions -> PDF (line plot)
            plt.plot(
                target_values.numpy(),
                probs.numpy(),
                color="blue",
                label=str(distribution),
            )
            plt.ylabel("Density")
            plt.title(f"PDF of {distribution.__class__.__name__}")
            plt.legend(loc="best")

        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Create a dataset
    """data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [0.5, 1.5, 2.5], "C": [0.2, 0.8, 1.2]}
    )
    # Create a DAG
    dag = nx.DiGraph()
    dag.add_edges_from([("A", "C"), ("B", "C")])"""

    data = pd.read_pickle("../../frozen_lake.pkl")
    data.columns = ["obs_0", "action", "reward"]

    dag = nx.DiGraph()
    dag.add_edges_from([("obs_0", "reward"), ("action", "reward")])

    # Initialize the Bayesian Network
    bn = BayesianNetwork(dag=dag, data=data)

    # Print the structure
    bn.print_bn_structure()

    # Infer CPDs for node C given evidence for A and B
    evidence = {
        "action": torch.tensor([2], device="cuda"),
        "obs_0": torch.tensor([14.0], device="cuda"),
    }  # ,
    cpd, pdf, target_values = bn.get_cpd_and_pdf("reward", evidence, 0)
    print("Conditional Distribution:", cpd)
    print("PDF: ", pdf)
    print("Target Values: ", target_values)
    bn.plot_cpds(cpd, target_values)

    dict_res = bn.get_all_cpds_and_pdfs("reward", ["action", "obs_0"])
    print(dict_res)
