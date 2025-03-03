from __future__ import annotations

from typing import Dict, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution

from cbn.base import min_tolerance
from cbn.base.inference import BaseInference
from cbn.base.node import NodeVariable
from cbn.inference.exact_inference import ExactInference, VariableElimination


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

        self.inference = self._setup_inference()

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
            node.set_global_index(node_idx)
            node.set_data(node_data=node_data, parents_data=parents_data)

    def _setup_inference(self, inference_config: Dict = None) -> BaseInference:
        if inference_config is None:
            return ExactInference(self)
        else:
            # inference_type = inference_config["type"]
            inference_technique = inference_config["technique"]
            kwargs = inference_config["kwargs"]

            if inference_technique == "exact_inference":
                return ExactInference(self, **kwargs)
            elif inference_technique == "variable_elimination":
                return VariableElimination(self, **kwargs)
            elif inference_technique == "belief_propagation":
                raise NotImplementedError
            else:
                raise ValueError(f"estimator type {inference_technique} is not defined")

    def get_cpd_and_pdf(
        self, node_name: str, evidence: Dict, uncertainty: float = 0.1
    ) -> [torch.distributions.Distribution, torch.Tensor, torch.Tensor]:
        """
        Infers CPDs for a given node given evidence for its parents.

        Args:
            node_name (str): The name of the node.
            evidence (dict): Evidence for the parents as {parent_name: value}.
            uncertainty (float): Uncertainty interval for matching evidence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and covariance of the node's CPDs.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist.")

        translated_evidence = self.translate_evidence(evidence, uncertainty)

        # Pass the translated evidence to the node's CPD computation. It returns node_cpds and target_values
        node_cpd, target_values = self.nodes[node_name].get_cpd(
            translated_evidence, uncertainty
        )
        # Compute pdf
        node_pdf = node_cpd.log_prob(target_values)

        return node_cpd, node_pdf, target_values

    def infer(self, node_name: str, parents_evidence: Dict, uncertainty: float = 0.1):
        translated_evidence = self.translate_evidence(parents_evidence, uncertainty)

        return self.inference.infer(node_name, translated_evidence, uncertainty)

    def translate_evidence(self, parents_evidence: Dict, uncertainty: float = None):
        # Convert evidence keys from column names to indices if needed
        return {
            self.column_mapping.get(key, key): (
                (
                    value
                    if isinstance(value, torch.Tensor)
                    else torch.tensor([value], device=self.device)
                ),
                (
                    (
                        uncertainty
                        if isinstance(uncertainty, torch.Tensor)
                        else torch.tensor([uncertainty], device=self.device)
                    )
                    if uncertainty
                    else min_tolerance
                ),
            )
            for key, value in parents_evidence.items()
        }

    def print_bn_structure(self):
        """Prints the structure of the Bayesian Network."""
        for node in self.dag.nodes:
            parents = list(self.dag.predecessors(node))
            print(f"Node: {node}, Parents: {parents}")

    def get_bn_structure(self):
        """Returns the structure of the Bayesian Network."""
        return self.dag

    def get_nodes(self):
        return self.nodes

    def get_domain(self, node_name: str):
        return self.nodes[node_name].get_domain()

    def get_prior(self, node_name: str):
        return self.nodes[node_name].get_prior()

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
    bn.get_bn_structure()

    target_node = "reward"
    # Infer CPDs for node C given evidence for A and B
    evidence = {
        "reward": torch.tensor([1.0], device="cuda"),
        # "action": torch.tensor([2], device="cuda"),
        # "obs_0": torch.tensor([14.0], device="cuda"),
    }
    cpd, pdf, target_values = bn.get_cpd_and_pdf(target_node, evidence)
    print("Conditional Distribution:", cpd)
    print("PDF: ", pdf)
    print("Target Values: ", target_values)
    bn.plot_cpds(cpd, target_values)

    """p_action, _, _ = bn.get_cpd_and_pdf("action", {})
    p_obs, _, _ = bn.get_cpd_and_pdf("obs_0", {})
    p_reward, pdf, target_values = bn.get_cpd_and_pdf(target_node, evidence)

    distr_to_multiply = [p_action, p_obs, p_reward]
    for d in distr_to_multiply:
        print(d)
    final = multiply_distributions(distr_to_multiply)

    print(final)"""
