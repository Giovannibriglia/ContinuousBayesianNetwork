from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd
import torch

from cbn.base import initial_uncertainty
from cbn.inference.exact import ExactInference
from cbn.parameters_learning.mle import MaximumLikelihoodEstimator


class BayesianNetwork:
    def __init__(
        self,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        parameters_learning_config: Dict = None,
        inference_config: Dict = None,
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
        self.nodes = list(dag.nodes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data, self.column_mapping = self._data_to_tensors(data)

        self.parameters_learning = self._setup_parameters_learning(
            parameters_learning_config
        )

        self.inference = self._setup_inference(inference_config)

    @staticmethod
    def _data_to_tensors(data, device_data_storage: str = "cpu"):
        """Converts data to a PyTorch tensor and creates a node-to-column mapping."""
        if isinstance(data, pd.DataFrame):
            tensor_data = torch.tensor(
                data.values.T, dtype=torch.float32, device=device_data_storage
            )
            columns_index_map = {
                col_name: idx for idx, col_name in enumerate(data.columns)
            }
        else:
            raise ValueError(
                "Data must be a pd.DataFrame, np.ndarray, or torch.Tensor."
            )

        return tensor_data, columns_index_map

    def _setup_parameters_learning(self, config: Dict):
        if config is None:
            return MaximumLikelihoodEstimator({}, device=self.device)
        else:
            estimator_name = config["name"]
            probability_estimator = config["probability_estimator"]
            kwargs = config["kwargs"]

            if estimator_name == "mle":
                return MaximumLikelihoodEstimator(
                    probability_estimator, self.device, **kwargs
                )
            elif estimator_name == "bayesian_estimator":
                raise NotImplementedError
            else:
                raise ValueError(f"estimator type {estimator_name} is not defined")

    def _setup_inference(self, config: Dict):
        if config is None:
            return ExactInference(self, device=self.device)
        else:
            # TODO
            return ExactInference(self, device=self.device)

    def get_domain(self, node: int | str):
        if isinstance(node, int):
            return self.data[node].unique(sorted=True).to(self.device)
        elif isinstance(node, str):
            return (
                self.data[self.column_mapping[node]].unique(sorted=True).to(self.device)
            )
        else:
            raise ValueError(f"{node} type not supported.")

    def get_ancestors(self, node: int | str):
        if isinstance(node, str):
            ancestors = nx.ancestors(self.dag, node)
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            if node_name is None:
                return set()
            ancestors = nx.ancestors(self.dag, node_name)
        else:
            raise ValueError(f"{node} type not supported.")

        # Sort ancestors from farthest to closest using topological sorting
        sorted_ancestors = list(
            nx.topological_sort(self.dag.subgraph(ancestors | {node}))
        )
        sorted_ancestors.remove(node)  # Remove the input node itself
        return sorted_ancestors

    def get_parents(self, node: int | str):
        if isinstance(node, str):
            return list(self.dag.predecessors(node))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return list(self.dag.predecessors(node_name))
        else:
            raise ValueError(f"{node} type not supported.")

    def get_children(self, node: int | str):
        if isinstance(node, str):
            return list(self.dag.successors(node))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return list(self.dag.successors(node_name))
        else:
            raise ValueError(f"{node} type not supported.")

    def get_structure(self):
        structure = {}

        # Get topological order to ensure parents appear before children
        topological_order = list(nx.topological_sort(self.dag))

        for node in topological_order:
            # Get direct parents
            parents = list(self.dag.predecessors(node))
            structure[node] = parents  # Store in dict

        return structure

    def get_cpd_and_pdf(
        self,
        target_node: str,
        evidence: Dict,
        uncertainty: float = initial_uncertainty,
        get_pdf: bool = True,
        normalize_pdf: bool = True,
        points_to_evaluate: torch.Tensor = None,
    ):
        """

        :param get_pdf:
        :param target_node:
        :param evidence: {"feat": torch.Tensor with shape [batch_size]
        :param uncertainty: float
        :return:
        """

        if len(evidence.keys()) > 0:
            first_key = list(evidence.keys())[0]

            batch_size = evidence[first_key].shape[0]
            num_evidence_features = len(evidence.keys())

            evidence_tensor = torch.zeros(
                (batch_size, num_evidence_features), device=evidence[first_key].device
            )
            filtered_data = torch.zeros(
                (num_evidence_features, self.data.shape[1]), device=self.data.device
            )

            count = 0
            target_node_index = None
            for feature_name, feature_values in evidence.items():
                evidence_tensor[:, count] = feature_values
                feature_index = self.column_mapping[feature_name]
                filtered_data[count, :] = self.data[feature_index]

                if feature_name == target_node:
                    target_node_index = count

                count += 1

            if target_node_index is None:
                target_node_index = self.column_mapping[target_node]
                target_node_values = self.data[target_node_index].unsqueeze(0)
                filtered_data = torch.cat([filtered_data, target_node_values], dim=0)
                target_node_index = filtered_data.shape[0] - 1
        else:
            target_node_index = 0
            batch_size = 1
            evidence_tensor = None
            filtered_data = self.data[self.column_mapping[target_node]].unsqueeze(0)

        cpd = self.parameters_learning.get_cpd(
            target_node_index, evidence_tensor, filtered_data, uncertainty
        )

        if get_pdf:
            if points_to_evaluate is None:
                if target_node in evidence.keys():
                    values_to_evaluate = evidence[target_node]
                else:
                    values_to_evaluate = self.get_domain(target_node)
            else:
                values_to_evaluate = points_to_evaluate

            # Compute log probabilities for each batch element
            pdf = torch.stack(
                [cpd[b].log_prob(values_to_evaluate) for b in range(batch_size)]
            )
            if normalize_pdf:
                if pdf.shape[1] == 1:
                    return cpd, torch.ones_like(pdf), values_to_evaluate

                # Normalize log probs in each batch separately
                min_vals = pdf.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
                max_vals = pdf.max(dim=1, keepdim=True)[0]  # (batch_size, 1)

                # Avoid division by zero: If min == max, set normalization to zero
                denom = max_vals - min_vals
                denom[denom == 0] = 1  # Prevent division by zero

                normalized_pdf = (
                    pdf - min_vals
                ) / denom  # (batch_size, n_values), now in [0, 1]

                return cpd, normalized_pdf, values_to_evaluate
            else:
                return cpd, pdf, values_to_evaluate

        else:
            return cpd, None, None

    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
    ):
        return self.inference.infer(target_node, evidence, do, uncertainty)
