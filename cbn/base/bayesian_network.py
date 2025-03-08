from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd
import torch

from cbn.base import initial_uncertainty, min_tolerance
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

    def get_nodes(self):
        return self.nodes

    def get_dag(self):
        return self.dag

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
        normalize_pdf: bool = True,
        points_to_evaluate: torch.Tensor = None,
    ):
        """

        :param points_to_evaluate:
        :param normalize_pdf:
        :param target_node:
        :param evidence: {"feat": torch.Tensor with shape [batch_size]
        :param uncertainty: float
        :return:
        """

        if len(evidence.keys()) > 0:
            evidence_features = list(evidence.keys())
            data_features = [
                self.column_mapping[feature] for feature in evidence_features
            ]

            n_queries = evidence[evidence_features[0]].shape[0]
            n_combinations = evidence[evidence_features[0]].shape[1]

            if target_node not in evidence_features:
                data_features.append(self.column_mapping[target_node])
                target_node_index = len(data_features) - 1
            else:
                target_node_index = evidence_features.index(target_node)

            evidence_tensor = torch.zeros(
                (n_queries, len(evidence_features), n_combinations)
            )
            for query_n in range(n_queries):
                for feat_ev_idx, feat_ev in enumerate(evidence_features):
                    if feat_ev in evidence.keys():
                        evidence_tensor[query_n, feat_ev_idx, :] = evidence[feat_ev][
                            query_n
                        ]

            filtered_data = (
                self.data[data_features].unsqueeze(0).expand(n_queries, -1, -1)
            )
        else:
            target_node_index = 0
            n_queries = (
                points_to_evaluate.shape[0] if points_to_evaluate is not None else 1
            )
            evidence_tensor = None
            filtered_data = self.data[self.column_mapping[target_node]].unsqueeze(0)

        cpd = self.parameters_learning.get_cpd(
            target_node_index, evidence_tensor, filtered_data, uncertainty
        )

        node_domain = self.get_domain(target_node).unsqueeze(0).expand(n_queries, -1)
        pdf = cpd.log_prob(node_domain)

        if normalize_pdf:
            pdf_normalized = self.safe_normalize_pdf(pdf)

            # Assert that each slice in the last dimension sums to 1
            assert torch.allclose(
                pdf_normalized.sum(dim=-1),
                torch.ones_like(pdf_normalized.sum(dim=-1)),
                atol=min_tolerance,
            ), "Normalization failed: Sums are not all 1."

            if points_to_evaluate is None:
                if target_node in evidence.keys():
                    values_to_evaluate, _ = evidence[
                        target_node
                    ].sort()  # [n_queries, n_values]
                    print("1")
                else:
                    values_to_evaluate = node_domain.expand(
                        n_queries, -1
                    )  # [n_queries, n_values]
                    print("2")
            else:
                values_to_evaluate, _ = (
                    points_to_evaluate.sort()
                )  # [n_queries, n_values]
                print("3")

            if not torch.equal(node_domain, values_to_evaluate):
                # node_domain shape [n_queries, n_tot_values]
                # values_to_evaluate [n_queries, n_values]

                # Find indices where node_domain == values_to_evaluate
                indices = torch.zeros_like(values_to_evaluate, dtype=torch.long)

                for i in range(n_queries):
                    indices[i] = torch.where(
                        node_domain[i].unsqueeze(0)
                        == values_to_evaluate[i].unsqueeze(1)
                    )[1]

                # Gather the corresponding pdf values
                pdf_normalized = torch.gather(
                    pdf_normalized, 1, indices
                )  # [n_queries, n_values]

            assert (
                pdf_normalized.dim() == 2
                and pdf_normalized.shape == values_to_evaluate.shape
            ), ValueError(
                f"pdf of {target_node} has shape: {pdf_normalized.shape}, it should be: {values_to_evaluate.shape}"
            )
            return cpd, pdf_normalized, values_to_evaluate
        else:
            return cpd, pdf, node_domain

    @staticmethod
    def safe_normalize_pdf(
        pdf: torch.Tensor, epsilon: float = min_tolerance
    ) -> torch.Tensor:
        """
        Safely normalize a PDF along the `n_values` dimension, handling large or negative values.

        Args:
            pdf (torch.Tensor): Input tensor of shape [n_queries, n_values].
            epsilon (float): Small value to prevent division by zero.

        Returns:
            torch.Tensor: Normalized PDF with the sum of each row equal to 1.
        """
        # Shift values to avoid extremely large negatives affecting normalization
        pdf_max = pdf.max(dim=1, keepdim=True).values
        pdf_shifted = pdf - pdf_max  # Ensure numerical stability

        # Convert to positive values (exponentiate)
        pdf_exp = torch.exp(pdf_shifted)

        # Normalize
        return pdf_exp / (pdf_exp.sum(dim=1, keepdim=True) + epsilon)

    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
    ):
        return self.inference.infer(target_node, evidence, do, uncertainty)
