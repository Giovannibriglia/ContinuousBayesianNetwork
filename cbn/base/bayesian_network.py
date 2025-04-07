from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch

from tqdm import tqdm

from cbn.base import BASE_MAX_CARDINALITY, KEY_MAX_CARDINALITY_FOR_DISCRETE
from cbn.base.node import Node


class BayesianNetwork:
    def __init__(
        self,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        parameters_learning_config: Dict = None,
        inference_config: Dict = None,
        **kwargs,
    ):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(
                "The provided graph is not a directed acyclic graph (DAG)."
            )

        self.initial_dag = dag
        self.column_mapping = {node: i for i, node in enumerate(self.initial_dag.nodes)}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["device"] = self.device if "device" not in kwargs else kwargs["device"]
        self.min_tolerance = kwargs.get("min_tolerance", 1e-10)
        self.uncertainty = kwargs.get("uncertainty", 1e-10)
        self.max_cardinality_for_discrete_domain = kwargs.get(
            KEY_MAX_CARDINALITY_FOR_DISCRETE, BASE_MAX_CARDINALITY
        )

        self.nodes_obj = None

        self._setup_parameters_learning(data, parameters_learning_config, **kwargs)

        self._setup_inference(inference_config)

    def _setup_parameters_learning(self, data: pd.DataFrame, config: Dict, **kwargs):

        estimator_name = config["estimator_name"]
        self.nodes_obj = {
            node: Node(
                node,
                estimator_name,
                config,
                self.get_parents(self.initial_dag, node),
                **kwargs,
            )
            for node in self.initial_dag.nodes
        }

        pbar = tqdm(self.initial_dag.nodes, total=len(self.initial_dag.nodes))
        # TODO: training in parallel
        for node in pbar:
            pbar.set_postfix(training_node=f"{node}")
            node_data = torch.tensor(data[node], device=self.device)

            node_parents = self.get_parents(self.initial_dag, node)
            parents_data = (
                torch.tensor(data[node_parents].values, device=self.device).T
                if node_parents
                else None
            )
            self.nodes_obj[node].fit(node_data, parents_data)
            pbar.set_postfix(desc="training done!")

    def _setup_inference(self, config: Dict):
        # TODO
        pass

    @staticmethod
    def get_nodes(dag: nx.DiGraph):
        return list(dag.nodes)

    def get_ancestors(self, dag: nx.DiGraph, node: int | str):
        if isinstance(node, str):
            ancestors = nx.ancestors(dag, node)
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            if node_name is None:
                return set()
            ancestors = nx.ancestors(dag, node_name)
        else:
            raise ValueError(f"{node} type not supported.")

        # Sort ancestors from farthest to closest using topological sorting
        sorted_ancestors = list(nx.topological_sort(dag.subgraph(ancestors | {node})))
        sorted_ancestors.remove(node)  # Remove the input node itself
        return sorted_ancestors

    def get_parents(self, dag: nx.DiGraph, node: int | str):
        if isinstance(node, str):
            return list(dag.predecessors(node))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return list(dag.predecessors(node_name))
        else:
            raise ValueError(f"{node} type not supported.")

    def get_children(self, dag: nx.DiGraph, node: int | str):
        if isinstance(node, str):
            return list(dag.successors(node))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return list(dag.successors(node_name))
        else:
            raise ValueError(f"{node} type not supported.")

    @staticmethod
    def get_structure(self, dag: nx.DiGraph):
        structure = {}

        # Get topological order to ensure parents appear before children
        topological_order = list(nx.topological_sort(dag))

        for node in topological_order:
            # Get direct parents
            parents = list(dag.predecessors(node))
            structure[node] = parents  # Store in dict

        return structure

    def get_pdf(
        self,
        target_node: str,
        evidence: Dict,
        N_max: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param target_node: str
        :param evidence: [n_queries, n_features, 1]
        :param N_max: int
        :return:
        """

        target_node_parents = self.get_parents(self.initial_dag, target_node)
        query = {}

        for feature, values in evidence.items():
            if feature in target_node_parents:
                query[feature] = values
            else:
                print(
                    f"{feature} is not parent of {target_node}, for this reason will not be considered in the probability computation."
                )
        domains, pdfs = self.nodes_obj[target_node].get_prob(query, N_max)

        return domains, pdfs

    @staticmethod
    def _safe_normalize_pdf(pdf: torch.Tensor, epsilon: float) -> torch.Tensor:
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
        evidence: Dict[str, torch.Tensor],
        do: List[str],
        N_max: int = 1024,
    ):
        """

        :param target_node:
        :param evidence: for each key a torch tensor with shape [n_queries, 1].
        :param do: list of str
        :param N_max:
        :return: [n_queries, n_values]
        """
        raise NotImplementedError
