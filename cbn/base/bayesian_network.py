from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from cbn.base import BASE_MAX_CARDINALITY, KEY_MAX_CARDINALITY_FOR_DISCRETE
from cbn.base.node import Node
from cbn.utils import choose_inference_obj


class BayesianNetwork:
    def __init__(
        self,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        parameters_learning_config: Dict,
        inference_config: Dict,
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

        self.log = kwargs.get("log", False)

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

        pbar = (
            tqdm(
                self.initial_dag.nodes,
                total=len(self.initial_dag.nodes),
                desc="training probability estimator...",
            )
            if self.log
            else self.initial_dag.nodes
        )
        self._train(data, pbar)

    def _setup_inference(self, config: Dict):
        self.inference_obj_name = config["inference_obj"]
        self.inference_obj = choose_inference_obj(self.inference_obj_name, config)

    def save_model(self, path: str):
        for node in self.nodes_obj:
            self.nodes_obj[node].save_model(path)

    @staticmethod
    def get_nodes(dag: nx.DiGraph):
        return sorted(list(dag.nodes))

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
            return sorted(list(dag.predecessors(node)))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return sorted((list(dag.predecessors(node_name))))
        else:
            raise ValueError(f"{node} type not supported.")

    def get_children(self, dag: nx.DiGraph, node: int | str):
        if isinstance(node, str):
            return sorted(list(dag.successors(node)))
        elif isinstance(node, int):
            node_name = next(
                (k for k, v in self.column_mapping.items() if v == node), None
            )
            return sorted(list(dag.successors(node_name)))
        else:
            raise ValueError(f"{node} type not supported.")

    def update_knowledge(self, data: pd.DataFrame):
        pbar = (
            tqdm(
                self.initial_dag.nodes,
                total=len(self.initial_dag.nodes),
                desc="updating probability estimator...",
            )
            if self.log
            else self.initial_dag.nodes
        )
        self._train(data, pbar)

    def _train(self, data: pd.DataFrame, pbar: Iterable | tqdm):
        is_tqdm = isinstance(pbar, tqdm)
        for node in pbar:
            if is_tqdm:
                pbar.set_postfix(updating_node=f"{node}")

            node_data = torch.tensor(
                np.array(data[node].values.tolist(), dtype=np.float32),
                device=self.device,
            )

            node_parents = self.get_parents(self.initial_dag, node)
            parents_data = (
                torch.tensor(
                    np.array(data[node_parents].values.tolist(), dtype=np.float32),
                    device=self.device,
                ).T
                if node_parents
                else None
            )
            self.nodes_obj[node].fit(node_data, parents_data)
            if is_tqdm:
                pbar.set_postfix(desc="done!")

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # {feature} is not parent of {target_node}, for this reason will not be considered in the probability computation.

        pdfs, target_node_domains, parents_domains = self.nodes_obj[
            target_node
        ].get_prob(query, N_max)

        # [n_queries, [n_samples]*n_parents, n_samples_target]
        # [n_queries, n_samples_node]
        # [n_queries, (n_parents)*n_samples_for_parent]

        return pdfs, target_node_domains, parents_domains

    def infer(
        self,
        target_node: str,
        evidence: Dict[str, torch.Tensor] = None,
        do: List[str] = None,
        N_max: int = 16,
        plot_prob=False,
    ):
        """

        :param plot_prob:
        :param target_node: The node (variable) whose marginal probability is to be computed.
        :param evidence: Dictionary where each node is a node name and each value is a torch tensor
                         of shape [n_queries, 1] that represents the observed value(s) for that node.
        :param do: List of str representing variables on which to intervene (not handled in this version).
        :param N_max: Maximum number of samples (i.e. possible discrete values per node)
        :return: A torch tensor of shape [n_queries, n_target_values] corresponding to the normalized
                 marginal probability of the target node for each query.
        """

        dag = self.initial_dag
        if do is not None:
            # TODO:
            # Future extension: modify the graph/densities for interventions.
            dag = self.initial_dag

        # Get the list of nodes that can affect the target
        ancestors_target_node = self.get_ancestors(dag, target_node)
        ancestors_target_node.append(target_node)  # include target itself

        # Build a list of factors; each factor is a tuple (scope, tensor)
        # where scope is a list of variables (strings) and tensor is a torch tensor
        # containing the conditional probability for that node given its parents.
        factors = {}
        target_node_domains = None
        for node in ancestors_target_node:
            # get_pdf returns three things:
            # - pdfs: tensor of shape [n_queries, *parent_dims, node_domain_size]
            # - node_domains: tensor of possible values for node (shape [n_queries, n_node_values])
            # - parents_domains: tensor of possible values for each parent (shape [n_queries, n_parent_values])
            pdfs, node_domains, parents_domains = self.get_pdf(node, evidence, N_max)

            """print(
                f"{node}: pdfs: {pdfs.shape}, node: {node_domains.shape}, parents: {parents_domains.shape if parents_domains is not None else None}"
            )"""
            factors[node] = pdfs
            if node == target_node:
                target_node_domains = node_domains

        if evidence is None:
            n_queries = 1
        else:
            if len(evidence.keys()) > 0:
                n_queries = evidence[next(iter(evidence.keys()))].shape[0]
            else:
                n_queries = 1

        n_samples_node = (
            self.nodes_obj[target_node].sample_domain(target_node, N_max).shape[0]
        )

        out_pdf = torch.ones((n_queries, n_samples_node), device=self.device)

        for node, pdf in factors.items():
            if pdf.dim() > 2:
                n_parents = pdf.dim() - 2  # it has parents
                if node == target_node:  # it is target
                    dims = list(range(1, n_parents + 1))
                    # print("0: ", dims)
                else:  # not target
                    # TODO: check
                    print("CHECK IF IT IS CORRECT")
                    dims = list(range(1, n_parents + 1))
                    # print("1:", dims)
            else:  # no parents
                dims = [1]
                # print("2: ", dims)

            dims = [int(d) for d in dims]  # make sure they're integers
            if pdf.dim() > 64:
                raise ValueError(
                    f"We can handle node with maximum 64 parents during inference; instead you have {pdf.dim()} parents"
                )
            else:
                x = torch.mean(pdf.to(torch.float32), dim=dims)
            out_pdf *= x
            # print(out_pdf.shape)

        out_pdf = out_pdf / out_pdf.max()

        if plot_prob:
            self.plot_prob(out_pdf, target_node_domains, target_node)

        assert (
            out_pdf.shape == target_node_domains.shape
        ), "pdf and domain must have same shape."

        return out_pdf, target_node_domains

    @staticmethod
    def plot_prob(pdf, domain, target_node: str):

        assert pdf.shape == domain.shape, "pdf and domain must have same shape."

        import matplotlib.pyplot as plt

        n_queries, n_samples_node = pdf.shape

        pdf_np = pdf.cpu().numpy()
        domain_np = domain.cpu().numpy()

        plt.figure(dpi=500)
        for q in range(n_queries):
            plt.plot(domain_np[q], pdf_np[q], label=f"query {q}")
        plt.xlabel(f"{target_node} domain")
        plt.ylabel("PDF")
        plt.xticks(domain_np[0])
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def benchmarking_df(
        self, data: pd.DataFrame, target_feature: str, batch_size: int = 128, **kwargs
    ) -> np.ndarray:

        dict_values = {
            feat: torch.tensor(data[feat].values, device="cpu")
            for feat in data.columns
            if feat != target_feature
        }

        pred_values = np.zeros((len(data),))

        progress_bar = tqdm(total=len(data), desc="benchmarking df cbn...")
        for n in range(0, len(data), batch_size):
            evidence = {
                feat: dict_values[feat][n : n + batch_size].unsqueeze(-1).to("cuda")
                for feat in data.columns
                if feat != target_feature
            }

            inference_probabilities, domain_values = self.infer(
                target_feature,
                evidence,
                plot_prob=False,
                N_max=16,
            )

            # Get indices of max probabilities along dimension 1 (columns)
            max_probabilities_indices = torch.argmax(
                inference_probabilities, dim=1, keepdim=True
            )  # Shape [batch_size, 1]
            # Gather the corresponding domain values
            pred_values_batch = (
                torch.gather(domain_values, dim=1, index=max_probabilities_indices)
                .squeeze(1)
                .cpu()
                .numpy()
            )  # Shape [batch_size]

            pred_values[n : n + batch_size] = pred_values_batch

            batch_end = min(n + batch_size, len(data))
            progress_bar.update(batch_end - n)  # Update by actual batch size

        return pred_values
