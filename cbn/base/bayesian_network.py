from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd
import torch

from cbn.base import initial_uncertainty
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
        pass

    def _get_domain(self, node: int):
        return self.data[node].unique(sorted=True).to(self.device)

    def get_cpd_and_pdf(
        self,
        target_node: str,
        evidence: Dict,
        uncertainty: float = initial_uncertainty,
        get_pdf: bool = False,
    ):
        """

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

        true_target_node_index = self.column_mapping[target_node]
        target_values = self._get_domain(true_target_node_index)
        target_values = target_values.expand(batch_size, -1)

        if get_pdf:
            pdf = cpd.log_prob(target_values)
            return cpd, pdf, target_values
        else:
            return cpd, None, target_values

    def infer(
        self, target_node: str, evidence: Dict, uncertainty: float = initial_uncertainty
    ):
        pass
