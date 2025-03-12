from abc import ABC, abstractmethod
from typing import Dict

import torch

from cbn.base import initial_uncertainty


class BaseInference(ABC):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        self.bn = bn
        self.device = device

    @abstractmethod
    def _infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
        plot_prob: bool = False,
        N_max: int = None,
    ):
        raise NotImplementedError

    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
        plot_prob: bool = False,
        N_max: int = None,
    ):
        """

        :param plot_prob:
        :param target_node: str
        :param evidence: Dict, for each feature there is torch.Tensor with shape [n_queries, 1]
        :param do:
        :param uncertainty:
        :return:
        """

        if evidence:
            first_key = next(iter(evidence))
            n_queries = evidence[first_key].shape[0]
        else:
            n_queries = 1

        if target_node in evidence:
            points_to_evaluate = evidence[target_node]
        else:
            points_to_evaluate = (
                self.bn.get_domain(target_node).unsqueeze(0).expand(n_queries, -1)
            )
        n_points_to_evaluate = min(N_max, points_to_evaluate.shape[1])

        output, points = self._infer(
            target_node, evidence, do, uncertainty, plot_prob, N_max
        )
        self._check_output(output, points, n_queries, n_points_to_evaluate)

        return output, points

    @staticmethod
    def _check_output(
        prob: torch.Tensor,
        points: torch.Tensor,
        n_queries: int,
        n_points_to_evaluate: int,
    ):
        assert prob.shape == points.shape, ValueError(
            f"prob and points must have the same shape. Now they have {prob.shape} - {points.shape}"
        )

        assert (
            prob.shape == points.shape == torch.Size([n_queries, n_points_to_evaluate])
        ), ValueError(
            f"prob and points must have the shape of {n_queries, n_points_to_evaluate}. Now they have {points.shape} -{points.shape}"
        )
