from abc import ABC, abstractmethod
from typing import Dict

from cbn.base import initial_uncertainty


class BaseInference(ABC):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        self.bn = bn
        self.device = device

    @abstractmethod
    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
    ):
        """

        :param target_node: str
        :param evidence: Dict, for each feature there is torch.Tensor with shape [n_queries, 1]
        :param do:
        :param uncertainty:
        :return:
        """
        raise NotImplementedError
