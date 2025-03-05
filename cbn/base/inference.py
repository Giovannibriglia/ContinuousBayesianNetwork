from abc import ABC, abstractmethod
from typing import Dict

from cbn.base import initial_uncertainty


class BaseInference(ABC):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        self.bn = bn
        self.device = device

        self.dag = self.bn.dag
        self.nodes = self.dag.nodes

    @abstractmethod
    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
    ):
        raise NotImplementedError
