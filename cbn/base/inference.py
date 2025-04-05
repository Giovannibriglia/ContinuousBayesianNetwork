from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseInference(ABC):
    def __init__(self, config: Dict, **kwargs):
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.if_log = kwargs.get("log", False)

    @abstractmethod
    def _setup_model(self, config: Dict, **kwargs):
        raise NotImplementedError

    def infer(self, target_node: str, evidence: Dict, do: Dict, **kwargs):

        self._infer()

    @abstractmethod
    def _infer(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        raise NotImplementedError
