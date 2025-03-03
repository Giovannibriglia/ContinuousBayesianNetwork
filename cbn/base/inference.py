from abc import ABC
from typing import Dict


class BaseInference(ABC):
    def __init__(self, bn):
        self.bn = bn

    def infer(self, node_name: str, parents_evidence: Dict, uncertainty: float = 0.1):
        raise NotImplementedError
