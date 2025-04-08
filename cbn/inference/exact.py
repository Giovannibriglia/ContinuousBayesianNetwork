from typing import Dict

from cbn.base.inference import BaseInference


class ExactInference(BaseInference):
    def __init__(self, config: Dict, **kwargs):
        super(ExactInference).__init__(config=config, **kwargs)

    def _setup_model(self, config: Dict, **kwargs):
        pass

    def _infer(self, target_node: str, evidence: Dict, do: Dict, **kwargs):
        pass
        # [n_queries, [n_samples]*n_parents, n_samples_target]
        # [n_queries, n_samples_node]
        # [n_queries, (n_parents)*n_samples_for_parent]
