from typing import Dict

from cbn.base.inference import BaseInference


class ExactInference(BaseInference):
    def __init__(self, bn, **kwargs):
        super().__init__(bn)

    def infer(self, target_node: str, evidence: Dict, uncertainty: float = 0.1, n=8):
        """
        Returns a dictionary {query_value: P(query_value | evidence)}
        by enumerating all possible assignments of the hidden variables.

        :param bn: A Bayesian Network object with methods:
                     - bn.get_nodes()
                     - bn.get_topological_order()
                     - bn.get_domain(node)
                     - bn.get_cpd_and_pdf(node, evidence, t)
        :param query_node: The variable of interest (string or identifier).
        :param evidence: dict {var: value} specifying observed evidence.
        :return: dict {query_value: probability} (normalized).
        """

        # dag = self.bn.get_bn_structure()
        # nodes = list(dag.nodes())

        # hidden_vars = [v for v in nodes if v not in evidence and v != target_node]
        pass

    def _infer(self, cpd):
        raise NotImplementedError


class VariableElimination(ExactInference):
    def __init__(self, bn, **kwargs):
        super().__init__(bn, **kwargs)

    def _infer(self, cpd):
        pass
