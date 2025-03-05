from typing import Dict

import torch

from cbn.base import initial_uncertainty
from cbn.base.inference import BaseInference


class ExactInference(BaseInference):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        super().__init__(bn, device, **kwargs)

        # technique = kwargs.get("kind", "perfect")

    def infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
    ):
        if len(evidence.keys()) > 0:
            first_key = list(evidence.keys())[0]

            batch_size = evidence[first_key].shape[0]
        else:
            batch_size = 1

        n_values = self.bn.get_domain(target_node).shape[0]

        """# Modify Bayesian Network structure if do-interventions exist
        if do:
            self.bn.intervene(do)  # Assuming self.bn.intervene removes incoming edges
        """

        ancestors = self.bn.get_ancestors(target_node)
        ancestors.append(target_node)
        # bn_structure = self.bn.get_structure()

        quantities = []

        for ancestor in ancestors:
            ancestor_parents = self.bn.get_parents(ancestor)

            points_to_evaluate = None
            if len(ancestor_parents) >= 1:
                extracted_evidence = {
                    key: values
                    for key, values in evidence.items()
                    if key in ancestor_parents
                }

                if ancestor is evidence.keys():
                    print("1")
                    extracted_evidence[ancestor] = evidence[ancestor]
                else:
                    print("2")
                    points_to_evaluate = self.bn.get_domain(ancestor)
                    # extracted_evidence[ancestor] = self.bn.get_domain(ancestor)
            else:
                if ancestor is evidence.keys():
                    print("3")
                    extracted_evidence = evidence[ancestor]
                else:
                    print("4")
                    extracted_evidence = {ancestor: self.bn.get_domain(ancestor)}

            cpd, pdf, _ = self.bn.get_cpd_and_pdf(
                ancestor, extracted_evidence, points_to_evaluate=points_to_evaluate
            )

            if ancestor == target_node:
                pdf_summed = pdf
            else:
                pdf_summed = torch.sum(pdf) / (pdf.shape[1] * pdf.shape[0])

            quantities.append(pdf_summed)

        # Perform element-wise multiplication
        result = torch.ones((batch_size, n_values), device="cuda:0")
        for t in quantities:
            result = result * t  # Broadcasting will handle multiplication

        assert result.shape == (
            batch_size,
            n_values,
        ), f"result shape is incorrect: {result.shape}. It should be [{batch_size}, {n_values}]"
        return result
