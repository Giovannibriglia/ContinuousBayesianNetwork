from typing import Dict

import torch

from cbn.base.parameters_estimator import BaseParameterEstimator


class MaximumLikelihoodEstimator(BaseParameterEstimator):
    """Parametric estimator for specific distributions."""

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device)

    def return_data(
        self,
        node_data: torch.Tensor,
        parents_data: torch.Tensor = None,
        query_node: torch.Tensor = None,
        query_parents: Dict = None,
    ) -> torch.Tensor:

        if query_parents is None or parents_data is None:
            # Case 1: No parent conditions, filter only on node_data
            if query_node is None:
                return node_data  # No filtering needed

            conditions = torch.stack(
                [
                    (node_data >= mean - unc) & (node_data <= mean + unc)
                    for mean, unc in query_node
                ]
            )

        else:
            # Case 2: Filter based on parents or both parents and nodes
            if query_node is None:
                conditions = [
                    (parents_data[key, :] >= mean - unc)
                    & (parents_data[key, :] <= mean + unc)
                    for key, (mean, unc) in query_parents.items()
                ]
            else:
                conditions = [
                    (
                        (parents_data[key_parent, :] >= mean_parent - unc_parent)
                        & (parents_data[key_parent, :] <= mean_parent + unc_parent)
                    )
                    & (
                        (node_data[i, :] >= mean_node - unc_node)
                        & (node_data[i, :] <= mean_node + unc_node)
                    )
                    for i, (
                        (mean_node, unc_node),
                        key_parent,
                        (mean_parent, unc_parent),
                    ) in enumerate(zip(query_node, query_parents.items()))
                ]

        # Combine all conditions element-wise
        combined_condition = torch.stack(conditions, dim=0).all(dim=0)

        # Find valid indices
        selected_indices = combined_condition.nonzero(as_tuple=True)[0]

        if selected_indices.numel() == 0:
            raise ValueError(
                "No matches found for the node data in the maximum likelihood estimator."
            )

        # Return filtered node data
        return node_data[:, selected_indices]
