import torch

from cbn.base.parameters_estimator import BaseParameterEstimator


class MaximumLikelihoodEstimator(BaseParameterEstimator):
    """Parametric estimator for specific distributions."""

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device)

    def return_data(
        self, parents_data: torch.Tensor, node_data: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:

        # Create conditions for filtering data based on parent evidence
        conditions = [
            (parents_data[i, :] >= mean - unc) & (parents_data[i, :] <= mean + unc)
            for i, (mean, unc) in enumerate(query)
        ]

        combined_condition = torch.stack(conditions, dim=0).all(dim=0)
        selected_indices = combined_condition.nonzero(as_tuple=True)[0]

        if selected_indices.numel() == 0:
            raise ValueError(
                "No matches found for Node. Unexpected failure in binary search."
            )

        # Select node data for this node
        return node_data[:, selected_indices]
