from typing import Dict

import torch

from cbn.base.parameters_estimator import BaseParameterLearning


class MaximumLikelihoodEstimator(BaseParameterLearning):
    def __init__(
        self, config_probability_estimator: Dict = None, device: str = "cpu", **kwargs
    ):
        super().__init__(config_probability_estimator, device, **kwargs)

    def _check_selected_data(self, data: torch.Tensor, selected_data: torch.Tensor):
        # TODO
        pass

    def _return_data(
        self,
        data: torch.Tensor,
        target_node_index: int,
        evidence: torch.Tensor = None,
        uncertainty: float = 0.0,
    ) -> torch.Tensor:
        """
        Filters the `data` tensor based on the `evidence` provided, returning
        only those samples in the target feature dimension that satisfy
        the evidence constraints. The output contains exactly `min_len` samples
        per query to avoid missing values.

        :param data: 3D tensor of shape [n_queries, n_features_tot, n_samples]
        :param target_node_index: int - index of the target feature in the second dimension of `data`
        :param evidence: 3D tensor of shape [n_queries, n_features_ev, 1],
                         may not include all features of target_node_index
        :param uncertainty: float - symmetric margin around each evidence value
        :return: extracted data of the target_node_index. Shape: [n_queries, n_min_selected_samples]
        """
        device = data.device
        n_queries, n_features_tot, n_samples = data.shape

        if evidence is None:
            return data[:, target_node_index, :]  # [n_queries, n_samples]

        # Create lower & upper bounds for the evidence constraints
        lower_bound = evidence - uncertainty
        upper_bound = evidence + uncertainty

        # Extract the relevant evidence features from data
        evidence_features = data[:, : evidence.shape[1], :]

        # Create a mask for values within the uncertainty bounds
        mask = (evidence_features >= lower_bound) & (evidence_features <= upper_bound)
        mask = mask.all(
            dim=1
        )  # Reduce across evidence features to ensure all conditions hold

        # Extract the target feature values
        target_values = data[:, target_node_index, :]

        # Count valid samples per query
        valid_counts = mask.sum(dim=1)
        min_len = valid_counts.min().item()

        if min_len == 0:
            return torch.empty(n_queries, 0, device=device)

        # Select first `min_len` valid samples for each query
        sorted_indices = torch.argsort(mask.int(), descending=True, dim=1)
        selected_data = torch.gather(
            target_values, dim=1, index=sorted_indices[:, :min_len]
        )

        return selected_data
