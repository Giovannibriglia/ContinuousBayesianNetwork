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
        Filters the data tensor based on the evidence provided.

        :param data: [batch_size, n_parents_features+1, n_samples]
        :param target_node_index: index of the target feature
        :param evidence: [batch_size, n_evidence_features] (may not include all features)
        :param uncertainty: float
        :return: torch.Tensor [batch_size, n_selected_samples_from_target_node]
        """
        batch_size, n_features, n_samples = data.shape

        if evidence is None:
            return data[
                :, target_node_index, :
            ]  # No filtering, return all target values

        batch_size_evidence, n_evidence_features = evidence.shape
        assert batch_size == batch_size_evidence, "Evidence and data shapes mismatch"

        # Reshape evidence for broadcasting
        evidence_reshaped = evidence.unsqueeze(
            -1
        )  # Shape: [batch_size, n_evidence_features, 1]

        # Compute bounds for filtering
        lower_bound = evidence_reshaped - uncertainty
        upper_bound = evidence_reshaped + uncertainty

        # Apply condition only on evidence features (excluding the target feature)
        mask = (data[:, :n_evidence_features, :] >= lower_bound) & (
            data[:, :n_evidence_features, :] <= upper_bound
        )

        # Reduce along the feature dimension: Keep samples that satisfy **all** conditions
        mask = mask.all(dim=1)  # Shape: [batch_size, n_samples]

        # Extract selected target values based on mask
        selected_data = [data[i, target_node_index, mask[i]] for i in range(batch_size)]

        # Pad selected samples to the maximum length for consistent shape
        max_selected = max(x.shape[0] for x in selected_data) if selected_data else 0
        selected_data_padded = torch.zeros(batch_size, max_selected, device=data.device)

        for i in range(batch_size):
            selected_data_padded[i, : selected_data[i].shape[0]] = selected_data[i]

        return selected_data_padded  # Shape: [batch_size, n_selected_samples]
