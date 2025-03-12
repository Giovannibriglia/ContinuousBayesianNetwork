from typing import Dict

import torch
from tqdm import tqdm

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
        max_n_queries_at_time: int = 1024,
    ) -> torch.Tensor:
        """
        Returns filtered data for `target_node_index` based on `evidence` and `uncertainty`,
        but does so in chunks to reduce memory usage.

        :param data: 3D tensor [n_queries, n_features_tot, n_samples]
        :param target_node_index: which feature to return
        :param evidence: 3D tensor [n_queries, n_features_ev, 1]
        :param uncertainty: margin around each evidence value
        :param max_n_queries_at_time: number of queries to handle per chunk
        :return: [n_queries, min_len] of valid samples
        """
        n_queries, n_features_tot, n_samples = data.shape

        # If there's no evidence, nothing to filter; just return the entire row
        if evidence is None:
            return data[:, target_node_index, :]

        # Precompute upper/lower bounds
        lower_bound = evidence - uncertainty
        upper_bound = evidence + uncertainty

        # -------------------------- Pass 1: find valid_counts for each query in chunks
        valid_counts_list = []

        bar = (
            tqdm(range(0, n_queries, max_n_queries_at_time), desc="chunking data...")
            if max_n_queries_at_time <= n_queries
            else range(0, n_queries, max_n_queries_at_time)
        )

        for start in bar:
            end = min(start + max_n_queries_at_time, n_queries)

            # Extract chunk
            chunk_evidence_features = data[
                start:end, : evidence.shape[1], :
            ]  # shape [chunk_size, n_features_ev, n_samples]
            chunk_lower = lower_bound[start:end]  # shape [chunk_size, n_features_ev, 1]
            chunk_upper = upper_bound[start:end]  # shape [chunk_size, n_features_ev, 1]

            # Create the boolean mask for this chunk: [chunk_size, n_features_ev, n_samples]
            c_mask = (chunk_evidence_features >= chunk_lower) & (
                chunk_evidence_features <= chunk_upper
            )

            # Now reduce across the evidence dimension to find samples that satisfy *all* evidence features
            c_mask = c_mask.all(dim=1)  # shape [chunk_size, n_samples]

            # Count how many valid samples each query has
            c_valid_counts = c_mask.sum(dim=1)  # shape [chunk_size]

            valid_counts_list.append(c_valid_counts)

        # Concatenate counts for all queries and find the global minimum
        valid_counts = torch.cat(valid_counts_list, dim=0)  # shape [n_queries]
        min_len = valid_counts.min().item()

        # If the global min_len = 0, then at least one query has 0 valid samples
        if min_len == 0:
            return torch.empty(n_queries, 0, device=self.device)

        # -------------------------- Pass 2: gather exactly `min_len` valid samples per query
        outputs = []
        for start in range(0, n_queries, max_n_queries_at_time):
            end = min(start + max_n_queries_at_time, n_queries)

            # Extract chunk
            chunk_data = data[
                start:end
            ]  # shape [chunk_size, n_features_tot, n_samples]
            chunk_evidence = evidence[start:end]

            c_lower = chunk_evidence - uncertainty
            c_upper = chunk_evidence + uncertainty

            # Evidence features for this chunk
            c_ev_features = chunk_data[
                :, : evidence.shape[1], :
            ]  # [chunk_size, n_features_ev, n_samples]
            # Target values for this chunk
            c_target_values = chunk_data[
                :, target_node_index, :
            ]  # [chunk_size, n_samples]

            # Build boolean mask again
            c_mask = (c_ev_features >= c_lower) & (c_ev_features <= c_upper)
            c_mask = c_mask.all(dim=1)  # shape [chunk_size, n_samples]

            # If you *only* need the first `min_len` valid samples (and don't care about ordering),
            # you could do something simpler with `nonzero`; but if you want to be consistent
            # with your code, you can still sort or do topk.  For example:
            c_mask_int = c_mask.int()
            sorted_indices = torch.argsort(c_mask_int, descending=True, dim=1)

            # Gather top `min_len` samples (the ones marked True will be first)
            selected_data = torch.gather(
                c_target_values, dim=1, index=sorted_indices[:, :min_len]
            )

            outputs.append(selected_data)

        # Concatenate results for all queries back together
        return torch.cat(outputs, dim=0)
