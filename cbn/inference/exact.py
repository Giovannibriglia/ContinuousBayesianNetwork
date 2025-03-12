from typing import Dict

import torch

from cbn.base import initial_uncertainty
from cbn.base.inference import BaseInference
from cbn.utils import uniform_sample_tensor


class ExactInference(BaseInference):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        super().__init__(bn, device, **kwargs)

        # technique = kwargs.get("kind", "perfect")

    def cartesian_product_features(
        self,
        features_dict: Dict[str, torch.Tensor],
        N_max: int = None,  # If not None, cap per-row unique values
        final_cap: int = None,  # If not None, cap final cartesian-product dimension
    ) -> Dict[str, torch.Tensor]:
        """
        1) For each feature (a tensor of shape [batch_size, n_values]), remove duplicates
           row by row using torch.unique().
        2) If N_max is not None, cap the rowwise unique values to N_max via uniform sampling.
        3) Pad each row to the max #unique among all rows. Store result in unique_features[f].
        4) Do a cartesian product across all features to find the 'column indices' for each feature.
        5) Gather from unique_features[f], producing shape [batch_size, total_combinations].
        6) If final_cap is not None, apply uniform sampling again on the final dimension.

        Returns a dict {feature_name -> [batch_size, (final_cap or total_combinations)]}.
        """

        # 1) Sort the feature names just for deterministic processing
        feature_names = sorted(features_dict.keys())

        # 2) Rowwise unique and (optionally) rowwise cap to N_max
        batch_size = None
        unique_features = {}

        for f in feature_names:
            tensor_f = features_dict[f]
            if batch_size is None:
                batch_size = tensor_f.shape[0]
            else:
                assert tensor_f.shape[0] == batch_size, (
                    "All features must have the same batch dimension. "
                    f"Mismatch on feature {f}."
                )

            # We'll collect one 1D tensor per row, each possibly different length
            row_list = []
            for b in range(batch_size):
                row_vals = tensor_f[b]  # shape [n_values]

                # a) row-wise unique (PyTorch's default for 1D is sorted=True by default)
                unique_row_vals, _ = torch.unique(row_vals, return_inverse=True)

                # b) If we want a per-row cap (N_max), apply uniform sampling to that row
                if N_max is not None and unique_row_vals.numel() > N_max:
                    # Put it into shape [1, row_length] so we can reuse uniform_sample_tensor
                    row_2d = unique_row_vals.unsqueeze(0)  # shape [1, row_length]
                    row_2d = uniform_sample_tensor(
                        row_2d, n_samples=N_max
                    )  # shape [1, N_max]
                    unique_row_vals = row_2d.squeeze(0)  # shape [N_max]

                row_list.append(unique_row_vals)

            # c) Pad to a consistent width across all rows for this feature
            max_unique_cols = max(r.size(0) for r in row_list)
            unique_2d = torch.zeros(batch_size, max_unique_cols, device=self.device)
            for b in range(batch_size):
                n_uniq = row_list[b].size(0)
                unique_2d[b, :n_uniq] = row_list[b]

            unique_features[f] = unique_2d  # shape [batch_size, max_unique_cols]

        # 3) Cartesian product across all features
        n_values_list = [unique_features[f].shape[1] for f in feature_names]
        # If any feature has 0 columns, cartesian_prod will be empty
        # (that can happen if a row was fully capped out).
        # But let's assume we always have at least 1 unique.
        indices = torch.cartesian_prod(
            *(torch.arange(nv, device=self.device) for nv in n_values_list)
        )
        # shape: [total_combinations, num_features]

        # total_combinations = indices.shape[0]

        # 4) Gather from unique_features to shape [batch_size, total_combinations]
        #    for each feature
        output = {}
        for i, f in enumerate(feature_names):
            # Each row i in "indices" says which column index to pull from feature f
            idx = indices[:, i].view(1, -1).expand(batch_size, -1)
            gathered = torch.gather(unique_features[f], dim=1, index=idx)

            # 5) final cap if desired: limit the final dimension from total_combinations -> final_cap
            if final_cap is not None:
                gathered = uniform_sample_tensor(gathered, n_samples=final_cap)

            output[f] = gathered  # shape [batch_size, final_cap or total_combinations]

        return output

    def _infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
        plot_prob: bool = False,
        N_max: int = None,
    ):
        # Determine batch size
        if evidence:
            first_key = next(iter(evidence))
            n_queries = evidence[first_key].shape[0]
        else:
            n_queries = 1

        if target_node in evidence:
            points_to_evaluate = evidence[target_node]
        else:
            node_domain_expanded = (
                self.bn.get_domain(target_node).unsqueeze(0).expand(n_queries, -1)
            )
            points_to_evaluate = uniform_sample_tensor(node_domain_expanded, N_max)

        n_points_to_evaluate = points_to_evaluate.shape[1]

        nodes = self.bn.get_nodes()
        n_nodes = len(nodes)

        numerator = torch.zeros(
            (n_nodes, n_queries, n_points_to_evaluate), device="cuda:0"
        )
        denominator = torch.zeros((n_nodes, n_queries), device="cuda:0")

        for node_idx, node in enumerate(nodes):
            evidence_for_inference = {}
            node_parents = self.bn.get_parents(node)
            node_children = self.bn.get_children(node)
            if len(node_parents) > 0:
                for parent in node_parents:
                    if parent in evidence.keys():
                        evidence_for_inference[parent] = evidence[parent]

            """if node in evidence.keys():
                evidence_for_inference[node] = evidence[node]"""

            if len(node_children) > 0:
                # TODO: perhaps pending issue. Check the correctness
                for child in node_children:
                    if child in evidence.keys():
                        evidence_for_inference[child] = evidence[child]

            """if node != target_node:
                if node in evidence.keys():
                    evidence_for_inference[node] = evidence[node]"""

            if len(evidence_for_inference.keys()) > 1:
                # print("1")
                all_combinations_for_evidence = self.cartesian_product_features(
                    evidence_for_inference
                )
            elif len(evidence_for_inference.keys()) == 1:
                # print("2")
                all_combinations_for_evidence = evidence_for_inference
            else:
                # print("3")
                all_combinations_for_evidence = {}

            # print(f"All combinations for {node}: ", all_combinations_for_evidence)
            _, pdf, domain_values = self.bn.get_cpd_and_pdf(
                node,
                all_combinations_for_evidence,
                uncertainty=uncertainty,
                points_to_evaluate=(
                    points_to_evaluate if node == target_node else None
                ),
                N_max=N_max,
            )

            if plot_prob:
                self.bn.plot_prob(pdf, domain_values, title=f"CPD of {node}")

            summed_pdf = pdf.sum(-1)

            denominator[node_idx] = summed_pdf

            summed_pdf = summed_pdf.unsqueeze(-1).expand(-1, n_points_to_evaluate)
            numerator[node_idx] = pdf if node == target_node else summed_pdf

        producer_numerator = numerator.prod(dim=0)  # [n_queries, n_points]
        producer_denominator = denominator.prod(dim=0)  # [n_queries]

        result = torch.where(
            producer_denominator.unsqueeze(-1).expand(-1, n_points_to_evaluate) == 0,
            torch.tensor(0.0, device=producer_denominator.device),  # Replace 0/0 with 0
            producer_numerator
            / producer_denominator.unsqueeze(-1).expand(-1, n_points_to_evaluate),
        )

        return result, points_to_evaluate
