from typing import Dict

import torch

from cbn.base import initial_uncertainty
from cbn.base.inference import BaseInference


class ExactInference(BaseInference):
    def __init__(self, bn, device: str = "cpu", **kwargs):
        super().__init__(bn, device, **kwargs)

        # technique = kwargs.get("kind", "perfect")

    """def infer(
        self,
        target_node: str,
        evidence: dict,
        do: dict,
        uncertainty: float = 0.0,  # or whatever your default
    ):
        # Determine batch size
        if evidence:
            # Grab first key’s shape as batch size (assuming all evidence has same batch size)
            first_key = next(iter(evidence))
            batch_size = evidence[first_key].shape[0]
        else:
            batch_size = 1

        # Number of possible values for target_node
        n_values = self.bn.get_domain(target_node).shape[0]


        #if do:
            #self.bn.intervene(do)


        # Get all ancestors of target_node (plus the node itself)
        ancestors = self.bn.get_ancestors(target_node)
        ancestors.append(target_node)

        # We'll collect each factor here
        factors = []

        for ancestor in ancestors:
            # Get the parents of this ancestor
            ancestor_parents = self.bn.get_parents(ancestor)

            # Build the relevant evidence for these parents
            extracted_evidence = {
                k: v for k, v in evidence.items() if k in ancestor_parents
            }

            # If this ancestor itself is known in the evidence, we add that too
            # (e.g., if X is the ancestor and X in evidence).
            if ancestor in evidence:
                extracted_evidence[ancestor] = evidence[ancestor]
                points_to_evaluate = None
            else:
                # If not in evidence, we might need to evaluate all possible domain values for `ancestor`
                points_to_evaluate = self.bn.get_domain(ancestor)

            # Retrieve CPD/PDF for the ancestor given the extracted_evidence.
            # `cpd` is the conditional probability distribution if you need it,
            # `pdf` is presumably shaped like (domain_size, batch_size) or similar.
            cpd, pdf, _ = self.bn.get_cpd_and_pdf(
                ancestor, extracted_evidence, points_to_evaluate=points_to_evaluate
            )

            # If this is the target node, we may want the PDF in shape [batch_size, domain_of_target].
            # We'll assume `pdf` already has shape [batch_size, domain_size] or [domain_size, batch_size].
            # If needed, you can reorder dimensions or normalize:
            if ancestor == target_node:
                # Example: normalize over domain dimension if pdf is [batch_size, n_values].
                # Make sure your `self.normalize_pdf` does exactly that.
                pdf_summed = self.normalize_pdf(pdf)
            else:
                # For non-target nodes in many standard factorization approaches, you'd multiply the *corresponding*
                # probabilities per batch. If your BN code returns pdf in shape [batch_size, domain_size],
                # you typically do something like factor = pdf (elementwise) for each batch config.
                #
                # However, your snippet was summing or averaging:
                # pdf_summed = torch.sum(pdf) / (pdf.shape[0] * pdf.shape[1])
                #
                # That effectively collapses the entire distribution to a single scalar.
                # Unless you have a specific reason for that, a more typical approach is to keep the
                # probabilities per batch dimension and multiply them in.
                # Below, we just keep your summation if that’s truly intended:
                pdf_summed = torch.sum(pdf) / (pdf.shape[0] * pdf.shape[1])

            # Collect the factor
            factors.append(pdf_summed)

        # Multiply all the factors.
        # Depending on shapes, you might need broadcasting or per-batch multiplication.
        # Here we initialize result as shape [batch_size, n_values] of ones:
        result = torch.ones((batch_size, n_values), device="cuda:0")

        for factor in factors:
            # factor might be shape [batch_size, n_values] for the target node
            # or a scalar for the non-target nodes (per the line above).
            # Broadcasting should handle it:
            result = result * factor

        # Final shape check
        assert result.shape == (
            batch_size,
            n_values,
        ), f"result shape is {result.shape}, expected {(batch_size, n_values)}."

        return result

    @staticmethod
    def normalize_pdf(pdf):

        # Compute the normalization factor by summing over the second axis
        normalization_factor = pdf.sum(
            dim=1, keepdim=True
        )  # Shape: [batch_size, 1, n_values]

        # Normalize the pdf
        normalized_pdf = pdf / (
            normalization_factor + min_tolerance
        )  # Add small value to avoid division by zero

        # Return shape [batch_size, n_values]
        return normalized_pdf.sum(dim=1)"""

    def cartesian_product_features(self, features_dict):
        """
        Given a dict of {feature_name: tensor of shape [batch_size, n_values]},
        this version:
          1) Removes duplicate columns for each feature, per batch example.
          2) Then performs the cartesian product over unique columns only.
          3) Gathers results into final shape [batch_size, total_combinations].

        Note: If batch_size > 1, we remove duplicates for each row individually.
              So row 0's duplicates are removed independently of row 1's duplicates, etc.
        """

        # Sort features for deterministic order
        feature_names = sorted(features_dict.keys())

        batch_size = None
        # We'll store, for each feature, the "unique version" of the feature
        # plus a mapping from "original column index" -> "unique column index".
        # We'll handle duplicates for each batch row separately.
        unique_features = {}
        orig_to_unique_map = {}

        for f in feature_names:
            tensor_f = features_dict[f]
            if batch_size is None:
                batch_size = tensor_f.shape[0]
            else:
                assert tensor_f.shape[0] == batch_size, (
                    "All features must have the same batch dimension. "
                    f"Mismatch on feature {f}."
                )

            # shape: [batch_size, n_values]
            # We'll do duplicates removal row by row.
            # Example approach: for each row, call torch.unique on that row,
            # gather the results, and store the mapping from the row's columns -> unique col index.
            # Then we cat everything across rows. But that can complicate usage of cartesian_prod,
            # which is a single product across all columns for the entire batch.
            #
            # A simpler approach is to do the unique operation *column-wise across the entire batch dimension*
            # if that suits your data. Or do it row-by-row in a loop.

            # For demonstration, let's do it "row by row":
            # unique_values[i], unique_indices[i], inverse_indices[i] for row i in [0..batch_size-1]
            row_list = []
            map_list = []
            for b in range(batch_size):
                row_vals = tensor_f[b]  # shape [n_values]
                # remove duplicates in this row
                unique_row_vals, inverse_indices = torch.unique(
                    row_vals, return_inverse=True
                )
                # unique_row_vals: shape [num_unique]
                # inverse_indices: shape [n_values], telling us how each original col maps to [0..num_unique-1]

                row_list.append(unique_row_vals)
                map_list.append(inverse_indices)

            # Now we have a list of length batch_size, each a 1D tensor with possibly different num_unique.
            # We'll store them in a single 2D tensor by padding to the max number of unique columns for that feature.
            max_unique_cols = max(row.size(0) for row in row_list)
            # e.g. build shape [batch_size, max_unique_cols]
            unique_2d = torch.zeros(batch_size, max_unique_cols, device=self.device)
            map_2d = torch.zeros(
                batch_size, tensor_f.shape[1], dtype=torch.long, device=self.device
            )
            for b in range(batch_size):
                n_uniq = row_list[b].numel()
                unique_2d[b, :n_uniq] = row_list[b]
                map_2d[b] = map_list[b]

            unique_features[f] = unique_2d  # [batch_size, max_unique_cols]
            orig_to_unique_map[f] = (
                map_2d  # [batch_size, n_values] -> index in [0..max_unique_cols-1]
            )

        # Now, each feature f has a "unique_2d" (some might have different max_unique_cols).
        # For cartesian product, we need a single dimension length for each feature. Let's get them:
        n_values_list = []
        for f in feature_names:
            n_values_list.append(unique_features[f].shape[1])

        # cartesian_prod on those dimensions
        indices = torch.cartesian_prod(
            *(torch.arange(nv, device=self.device) for nv in n_values_list)
        )
        # shape: [total_combinations, n_features]

        # Gather from the unique_features
        # shape after gather => [batch_size, total_combinations]
        output = {}
        for i, f in enumerate(feature_names):
            # indices[:, i] are the column indices to pick from unique_features[f]
            idx = (
                indices[:, i].view(1, -1).expand(batch_size, -1)
            )  # [batch_size, total_combinations]
            # gather from unique_features
            gathered = torch.gather(unique_features[f], dim=1, index=idx)
            output[f] = gathered

        return output

    def _infer(
        self,
        target_node: str,
        evidence: Dict,
        do: Dict,
        uncertainty: float = initial_uncertainty,
        plot_prob: bool = False,
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
            points_to_evaluate = (
                self.bn.get_domain(target_node).unsqueeze(0).expand(n_queries, -1)
            )

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
                    """else:
                        evidence_for_inference[parent] = (
                            self.bn.get_domain(parent)
                            .unsqueeze(0)
                            .expand(n_queries, -1)
                        )"""

            if len(node_children) > 0:
                # TODO: pending issue. Check the correctness
                for child in node_children:
                    if child in evidence.keys():
                        evidence_for_inference[child] = evidence[child]

            """else:
                if node in evidence.keys():
                    evidence_for_inference[node] = evidence[node]"""

            """else:
                            evidence_for_inference[node] = self.bn.get_domain(node).expand(
                                n_queries, -1
                            )"""

            if len(evidence_for_inference.keys()) > 1:
                all_combinations_for_evidence = self.cartesian_product_features(
                    evidence_for_inference
                )
            elif len(evidence_for_inference.keys()) == 1:
                all_combinations_for_evidence = evidence_for_inference
            else:
                all_combinations_for_evidence = {}

            _, pdf, domain_values = self.bn.get_cpd_and_pdf(
                node,
                all_combinations_for_evidence,
                points_to_evaluate=(
                    points_to_evaluate if node == target_node else None
                ),
            )

            if plot_prob:
                self.bn.plot_prob(pdf, domain_values, title=f"CPD of {node}")

            summed_pdf = pdf.sum(-1)

            denominator[node_idx] = summed_pdf

            summed_pdf = summed_pdf.unsqueeze(-1).expand(-1, n_points_to_evaluate)
            numerator[node_idx] = pdf if node == target_node else summed_pdf

        producer_numerator = numerator.prod(dim=0)  # [n_queries, n_points]
        producer_denominator = denominator.prod(dim=0)  # [n_queries]

        """res = producer_numerator / producer_denominator.unsqueeze(-1).expand(
            -1, n_points_to_evaluate
        )"""

        result = torch.where(
            producer_denominator.unsqueeze(-1).expand(-1, n_points_to_evaluate) == 0,
            torch.tensor(0.0, device=producer_denominator.device),  # Replace 0/0 with 0
            producer_numerator
            / producer_denominator.unsqueeze(-1).expand(-1, n_points_to_evaluate),
        )

        return result, points_to_evaluate
