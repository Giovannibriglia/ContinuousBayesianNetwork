from typing import Dict

import torch

from cbn.base.parameter_learning import BaseParameterLearningEstimator


class BruteForce(BaseParameterLearningEstimator):
    def __init__(self, config: Dict, **kwargs):
        super(BruteForce, self).__init__(config=config, **kwargs)
        self.mle_tensor = None
        self._setup_model(config, **kwargs)

    def _setup_model(self, config: Dict = None, **kwargs):
        pass

    def _fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None):
        """

        :param node_data: shape [n_samples]
        :param parents_data: shape [n_parents_features, n_samples]
        """
        # [n_samples, 1]
        node_data = node_data.view(-1, 1)

        if parents_data is not None:
            # parents_data: [n_parents_features, n_samples]
            # we transpose => [n_samples, n_parents]
            parents_data = parents_data.T
            all_data = torch.empty(
                (node_data.shape[0], parents_data.shape[1] + 1),
                dtype=node_data.dtype,
                device=node_data.device,
            )
            all_data[:, :-1] = parents_data  # fill parents
            all_data[:, -1] = node_data.squeeze(-1)
        else:
            all_data = node_data  # shape [n_samples, 1] if no parents

        # unique_rows: shape [n_unique, n_parents + 1]
        # counts: frequency of each unique row
        unique_rows, counts = torch.unique(all_data, dim=0, return_counts=True)
        probs = counts.float() / counts.sum()

        # mle_tensor: [n_unique, n_parents + 1 + 1]
        #    columns = [...parent cols..., node col, probability col]
        self.mle_tensor = torch.empty(
            (unique_rows.shape[0], unique_rows.shape[1] + 1),
            dtype=unique_rows.dtype,
            device=self.device,
        )
        self.mle_tensor[:, :-1] = unique_rows
        self.mle_tensor[:, -1] = probs

    def _get_prob(self, point_to_evaluate: torch.Tensor, query: torch.Tensor = None):
        """
        Compute P(node_value | parents) for each query in 'query' and
        each candidate node value in 'domain'.

        :param point_to_evaluate: [n_queries, n_values]
        :param query:  [n_queries, n_parents, 1]
        :return:       [n_queries, n_values] of conditional probabilities
        """

        # --- 1) Basic checks ---
        assert (
            self.mle_tensor is not None
        ), "MLE tensor not fitted yet. Call _fit() first."
        assert (
            query.dim() == 3 and query.shape[-1] == 1
        ), f"Query must be [n_queries, n_parents, 1]. Got {query.shape}."

        n_queries, n_parents, _ = query.shape
        n_node_values = point_to_evaluate.shape[1]

        if point_to_evaluate.shape[0] != n_queries:
            raise ValueError(
                f"'domain' first dimension must match number of queries. "
                f"Got domain.shape[0] = {point_to_evaluate.shape[0]}, expected {n_queries}."
            )

        # --- 2) Extract MLE data & debug prints ---
        # shape = [n_mle, n_parents + 1]
        mle_data = self.mle_tensor[:, :-1]
        # shape = [n_mle]
        mle_probs = self.mle_tensor[:, -1]

        """print("=== DEBUG: Inside _get_prob ===")
        print(f"mle_data shape = {mle_data.shape}")
        print(f"mle_data dtype = {mle_data.dtype}")
        if mle_data.shape[0] > 0:
            print("First row of mle_data (parents... node_value):", mle_data[0])
            print("First probability in mle_probs:", mle_probs[0])
        else:
            print("mle_data is EMPTY")"""

        # --- 3) Build [parents + node_value] in the shape [n_queries * n_values, n_parents + 1] ---
        parent_query = query.squeeze(-1)  # => [n_queries, n_parents]
        # => [n_queries, n_node_values, n_parents]
        parent_query_exp = parent_query.unsqueeze(1).expand(-1, n_node_values, -1)
        # => [n_queries, n_node_values, n_parents + 1]
        full_query = torch.cat(
            [parent_query_exp, point_to_evaluate.unsqueeze(-1)], dim=-1
        )
        # => [n_queries * n_node_values, n_parents + 1]
        flat_query = full_query.view(-1, n_parents + 1)

        """# Debug prints
        print(f"flat_query shape = {flat_query.shape}")
        print(f"flat_query dtype = {flat_query.dtype}")
        if flat_query.shape[0] > 0:
            print("First row of flat_query (parents... node_value):", flat_query[0])
        else:
            print("flat_query is EMPTY.")"""

        # Optional: If you suspect a dtype mismatch, you can unify them:
        # flat_query = flat_query.to(mle_data.dtype)

        # --- 4) Joint matches => P(parents + node_value) ---
        joint_matches = (flat_query[:, None, :] == mle_data[None, :, :]).all(dim=-1)
        joint_probs = (joint_matches * mle_probs).sum(dim=-1)

        """# Debug: how many matches per row
        sum_joint_matches = joint_matches.sum(dim=1)
        print("joint_matches.sum(dim=1) =", sum_joint_matches[:50], "...")"""

        # --- 5) Parent matches => P(parents) ---
        flat_parents = flat_query[:, :-1]
        mle_parents = mle_data[:, :-1]
        parent_matches = (flat_parents[:, None, :] == mle_parents[None, :, :]).all(
            dim=-1
        )
        parent_probs = (parent_matches * mle_probs).sum(dim=-1)

        """sum_parent_matches = parent_matches.sum(dim=1)
        print("parent_matches.sum(dim=1) =", sum_parent_matches[:50], "...")"""

        """# Example: see which node values are in training for the 1st row's parents
        if flat_parents.shape[0] > 0:
            parents_0 = flat_parents[0]
            # Find all training rows that match these parents
            match_mask = (mle_parents == parents_0).all(dim=1)
            matching_indices = match_mask.nonzero(as_tuple=True)[0]
            if matching_indices.numel() > 0:
                print("DEBUG: For the 1st query's parents, we found training combos =>")
                for idx in matching_indices:
                    print(
                        "  MLE row idx =",
                        idx.item(),
                        "| node_value =",
                        mle_data[idx, -1].item(),
                        "| prob =",
                        mle_probs[idx].item(),
                    )
            else:
                print(
                    "DEBUG: No parent row found in training matching the 1st query's parents"
                )"""

        # --- 6) Compute conditional PDF ---
        eps = 1e-10
        pdf = joint_probs / (parent_probs + eps)
        pdf = pdf.view(n_queries, n_node_values)

        return pdf

    def _sample(self, N: int, **kwargs):
        """
        Draw N samples from the empirical (joint) distribution stored in self.mle_tensor.

        self.mle_tensor has shape [num_unique, n_parents + 1 + 1_for_prob].
        The last column is the probability of each unique combination.
        """
        assert (
            self.mle_tensor is not None
        ), "MLE tensor not fitted yet. Call _fit() first."

        # 1) Extract the probabilities for each unique [parents... node_value] combo
        probs = self.mle_tensor[:, -1]  # shape [num_unique]
        # 2) Sample indices from [0..num_unique-1] with replacement, weighted by probs
        indices = torch.multinomial(probs, N, replacement=True)
        # 3) Return the corresponding [parents... node_value] columns,
        #    ignoring the probability column
        samples = self.mle_tensor[indices, :-1]  # shape [N, n_parents + 1]

        return samples

    def save_model(self, path: str):
        raise NotImplementedError

    def load_model(self, path: str):
        raise NotImplementedError
