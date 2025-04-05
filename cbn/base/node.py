import random
from typing import Dict, List, Tuple

import torch

from cbn.base import (
    BASE_MAX_CARDINALITY,
    KEY_CONTINUOUS,
    KEY_DISCRETE,
    KEY_MAX_CARDINALITY_FOR_DISCRETE,
)
from cbn.utils import choose_probability_estimator


class Node:
    def __init__(
        self,
        node_name: str,
        estimator_name: str,
        parameter_learning_config: Dict,
        parents_names: List[str] = None,
        **kwargs,
    ):
        self.node_name = node_name
        self.parameter_learning_config = parameter_learning_config

        self.parents_names = parents_names if parents_names else []

        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_cardinality_for_discrete_domain = kwargs.get(
            KEY_MAX_CARDINALITY_FOR_DISCRETE, BASE_MAX_CARDINALITY
        )
        self.plot_prob = kwargs.get("plot_prob", False)
        self.fixed_dtype = kwargs.get("fixed_dtype", torch.float32)

        self.estimator = choose_probability_estimator(
            estimator_name, parameter_learning_config, **kwargs
        )

        self.info = {}

    def fit(self, node_data: torch.Tensor, parents_data: torch.Tensor = None, **kwargs):
        """

        :param node_data: shape [n_samples]
        :param parents_data: shape [n_parents_features, n_samples]
        """
        node_data.to(self.fixed_dtype)

        if len(self.parents_names) > 0:
            if parents_data is not None:
                if len(self.parents_names) != parents_data.shape[0]:
                    raise ValueError(
                        f"number of parents features in input ({parents_data.shape[0]}) is not equal to number of parents node set ({len(self.parents_names)})"
                    )
                else:
                    parents_data.to(self.fixed_dtype)

                    # Sorting parents names and data
                    start_parents_names = self.parents_names
                    # Sort the list
                    self.parents_names = sorted(self.parents_names)

                    # Get indices of the sorted elements in the original list
                    index_map = [
                        start_parents_names.index(val) for val in self.parents_names
                    ]

                    # Reorder the tensor accordingly
                    parents_data = parents_data[index_map]

            else:
                raise ValueError(
                    f"parents data is empty; should be [{node_data.shape[0], len(self.parents_names)}]"
                )
        else:
            if parents_data is not None:
                raise ValueError("there are no parents for which setting data.")

        self.estimator.fit(node_data, parents_data)

        unique_node_data = torch.unique(node_data)
        self.info[self.node_name] = [
            torch.min(node_data),
            torch.max(node_data),
            (
                KEY_CONTINUOUS
                if len(unique_node_data) > self.max_cardinality_for_discrete_domain
                else KEY_DISCRETE
            ),
            unique_node_data,
        ]

        if parents_data is not None and len(self.parents_names) > 0:
            unique_parents_data = torch.unique(parents_data, dim=1)
            for i, parent in enumerate(self.parents_names):
                self.info[parent] = [
                    torch.min(parents_data[i]),
                    torch.max(parents_data[i]),
                    (
                        KEY_CONTINUOUS
                        if len(unique_parents_data[i])
                        > self.max_cardinality_for_discrete_domain
                        else KEY_DISCRETE
                    ),
                    unique_parents_data[i],
                ]

    def sample(self, N: int, **kwargs) -> torch.Tensor:
        return self.estimator.sample(N, **kwargs)

    def get_prob(
        self, query: Dict[str, torch.Tensor], N: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param query: dict of torch. Tensors, each one has shape [n_queries, 1]
        :param N: number of samples required in evidence is not provided
        :return: pdf and domain. shape [n_queries, 1] if node is the query else [n_queries, n_values]. n_values = N if some evidence is missed else 1
        """

        n_queries = query[next(iter(query.keys()))].shape[0]
        n_parents = len(self.parents_names)

        # check n_queries
        assert all(
            [query[feat].shape[0] == n_queries for feat in query.keys()]
        ), ValueError("n_queries must be equal for all features.")
        assert all([query[feat].dim() == 2 for feat in query.keys()]), ValueError(
            "each tensor for the query variables must have dimension = 2."
        )

        # get_target_feature
        node_query = query.pop(self.node_name, None)

        # setup query
        parents_query, parents_domains = self._setup_parents_query(query, N)
        # if evidence is complete: [n_queries, n_parents_features, 1], [n_queries, n_parents_features, 1] else
        # [n_queries, n_parents_features, n_combinations], [n_queries, n_parents_features, N]

        n_samples_for_parent = parents_query.shape[2]

        if node_query is None:
            target_node_domains = (
                self._sample_domain(self.node_name, N)
                .unsqueeze(0)
                .expand(
                    n_queries,
                    -1,
                )
            )
        else:
            # TODO
            target_node_domains = node_query

        if parents_query.shape[2] > 1:
            pdfs = torch.empty((n_queries, n_parents, n_samples_for_parent))
            for i in range(n_queries):
                new_parents_query = parents_query[i].T.unsqueeze(-1)
                pdfs[i] = self.estimator.get_prob(
                    target_node_domains, new_parents_query
                ).T  # [n_queries, domain_node_feature], [n_queries, domain_node_feature]
        elif parents_query.shape[2] == 0:
            raise ValueError(f"parents_query has bad shape: {parents_query.shape}")
        else:
            # evaluate query
            pdfs = self.estimator.get_prob(
                target_node_domains, parents_query
            )  # [n_queries, domain_node_feature], [n_queries, domain_node_feature]

        # pdfs: [n_queries, n_node_values]
        # target_node_domains: [n_queries, n_node_values]
        # parents_domains: [n_queries, n_parents, n_samples_for_parent]

        if self.plot_prob:
            self._plot_pdfs(pdfs, target_node_domains, parents_domains)

        assert pdfs.shape == target_node_domains.shape, ValueError(
            f"pdf and domain must have same shape; instead: {pdfs.shape} and {target_node_domains.shape}"
        )
        return pdfs, target_node_domains, parents_domains

    def _setup_parents_query(
        self, query: Dict[str, torch.Tensor], N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: query as a Dict of torch.Tensors, each one has shape [n_queries, 1]
        :param N: number of samples required if evidence is not provided
        :return: query as a torch.Tensor with shape [n_queries, n_parents_features, n_parents_combinations] and parents evaluation points [n_queries, n_parents_features, 1 or N]
        """

        query_features = sorted(list(query.keys()))
        query = {key: query[key] for key in query_features}
        if len(query_features) > 0:
            parents_in_query = list(query.keys())
            n_start_queries = query[parents_in_query[0]].shape[0]

            assert all(
                query_feat in self.parents_names for query_feat in query_features
            ), ValueError("You have specified parent features that don't exist")

            if query_features == self.parents_names:
                new_query = torch.zeros(
                    (n_start_queries, len(self.parents_names), 1),
                    device=self.device,
                )
                for i, feature_tensor in enumerate(query.values()):
                    new_query[:, i, :] = feature_tensor

                parents_evaluation_points = new_query
            else:
                parents_evaluation_points = torch.empty(
                    (n_start_queries, len(self.parents_names), N),
                    device=self.device,
                    dtype=self.fixed_dtype,
                )

                n_missing_parents = 0
                for i, parent_feature in enumerate(self.parents_names):
                    if parent_feature in query_features:
                        # print("1: ", query[parent_feature].expand(-1, N).shape)
                        parents_evaluation_points[:, i, :] = query[
                            parent_feature
                        ].expand(-1, N)
                    else:
                        n_missing_parents += 1

                        uniform_distribution = (
                            self._sample_domain(parent_feature, N)
                            .unsqueeze(0)
                            .expand(n_start_queries, -1)
                        )
                        # print("2: ", uniform_distribution.shape)
                        parents_evaluation_points[:, i, :] = uniform_distribution

                new_query = self._batched_meshgrid_combinations(
                    parents_evaluation_points
                )
        else:
            n_start_queries = 1

            parents_evaluation_points = torch.empty(
                (n_start_queries, len(self.parents_names), N),
                device=self.device,
                dtype=self.fixed_dtype,
            )
            for i, parent_feature in enumerate(self.parents_names):
                uniform_distribution = (
                    self._sample_domain(parent_feature, N)
                    .unsqueeze(0)
                    .expand(n_start_queries, -1)
                )
                parents_evaluation_points[i] = uniform_distribution

            n_missing_parents = len(self.parents_names)
            new_query = self._batched_meshgrid_combinations(parents_evaluation_points)

        return new_query, parents_evaluation_points

    def _sample_domain(self, node: str, N: int = 1024) -> torch.Tensor:
        min_value, max_value, domain_kind, domain_values = self.info[node]
        cardinality = domain_values.shape[0]

        # If it's a discrete domain, you might just return all values
        # (or sample from them randomly if you prefer).
        if domain_kind == KEY_DISCRETE:
            return domain_values

        # Otherwise, assume it's continuous (or at least "sortable").
        if N < cardinality:
            # Uniformly select N points by index from domain_values
            indices = torch.linspace(start=0, end=cardinality - 1, steps=N)
            indices = indices.round().long()
            sampled = domain_values[indices]
            return sampled

        elif N == cardinality:
            # Exactly the same size => just return domain_values as-is
            return domain_values

        else:
            # N > cardinality
            # 1) We'll keep all of the original domain_values
            # 2) We'll add (N - cardinality) new values, chosen randomly
            #    in [min_value, max_value], ensuring they're not duplicates.
            needed = N - cardinality
            existing = set(domain_values.tolist())
            new_values = []

            # Try to add 'needed' distinct new float values
            # This can be slow if domain_values is huge or if min_value == max_value.
            while len(new_values) < needed:
                candidate = random.uniform(min_value, max_value)
                if candidate not in existing:
                    new_values.append(candidate)
                    existing.add(candidate)

            # Convert our new list to a Tensor and concatenate
            new_values_tensor = torch.tensor(
                new_values, dtype=domain_values.dtype, device=self.device
            )
            out = torch.cat([domain_values, new_values_tensor])

            # Finally, sort before returning
            out, _ = torch.sort(out)
            return out

    def _batched_meshgrid_combinations(
        self, input_tensor: torch.Tensor, indexing: str = "ij"
    ) -> torch.Tensor:
        """
        Given a tensor with shape [n_parents, n_queries, N], return a tensor of shape [n_queries, n_parents, N^n_parents],
        where each batch m builds a meshgrid from [t[m] for t in tensors], and flattens the result.

        Args:
            input_tensor (torch.Tensor): shape: [n_parents, n_queries, N]
            indexing (str): 'ij' or 'xy' indexing (default: 'ij')

        Returns:
            Tensor: shape [n_queries, n_parents, N^n_parents]
        """
        # input_tensor shape is [n_queries, n_parents, N]
        n_queries, n_parents, N = input_tensor.shape

        # Output shape: [n_queries, n_parents, N^n_parents]
        n_combinations = N**n_parents
        out = torch.empty(
            (n_queries, n_parents, n_combinations),
            dtype=self.fixed_dtype,
            device=self.device,
        )

        for m in range(n_queries):
            # slices: list of n_parents 1D tensors (each of length N),
            # pulled from row i of input_tensor[m].
            # i.e., input_tensor[m, i, :] is shape [N]
            slices = [input_tensor[m, i].to(self.fixed_dtype) for i in range(n_parents)]

            # Create the meshgrid: list of n_parents grids, each of shape [N, ..., N]
            mesh = torch.meshgrid(*slices, indexing=indexing)

            # Stack the meshgrid results along dim=0 => [n_parents, N, N, ...]
            stacked = torch.stack(mesh, dim=0)

            # Flatten the cartesian product => [n_parents, N^n_parents]
            out[m] = stacked.reshape(n_parents, -1)

        return out

    def save_node(self, path: str):
        raise NotImplementedError  # model (and info domains?)

    def load_node(self, path: str):
        raise NotImplementedError  # model (and info domains?)

    @staticmethod
    def _plot_pdfs(
        pdfs: torch.Tensor, node_domains: torch.Tensor, parents_domains: torch.Tensor
    ):
        """
        Plot PDFs for the target node (node_domains) that have been computed
        given the parents' domains (parents_domains). The function also prints out
        the parents' domain values per query to clarify what led to each PDF.

        Args:
            pdfs (torch.Tensor): Shape [n_queries, n_node_values], containing
                                 probabilities (PDF) for the node's domain.
            node_domains (torch.Tensor): Shape [n_queries, n_node_values], containing
                                         the actual domain values of the node being plotted.
            parents_domains (torch.Tensor): Shape [n_queries, n_parents, n_parents_values],
                                            containing the domain values of the parent nodes
                                            that condition each PDF.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        """node_domains = node_domains.T.unsqueeze(0).expand(
            parents_domains.shape[0], -1, -1
        )"""

        pdfs_np = pdfs.detach().cpu().numpy()
        node_domains_np = node_domains.detach().cpu().numpy()
        parents_domains_np = parents_domains.detach().cpu().numpy()

        # Make sure first dimensions match
        if pdfs_np.shape[0] != node_domains_np.shape[0]:
            raise ValueError(
                "pdfs and node_domains must have the same number of queries."
            )

        # Log shapes for debugging
        print("pdfs_np shape:", pdfs_np.shape)
        print("node_domains_np shape:", node_domains_np.shape)
        print("parents_domains_np shape:", parents_domains_np.shape)

        # Plot
        """plt.figure()
        for i in range(len(pdfs_np)):
            plt.plot(node_domains_np[i], pdfs_np[i], label=f"query {i}")
            plt.xlabel("Domain")
            plt.ylabel("PDF")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()"""

        # Compute most-probable value
        max_idxs = np.argmax(pdfs_np, axis=1)  # shape: (n_queries,)
        most_probable_values = node_domains_np[
            np.arange(len(node_domains_np)), max_idxs
        ]
        print("Prediction:", most_probable_values)
        print(np.unique(most_probable_values))
        # Plot the predictions
        plt.figure()
        plt.scatter(np.arange(len(most_probable_values)), most_probable_values, s=30)
        plt.xlabel("Query Index")
        plt.ylabel("Most probable value")
        # plt.xticks(np.unique(most_probable_values))
        plt.ylim(ymin=-0.01, ymax=1.01)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
