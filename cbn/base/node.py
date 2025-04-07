import random

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        :param query: dict of torch.Tensors, each with shape [n_queries, 1]
        :param N: number of samples if evidence is not provided
        :return: pdf, target_node_domains, parents_domains
                 - pdf: tensor of shape [n_queries, d_0, d_1, ..., d_{n_parents}, n_samples_node]
                 - target_node_domains: tensor of shape [n_queries, n_samples_node]
                 - parents_domains: a list (length n_parents) of tensors, each of shape [n_queries, d_i]
        """
        # Validate and get n_queries.
        if query:
            n_queries = next(iter(query.values())).shape[0]
            for feat, tensor in query.items():
                assert tensor.shape[0] == n_queries, ValueError(
                    "n_queries must be equal for all features."
                )
                assert tensor.dim() == 2, ValueError(
                    "Each query tensor must be of dimension 2."
                )
        else:
            n_queries = 1

        # Extract target node query (if any)
        node_query = query.pop(self.node_name, None)

        # Set up parent's query.
        parents_query, parents_domains = self._setup_parents_query(query, N)
        total_parents_combinations = parents_query.shape[2]
        print(total_parents_combinations)
        # parents_query has shape [n_queries, n_parents, total_parents_combinations]
        # parents_domains is a list of length n_parents, each tensor of shape [n_queries, d_i]

        # Get target node evaluation points.
        if node_query is None:
            target_node_domains = (
                self._sample_domain(self.node_name, N)
                .unsqueeze(0)
                .expand(n_queries, -1)
            )
        else:
            target_node_domains = node_query
        n_samples_node = target_node_domains.shape[1]

        parent_dims = []
        if total_parents_combinations > 1:
            for _ in self.parents_names:
                parent_dims.append(N)
        else:
            for _ in self.parents_names:
                parent_dims.append(1)

        # Initialize pdf tensor.
        pdfs = torch.empty(
            (n_queries, total_parents_combinations, n_samples_node),
            dtype=self.fixed_dtype,
            device=self.device,
        )
        if total_parents_combinations > 1:
            new_parents_query = parents_query.permute(2, 1, 0)
            for i in range(new_parents_query.shape[2]):
                q = new_parents_query[:, :, i, None]
                new_target_node_domains = (
                    target_node_domains[i]
                    .unsqueeze(0)
                    .expand(total_parents_combinations, -1)
                )
                pdfs[i, :, :] = self.estimator.get_prob(new_target_node_domains, q)
        else:
            # For each configuration (Cartesian product index), compute the pdf.
            for i in tqdm(range(total_parents_combinations)):
                # q has shape [n_queries, n_parents, 1]: one value per parent for this configuration.
                q = parents_query[:, :, i, None]
                # Each call returns [n_queries, n_samples_node]
                pdfs[:, i, :] = self.estimator.get_prob(target_node_domains, q)

        # Reshape pdfs to include a separate dimension for each parent.
        new_shape = [n_queries] + parent_dims + [n_samples_node]
        pdfs = pdfs.view(*new_shape)

        if self.plot_prob:
            self._plot_pdfs(pdfs, target_node_domains, parents_domains)

        return pdfs, target_node_domains, parents_domains

    def _setup_parents_query(
        self, query: Dict[str, torch.Tensor], N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: Dict of torch.Tensors, each with shape [n_queries, 1]
        :param N: number of samples if evidence is not provided
        :return:
             new_query: tensor of shape [n_queries, n_parents, total_combinations],
                        where total_combinations is the Cartesian product of each parent's evaluation points.
             parents_evaluation_points: tensor of shape [n_queries, n_parents, (1 or N)]
        """
        query_features = sorted(list(query.keys()))
        # Reorder query according to the sorted keys.
        query = {key: query[key] for key in query_features}

        if query_features:
            n_start_queries = query[query_features[0]].shape[0]
            # Ensure all provided query features are valid parents.
            assert all(
                feat in self.parents_names for feat in query_features
            ), ValueError("You have specified parent features that don't exist")

            if query_features == self.parents_names:
                # All parent evidences provided.
                new_query = torch.zeros(
                    (n_start_queries, len(self.parents_names), 1), device=self.device
                )
                for i, parent in enumerate(self.parents_names):
                    new_query[:, i, :] = query[parent]
                parents_evaluation_points = new_query
            else:
                # Not all parents provided: fill evaluation points.
                parents_evaluation_points = torch.empty(
                    (n_start_queries, len(self.parents_names), N),
                    device=self.device,
                    dtype=self.fixed_dtype,
                )
                for i, parent_feature in enumerate(self.parents_names):
                    if parent_feature in query_features:
                        # Provided evidence is expanded to have N columns.
                        parents_evaluation_points[:, i, :] = query[
                            parent_feature
                        ].expand(-1, N)
                    else:
                        # Sample uniformly from the parent's domain.
                        uniform_distribution = (
                            self._sample_domain(parent_feature, N)
                            .unsqueeze(0)
                            .expand(n_start_queries, -1)
                        )
                        parents_evaluation_points[:, i, :] = uniform_distribution

                # Create a meshgrid (Cartesian product) across parent dimensions.
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
                parents_evaluation_points[:, i, :] = uniform_distribution
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

    def _plot_pdfs(
        self,
        pdfs: torch.Tensor,
        node_domains: torch.Tensor,
        parents_domains: torch.Tensor,
    ):
        """
        Args:
            pdfs (torch.Tensor): Shape [n_queries, d0, d1, ..., d_{n_parents-1}, n_samples_node],
                                 where each d_i is either 1 (provided evidence) or N (sampled).
            node_domains (torch.Tensor): Shape [n_queries, n_node_values],
                                         containing the domain values for the target node.
            parents_domains (torch.Tensor): Shape [n_queries, n_parents, (1 or N)],
                                            containing the domain values for each parent.
        """
        import math

        import matplotlib.pyplot as plt
        import numpy as np

        """print("pdfs.shape: ", pdfs.shape)
        print("node_domains.shape: ", node_domains.shape)
        print("parents_domains.shape: ", parents_domains.shape)"""

        # Convert tensors to NumPy arrays.
        pdfs_np = pdfs.detach().cpu().numpy()
        node_domains_np = node_domains.detach().cpu().numpy()
        parents_domains_np = parents_domains.detach().cpu().numpy()

        n_queries = pdfs_np.shape[0]
        # The parent's dimensions are pdfs_np.shape[1:-1]
        parent_dims = pdfs_np.shape[1:-1]
        n_parents = len(parent_dims)
        # n_samples_node = pdfs_np.shape[-1]

        # For each query, create a figure with one subplot per parent.
        for q in range(n_queries):
            n_cols = min(n_parents, 2)
            n_rows = math.ceil(n_parents / n_cols)
            fig = plt.figure(figsize=(8 * n_cols, 4 * n_rows), dpi=500)

            subtitle = f"Query {q} - P({self.node_name}|"
            # For each parent, average (marginalize) over all other parent's dimensions.
            # pdfs_np[q] has shape [d0, d1, ..., d_{n_parents-1}, n_samples_node]
            for p in range(n_parents):
                ax = fig.add_subplot(n_rows, n_cols, p + 1, projection="3d")

                # Compute the axes over which to average: all parent axes except p.
                n_parent_axes = len(pdfs_np[q].shape) - 1  # number of parent dimensions
                axes_to_avg = tuple(i for i in range(n_parent_axes) if i != p)
                # Marginalize over all other parents: result shape will be (d_p, n_samples_node)
                pdf_plot = np.mean(pdfs_np[q], axis=axes_to_avg)

                # Get the parent's domain values from parents_domains_np.
                # parents_domains_np[q, p, :] has shape (d_p,)
                parent_vals = parents_domains_np[q, p, :]
                node_vals = node_domains_np[q]  # shape (n_samples_node,)

                unique_parent_vals = np.unique(parent_vals)

                if len(unique_parent_vals) > 1:
                    subtitle += f"{self.parents_names[p]}, "
                    # Create a meshgrid: X for node domain, Y for parent domain.
                    P, N_ = np.meshgrid(parent_vals, node_vals, indexing="ij")
                    # pdf_plot should have shape [d_p, n_samples_node]
                    surf = ax.plot_surface(
                        N_,
                        P,
                        pdf_plot,
                        cmap="viridis",
                        edgecolor="k",
                        linewidth=0.8,
                        alpha=0.8,
                    )
                    ax.set_title(f"{self.parents_names[p]}")
                    ax.set_xlabel(f"Domain of {self.node_name}")
                    ax.set_ylabel(f"Domain of {self.parents_names[p]}")
                    ax.set_zlabel("PDF")
                    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15)
                else:
                    subtitle += f"{self.parents_names[p]}={unique_parent_vals[0]}, "
                    # Only one value for this parent: use a simple 2D line plot.
                    # pdf_plot is of shape [1, n_samples_node] -> squeeze to [n_samples_node]
                    ax = fig.add_subplot(n_rows, n_cols, p + 1)
                    ax.plot(node_vals, pdf_plot[0], marker="o")
                    ax.set_title(f"{self.parents_names[p]} = {unique_parent_vals[0]}")
                    ax.set_xlabel(f"Domain of {self.node_name}")
                    ax.set_ylabel("PDF")
                    ax.set_ylim(ymin=-0.001, ymax=1.001)
                    ax.grid(True)

            subtitle = subtitle[:-2]
            subtitle += ")"
            fig.suptitle(subtitle)
            plt.show()
