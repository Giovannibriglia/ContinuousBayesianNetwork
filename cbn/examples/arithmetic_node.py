import random

import torch
import yaml

from cbn.base.node import Node


def arithmetic_node(
    estimator_name: str, n_samples: int = 100, n_queries: int = 10, seed: int = 42
):

    torch.random.manual_seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_name = "A"
    parents_names = ["B", "C", "D", "E"]

    with open(f"../conf/parameter_learning/{estimator_name.lower()}.yaml", "r") as file:
        parameter_learning_config = yaml.safe_load(file)

    node1 = Node(
        node_name,
        estimator_name,
        parameter_learning_config,
        parents_names,
        log=True,
        plot_prob=True,
    )

    B = torch.linspace(1, 100, n_samples, device=device)
    C = torch.linspace(40, 60, n_samples, device=device)
    D = torch.linspace(50, 55, n_samples, device=device)
    E = torch.linspace(700, 750, n_samples, device=device)
    parents_data = torch.cat(
        [B.unsqueeze(0), C.unsqueeze(0), D.unsqueeze(0), E.unsqueeze(0)], dim=0
    )

    node_data = 4 * B + C / 10 + E + torch.sqrt(D)

    node1.fit(node_data, parents_data)

    # print(random_queries)
    print("Ground truth: ", node_data[:n_queries])
    query = {  # check n_queries
        # "B": B[:n_queries].unsqueeze(-1),
        # "C": C[:n_queries].unsqueeze(-1),
        # "D": D[:n_queries].unsqueeze(-1),
        # "E": E[:n_queries].unsqueeze(-1),
    }

    pdfs, target_node_domains, parents_domains = node1.get_prob(query, N=6)
    #
    print("Pdf: ", pdfs.shape)
    print("Target domain: ", target_node_domains.shape)
    print("Parents domain: ", parents_domains.shape)


if __name__ == "__main__":
    e = "mle"  # input("Estimator: ")
    n = 1000  # int(input("Number of samples for each feature: "))
    q = 1  # int(input("Number of queries: "))
    arithmetic_node(e, n, q)
