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
    D = torch.linspace(40, 60, n_samples, device=device)
    E = torch.linspace(40, 60, n_samples, device=device)
    parents_data = torch.cat(
        [B.unsqueeze(0), C.unsqueeze(0), D.unsqueeze(0), E.unsqueeze(0)], dim=0
    )

    node_data = 4 * B + C / 10 * E + torch.sqrt(D)

    node1.fit(node_data, parents_data)

    list_values = range(0, n_samples, 1)

    random_queries = random.sample(list_values, n_queries)
    # print(random_queries)
    print("Ground truth: ", node_data[random_queries])
    query = {  # check n_queries
        "B": B[random_queries].unsqueeze(-1),
        "C": C[random_queries].unsqueeze(-1),
        "D": D[random_queries].unsqueeze(-1),
        "E": E[random_queries].unsqueeze(-1),
    }

    pdf, domain = node1.get_prob(query, N=4)

    print("Pdf: ", pdf.shape)
    print("Domain: ", domain.shape)


if __name__ == "__main__":
    e = input("Estimator: ")
    n = int(input("Number of samples for each feature: "))
    q = int(input("Number of queries: "))
    arithmetic_node(e, n, q)
