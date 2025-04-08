import random

import pandas as pd
import torch
import yaml

from cbn.base.node import Node


def frozen_lake_node(estimator_name: str, n_queries: int = 10, seed: int = 42):

    torch.random.manual_seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_name = "reward"
    parents_names = ["obs", "action"]

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

    df = pd.read_pickle("frozen_lake.pkl")
    obs = torch.tensor(df[0].values, dtype=torch.float32, device=device)
    action = torch.tensor(df[1].values, dtype=torch.float32, device=device)

    train_x = torch.cat([obs.unsqueeze(0), action.unsqueeze(0)], dim=0).to("cuda")
    train_y = torch.tensor(df[2].values, dtype=torch.float32).to("cuda")

    node1.fit(train_y, train_x)

    query = {
        # "obs": obs[:n_queries].unsqueeze(-1),
        # "action": action[:n_queries].unsqueeze(-1),
        # "reward": train_y[:n_queries].unsqueeze(-1),
    }

    """query = {
        "obs": torch.tensor([[14], [4], [3]], device=device),
        # "action": torch.tensor([[2], [1], [3]], device=device),
    }"""

    pdfs, target_node_domains, parents_domains = node1.get_prob(query, N=64)

    print("Pdf: ", pdfs.shape)
    print("Target domain: ", target_node_domains.shape)
    print("Parents domain: ", parents_domains.shape)


if __name__ == "__main__":
    e = input("Estimator: ")
    q = int(input("Number of queries: "))
    frozen_lake_node(e, q)
