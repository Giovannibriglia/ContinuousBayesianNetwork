import networkx as nx

import pandas as pd
import torch
import yaml

from cbn.base.bayesian_network import BayesianNetwork

if __name__ == "__main__":
    # Create a dataset
    data = pd.read_pickle("frozen_lake.pkl")
    data.columns = ["obs_0", "action", "reward"]

    dag = nx.DiGraph()
    dag.add_edges_from([("obs_0", "reward"), ("action", "reward")])

    # Load the YAML file
    with open("../conf/parameter_learning/neural_network.yaml", "r") as file:
        parameters_learning_config = yaml.safe_load(file)

    # Load the YAML file
    with open("../conf/inference/exact.yaml", "r") as file:
        inference_config = yaml.safe_load(file)

    kwargs = {"log": False, "plot_prob": True}

    # Initialize the Bayesian Network
    bn = BayesianNetwork(
        dag=dag,
        data=data,
        parameters_learning_config=parameters_learning_config,
        inference_config=inference_config,
        **kwargs,
    )

    target_node = "reward"
    # Infer CPDs for node C given evidence for A and B
    evidence = {
        # "reward": torch.tensor([[1], [1], [1]], device="cuda"),
        "action": torch.tensor([[1], [2], [3]], device="cuda"),
        # "obs_0": torch.tensor([[10], [14], [10]], device="cuda"),
    }

    """pdfs, target_node_domains, parents_domains = bn.get_pdf(
        target_node,
        evidence,
        N_max=64,
    )
    print("PDF: ", pdfs.shape)
    print("Target node Domains: ", target_node_domains.shape)
    print("Parents Domains: ", parents_domains.shape)"""

    pdf, domain = bn.infer(target_node, evidence, N_max=64, plot_prob=True)
