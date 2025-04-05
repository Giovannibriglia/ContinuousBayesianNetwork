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
    with open("../conf/parameter_learning/mle.yaml", "r") as file:
        parameters_learning_config = yaml.safe_load(file)

    kwargs = {"log": False, "plot_prob": True}

    # Initialize the Bayesian Network
    bn = BayesianNetwork(
        dag=dag,
        data=data,
        parameters_learning_config=parameters_learning_config,
        **kwargs,
    )

    target_node = "reward"
    # Infer CPDs for node C given evidence for A and B
    evidence = {
        # "reward": torch.tensor([[1], [0]], device="cuda"),
        "action": torch.tensor([[2], [2], [3]], device="cuda"),
        "obs_0": torch.tensor([[10], [14], [10]], device="cuda"),
    }
    domain_values, pdf = bn.get_pdf(
        target_node,
        evidence,
        N_max=16,
    )
    print("PDF: ", pdf.shape)
    print("Domain: ", domain_values.shape)

    # bn.plot_prob(pdf, domain_values)
