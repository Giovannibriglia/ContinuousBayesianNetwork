import networkx as nx
import pandas as pd
import torch

from cbn.base.bayesian_network import BayesianNetwork

if __name__ == "__main__":
    # Create a dataset

    data = pd.read_pickle("../../frozen_lake.pkl")
    data.columns = ["obs_0", "action", "reward"]

    dag = nx.DiGraph()
    dag.add_edges_from([("obs_0", "reward"), ("action", "reward")])

    # Initialize the Bayesian Network
    bn = BayesianNetwork(dag=dag, data=data)

    # Print the structure
    # bn.get_bn_structure()

    target_node = "obs_0"
    # Infer CPDs for node C given evidence for A and B
    evidence = {
        "reward": torch.tensor([[1], [1], [0]], device="cuda"),
        "action": torch.tensor([[2], [2], [1]], device="cuda"),
        # "obs_0": torch.tensor([[14], [14], [3]], device="cuda"),
    }
    prob, domain = bn.infer(target_node, evidence, {})
    print("Probabilities: ", prob)
    print("Domains values: ", domain)
    print(prob.shape, domain.shape)
