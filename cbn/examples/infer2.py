import networkx as nx
import numpy as np
import pandas as pd
import torch

from cbn.base.bayesian_network import BayesianNetwork

if __name__ == "__main__":
    # Number of samples
    n_samples = 1000000

    # Generating random values for A, B, and F
    np.random.seed(42)
    A = np.random.normal(100, 10, (n_samples,))
    B = np.random.normal(200, 20, (n_samples,))
    C = np.random.uniform(200, 300, (n_samples,))
    F = np.random.normal(500, 500, (n_samples,))

    # Computations
    D = A + B * C
    E = D * F

    # Creating DataFrame
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "F": F, "E": E})

    dag = nx.DiGraph()
    dag.add_edges_from([("A", "D"), ("B", "D"), ("C", "D"), ("D", "E"), ("F", "E")])

    bn = BayesianNetwork(dag=dag, data=df)

    target_node = "E"

    evidence = {
        # "reward": torch.tensor([[0], [1\]], device="cuda"),
        "A": torch.tensor(A[:10].reshape(-1, 1), device="cuda"),
        # "A": torch.tensor([[A[0]]], device="cuda"),
        "B": torch.tensor(B[:10].reshape(-1, 1), device="cuda"),
        # "B": torch.tensor([[B[0]]], device="cuda"),
    }
    prob, domain_values = bn.infer(
        target_node, evidence, plot_prob=True, uncertainty=1, N_max=128
    )
    print("Probabilities: ", prob)
    print("Domains: ", domain_values)
    print("Shapes: ", prob.shape, domain_values.shape)

    bn.plot_prob(prob, domain_values, title="Inference Output")
