import networkx as nx
import pandas as pd
import torch

from cbn.base.bayesian_network import BayesianNetwork
from matplotlib import pyplot as plt


def plot_tensor(tensor, label):
    # Remove any singleton dimensions (shape [1, 11] -> [11])
    values = tensor.cpu().numpy().squeeze()
    # Convert to a Python list or NumPy array for plotting
    plt.plot(values.tolist(), marker="o", label=label)

    # Optional labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of Tensor Values")
    plt.legend(loc="best")


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
        # "reward": torch.tensor([[1]], device="cuda"),
        # "action": torch.tensor([2, 2], device="cuda"),
        # "obs_0": torch.tensor([14, 10], device="cuda"),
        "obs_0": torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 13.0, 14.0]],
            device="cuda:0",
        )
    }
    cpd, pdf, target_values = bn.get_cpd_and_pdf(
        target_node,
        evidence,
        points_to_evaluate=torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 13.0, 14.0],
            device="cuda:0",
        ),
        normalize_pdf=True,
    )
    print("Conditional Distribution:", cpd)
    print("PDF: ", pdf)
    # print("Target Values: ", target_values)
    plot_tensor(pdf, "1")
    print("*********************************************************************")
    evidence = {
        # "reward": torch.tensor([[1]], device="cuda"),
        # "action": torch.tensor([2, 2], device="cuda"),
        # "obs_0": torch.tensor([14, 10], device="cuda"),
    }
    cpd, pdf, target_values = bn.get_cpd_and_pdf(
        target_node,
        evidence,
        points_to_evaluate=torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 13.0, 14.0],
            device="cuda:0",
        ),
        normalize_pdf=True,
    )
    print("Conditional Distribution:", cpd)
    print("PDF: ", pdf)
    # print("Target Values: ", target_values)
    plot_tensor(pdf, "2")
    print("*********************************************************************")
    evidence = {
        # "reward": torch.tensor([[1]], device="cuda"),
        # "action": torch.tensor([2, 2], device="cuda"),
        # "obs_0": torch.tensor([14, 10], device="cuda"),
        "obs_0": torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 13.0, 14.0]],
            device="cuda:0",
        )
    }
    cpd, pdf, target_values = bn.get_cpd_and_pdf(
        target_node,
        evidence,
        normalize_pdf=True,
    )
    print("Conditional Distribution:", cpd)
    print("PDF: ", pdf)
    # print("Target Values: ", target_values)
    plot_tensor(pdf, "3")

    plt.show()
