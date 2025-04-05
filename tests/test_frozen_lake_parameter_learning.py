import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from cbn.utils import choose_probability_estimator


def frozen_lake_parameter_learning(
    estimator_name: str, n_samples_eval_for_feature: int = 100
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_pickle("../cbn/examples/frozen_lake.pkl")
    obs = torch.tensor(df[0].values, dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.tensor(df[1].values, dtype=torch.float32, device=device).unsqueeze(0)

    train_x = torch.cat([obs, action], dim=0).to("cuda")
    train_y = torch.tensor(df[2].values, dtype=torch.float32).to("cuda")

    with open(f"../cbn/conf/parameter_learning/{estimator_name}.yaml", "r") as file:
        parameter_learning_config = yaml.safe_load(file)

    estimator = choose_probability_estimator(estimator_name, parameter_learning_config)

    estimator.fit(train_y, train_x)

    obs_test = torch.linspace(obs.min(), obs.max(), n_samples_eval_for_feature)
    action_test = torch.linspace(action.min(), action.max(), n_samples_eval_for_feature)
    AA, BB = torch.meshgrid(obs_test, action_test, indexing="ij")
    test_x = torch.stack([AA.reshape(-1), BB.reshape(-1)], dim=1).to("cuda")
    " *********************************************************************** "

    n_queries = test_x.shape[0]
    test_x = test_x.unsqueeze(-1)

    pdfs, domains = estimator.get_prob(test_x)

    """for i in range(n_queries):
        pdf = pdfs[i].cpu().numpy()
        domain = domains[i].cpu().numpy()
        plt.figure()
        plt.title("Posterior Predictive Mean")
        plt.plot(domain, pdf)
        plt.xlabel("Domain")
        plt.ylabel("PDF")
        plt.show()"""

    domain_np = domains.cpu().numpy()
    pdfs_np = pdfs.cpu().numpy()

    max_values = np.zeros(n_queries)

    for i in range(n_queries):
        max_idx = np.argmax(pdfs_np[i])
        max_domain_value = domain_np[i][max_idx]
        max_values[i] = max_domain_value  # pdfs_np[max_idx]

    max_values = max_values.reshape(AA.shape)

    plt.figure()
    plt.title("Posterior Predictive Mean of reward(obs, action)")
    plt.contourf(AA.cpu().numpy(), BB.cpu().numpy(), max_values)
    plt.colorbar(label="Mean prediction")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Plot of PDF (All Queries)")

    all_x = []
    all_y = []
    all_z = []
    all_pdf = []

    test_x_np = test_x.cpu().numpy()

    for i in range(n_queries):
        # domain_np[i]: shape (n_values,) e.g. 100 points
        # pdfs_np[i]:   shape (n_values,)
        domain_i = domain_np[i]
        pdf_i = pdfs_np[i]

        # test_x_np[i, 0, 0] => the x-value of the i-th query
        # test_x_np[i, 1, 0] => the y-value of the i-th query
        x_i = test_x_np[i, 0, 0]
        y_i = test_x_np[i, 1, 0]

        # For each of the n_values points in the domain, replicate x_i, y_i
        for j in range(domain_i.shape[0]):
            all_x.append(x_i)
            all_y.append(y_i)
            all_z.append(domain_i[j])
            all_pdf.append(pdf_i[j])

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_pdf = np.array(all_pdf)

    # Create scatter with color = PDF
    sc = ax.scatter(all_x, all_y, all_z, c=all_pdf)  # Optionally add: cmap="viridis"

    fig.colorbar(sc, label="PDF")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Domain coordinate")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    e = input("Estimator: ")
    n = int(input("Number of samples used for evaluation: "))
    frozen_lake_parameter_learning(e, n)
