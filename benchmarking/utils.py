import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
import pyAgrum as gum
import torch
from pyAgrum.lib.discretizer import Discretizer

from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tqdm import tqdm


def collect_data_gymnasium(env_name, n_steps):
    # Initialize the environment
    if "FrozenLake" in env_name:
        env = gym.make(env_name, is_slippery=False)
    else:
        env = gym.make(env_name)

    # Determine action space size
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = 1
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_size = len(env.action_space.nvec)  # Number of discrete dimensions
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    # Check if both action and observation spaces are discrete
    if_discrete = isinstance(env.action_space, gym.spaces.Discrete) and isinstance(
        env.observation_space, gym.spaces.Discrete
    )

    if env_name == "Blackjack-v1":
        if_discrete = True

    # Reset the environment
    observation, info = env.reset()

    # Ensure observations are flattened correctly
    if isinstance(observation, tuple):
        observation = np.concatenate([np.array(obs).flatten() for obs in observation])
    else:
        observation = np.array(observation).flatten()

    obs_size = observation.shape[0]

    # Initialize storage for collected data
    data_dict = {f"obs_{n}": [] for n in range(obs_size)}

    # Define multiple action features if action_size > 1
    if action_size == 1:
        data_dict["action"] = []
    else:
        for i in range(action_size):
            data_dict[f"action_{i}"] = []

    data_dict["reward"] = []

    # Run the simulation
    for _ in tqdm(range(n_steps), desc=f"{env_name} - collecting data"):
        # Perform a random action
        action = env.action_space.sample()

        # Take a step in the environment
        next_observation, reward, done, truncated, info = env.step(action)

        # Ensure observations are flattened
        if isinstance(next_observation, tuple):
            next_observation = np.concatenate(
                [np.array(obs).flatten() for obs in next_observation]
            )
        else:
            next_observation = np.array(next_observation).flatten()

        action = np.atleast_1d(action)  # Ensure action is always an array

        # Store observations
        for i, obs in enumerate(observation):
            data_dict[f"obs_{i}"].append(obs)

        # Store multiple actions if action_size > 1
        if action_size == 1:
            data_dict["action"].append(action[0])  # Store single action
        else:
            for i in range(action_size):
                data_dict[f"action_{i}"].append(
                    action[i]
                )  # Store each action separately

        data_dict["reward"].append(reward)

        # Reset if done
        if done or truncated:
            observation, info = env.reset()
            if isinstance(observation, tuple):
                observation = np.concatenate(
                    [np.array(obs).flatten() for obs in observation]
                )
            else:
                observation = np.array(observation).flatten()
        else:
            observation = next_observation

    # Close environment
    env.close()

    # Convert collected data to DataFrame
    df = pd.DataFrame(data_dict)

    return df, if_discrete


def get_latest_envs(env_keys):
    env_dict = {}
    for env in env_keys:
        env_name, version = env.split("-v")
        version = int(version)
        if env_name not in env_dict or env_dict[env_name] < version:
            env_dict[env_name] = version
    return [f"{name}-v{version}" for name, version in env_dict.items()]


def discretize_dataframe(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Discretize a DataFrame with float values by obtaining N values for each column.

    Parameters:
        df (pd.DataFrame): The input DataFrame to discretize.
        N (int): The number of bins to discretize each column into.

    Returns:
        pd.DataFrame: A DataFrame with discretized float columns.
    """
    discretized_df = pd.DataFrame()

    for column in df.columns:
        unique_values = df[column].nunique()

        if unique_values < N:
            # Keep column as is if unique values are less than N
            discretized_df[column] = df[column]
        else:
            # Apply discretization for float values
            bins = np.linspace(df[column].min(), df[column].max(), N + 1)
            # Replace values with the bin midpoints
            discretized_df[column] = pd.cut(
                df[column], bins=bins, labels=False, include_lowest=True
            )
            # Optional: Replace bin indices with bin midpoints
            midpoints = (bins[:-1] + bins[1:]) / 2
            discretized_df[column] = discretized_df[column].map(
                lambda idx: midpoints[idx] if not pd.isna(idx) else np.nan
            )

    return discretized_df


def compute_confidence_interval(y_true, y_pred, confidence: float = 0.95):
    # Ensure no division by zero
    epsilon = 1e-12  # Small constant to avoid division by zero
    y_true_safe = np.where(y_true == 0, epsilon, y_true)

    # Calculate MAPE errors
    errors = np.abs((y_true - y_pred) / y_true_safe) * 10

    # Calculate the mean and standard error of the errors
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1) / np.sqrt(len(errors))

    dof = len(errors) - 1  # Degrees of freedom
    t_critical = t.ppf((1 + confidence) / 2, dof)  # Two-tailed t critical value

    # Margin of error
    margin_of_error = t_critical * std_error

    # Confidence interval
    lower_bound = mean_error - margin_of_error
    upper_bound = mean_error + margin_of_error

    return lower_bound.item(), upper_bound.item()


def report_metrics_as_latex(
    task_name,
    y_pred,
    y_true,
    if_discrete,
    computation_time,
    training_time,
    xlsx_path="report.xlsx",
    txt_path="report.txt",
    confidence_interval: float = 0.95,
):
    """
    Compute metrics and store them in a single table (pandas DataFrame) across multiple tasks.
    The function:
        1) Reads/creates a DataFrame from `xlsx_path`.
        2) Computes metrics for this task.
        3) Appends a new row for this task.
        4) Saves the updated DataFrame back to Excel.
        5) Converts the entire DataFrame into a single LaTeX table and writes it to `txt_path`.

    Args:
        task_name (str): Name of the task (environment or experiment).
        y_pred (array-like): Predicted values.
        y_true (array-like): True values.
        if_discrete (bool): If discrete or continuous problem.
        computation_time (float): Time taken for inference in seconds.
        xlsx_path (str): Path to the Excel file storing the aggregated table.
        txt_path (str): Path to a .txt file storing the LaTeX version of the entire table.
        confidence_interval (float): Confidence interval to use in continuous tasks.
    """

    # ---- 1) Compute metrics for the current task  ----
    metrics_dict = {"Task": task_name}  # We'll store all metrics as columns
    if not if_discrete:
        # Continuous (Regression) metrics
        mae_val = mean_absolute_error(y_true, y_pred)
        mse_val = mean_squared_error(y_true, y_pred)
        r2_val = r2_score(y_true, y_pred)
        mape_val = mean_absolute_percentage_error(y_true, y_pred)
        conf_int_val = compute_confidence_interval(
            np.array(y_true), np.array(y_pred), confidence_interval
        )  # returns a tuple

        metrics_dict["MAE"] = mae_val
        metrics_dict["MSE"] = mse_val
        metrics_dict["R2"] = r2_val
        metrics_dict["MAPE"] = mape_val
        metrics_dict[f"conf_int_{int(confidence_interval * 100)}"] = conf_int_val
    else:
        # Discrete (Classification) metrics
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        unique_labels = np.unique(y_true)

        # If it's binary classification with matching labels
        if len(unique_labels) == 2 and set(unique_labels).issubset(
            set(np.unique(y_pred))
        ):
            # The higher label is considered the positive class
            pos_label = max(unique_labels)
            metrics_dict["Accuracy"] = accuracy_score(y_true, y_pred) * 100
            metrics_dict["Precision"] = (
                precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                * 100
            )
            # Check if pos_label actually appears in y_true
            if pos_label in y_true:
                metrics_dict["Recall"] = (
                    recall_score(y_true, y_pred, pos_label=pos_label, zero_division=1)
                    * 100
                )
            else:
                metrics_dict["Recall"] = 0.0

            metrics_dict["F1"] = (
                f1_score(y_true, y_pred, pos_label=pos_label, zero_division=1) * 100
            )

            metrics_dict[f"conf_int_{int(confidence_interval * 100)}"] = (
                np.nan
            )  # returns a tuple
        else:
            # Multiclass classification
            metrics_dict["Accuracy"] = accuracy_score(y_true, y_pred) * 100
            metrics_dict["Precision"] = (
                precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )
            metrics_dict["Recall"] = (
                recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )
            metrics_dict["F1"] = (
                f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )
            metrics_dict[f"conf_int_{int(confidence_interval * 100)}"] = (
                np.nan
            )  # returns a tuple

    metrics_dict["Training Time (s)"] = training_time
    # Add computation time
    metrics_dict["Computation Time (s)"] = computation_time

    # ---- 2) Read or create the DataFrame from disk  ----
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        df = pd.DataFrame()

    # ---- 3) Merge columns: union of existing columns with new metrics  ----
    # We do this so that even if the new row has new metric columns, we incorporate them.
    all_columns = set(df.columns).union(metrics_dict.keys())
    # Force a sorted list of columns (optional) or just keep them unsorted
    all_columns = sorted(all_columns)
    # Remove the three desired columns from the original list
    all_columns.remove("Task")
    all_columns.remove(f"conf_int_{int(confidence_interval * 100)}")
    all_columns.remove("Computation Time (s)")

    # Reconstruct the column order
    new_column_order = (
        ["Task"]
        + all_columns
        + [f"conf_int_{int(confidence_interval * 100)}", "Computation Time (s)"]
    )

    # Reindex the existing df to have the union of columns
    df = df.reindex(columns=new_column_order)

    # Create the row as a DataFrame (1 row)
    row_df = pd.DataFrame([metrics_dict], columns=new_column_order)

    # ---- 4) Append new row to the DataFrame  ----
    df = pd.concat([df, row_df], ignore_index=True)

    # ---- 5) Write the updated DataFrame to Excel  ----
    df.to_excel(xlsx_path, index=False)

    # ---- 6) Convert the entire DataFrame to LaTeX  ----
    latex_table = df.to_latex(index=False, na_rep="NaN", float_format="%.2e")

    # You can also generate a more "raw" LaTeX (with tabular environment, etc.) if you prefer:
    # latex_table = df.style.format(precision=4).to_latex()  # for a styled version
    #
    # or do something more manual like:
    #
    # latex_table = "\\begin{table}[h!]\n\\centering\n"
    # latex_table += df.to_latex(index=False, na_rep="NaN", float_format="%.4f")
    # latex_table += "\\caption{All tasks}\n\\end{table}\n"

    # ---- 7) Write (overwrite or append) the LaTeX table to the .txt file  ----
    # Typically, you'd overwrite to keep it up-to-date with the entire table:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("All tasks, aggregated metrics\n\n")
        f.write(latex_table)
        f.write("\n")

    """print(f"Updated Excel file: {xlsx_path}")
    print(f"Updated LaTeX table in: {txt_path}")"""


def define_dag(df: pd.DataFrame, target_feature: str) -> Tuple[nx.DiGraph, List, List]:

    observation_features = [s for s in df.columns if "obs" in s]
    intervention_features = [s for s in df.columns if "action" in s]

    dag = [(feature, target_feature) for feature in observation_features]
    for int_feature in intervention_features:
        dag.append((int_feature, target_feature))

    G = nx.DiGraph()
    G.add_edges_from(dag)

    return G, observation_features, intervention_features


def generate_simulation_name(prefix: str = "analysis") -> str:
    """
    Generates a simulation name based on the current date and time.

    Args:
        prefix (str): Prefix for the simulation name (default is "Simulation").

    Returns:
        str: A string containing the prefix and a timestamp.
    """
    # Get the current date and time
    now = datetime.now()
    # Format the date and time into a readable string
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Combine the prefix and timestamp
    simulation_name = f"{prefix}_{timestamp}"
    return simulation_name


def benchmarking_df(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    target_node: str,
    parameters_learning_config: Dict,
    inference_config: Dict,
    discrete_target: bool,
    batch_size: int = 64,
    task_name: str = "task",
    file_path: str = "",
    test_size: float = 0.2,
) -> np.ndarray:
    from cbn.base.bayesian_network import BayesianNetwork

    if test_size > 0:
        data_train = data[: int(len(data) * (1 - test_size))]
        data_test = data[int(len(data) * (1 - test_size)) :]
    else:
        data_train = data
        data_test = data

    kwargs = {"log": False, "plot_prob": False}

    training_time = time.time()
    # Initialize the Bayesian Network
    cbn = BayesianNetwork(
        dag=dag,
        data=data_train,
        parameters_learning_config=parameters_learning_config,
        inference_config=inference_config,
        **kwargs,
    )
    training_time = time.time() - training_time

    dict_values = {
        feat: torch.tensor(data_test[feat].values, device="cpu")
        for feat in data_test.columns
        if feat != target_node
    }

    true_values = data_test[target_node].values
    pred_values = np.zeros((len(data_test),))

    computation_time = time.time()

    progress_bar = tqdm(total=len(data_test), desc="benchmarking df cbn...")
    for n in range(0, len(data_test), batch_size):
        evidence = {
            feat: dict_values[feat][n : n + batch_size].unsqueeze(-1).to("cuda")
            for feat in data_test.columns
            if feat != target_node
        }

        inference_probabilities, domain_values = cbn.infer(
            target_node,
            evidence,
            plot_prob=False,
            N_max=16,
        )

        # Get indices of max probabilities along dimension 1 (columns)
        max_probabilities_indices = torch.argmax(
            inference_probabilities, dim=1, keepdim=True
        )  # Shape [batch_size, 1]
        # Gather the corresponding domain values
        pred_values_batch = (
            torch.gather(domain_values, dim=1, index=max_probabilities_indices)
            .squeeze(1)
            .cpu()
            .numpy()
        )  # Shape [batch_size]

        pred_values[n : n + batch_size] = pred_values_batch

        batch_end = min(n + batch_size, len(data_test))
        progress_bar.update(batch_end - n)  # Update by actual batch size

    progress_bar.close()

    computation_time = time.time() - computation_time

    report_metrics_as_latex(
        task_name,
        pred_values,
        true_values,
        discrete_target,
        computation_time,
        training_time,
        xlsx_path=f"{file_path}/cbn.xlsx",
        txt_path=f"{file_path}/cbn.txt",
    )

    return pred_values


def benchmarking_df_pgmpy(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    target_node: str,
    discrete_target: bool,
    task_name: str = "task",
    file_path: str = "",
) -> np.ndarray:
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianNetwork

    # Convert NetworkX DAG to pgmpy BayesianNetwork
    model = BayesianNetwork([tuple(edge) for edge in dag.edges])

    # Learn CPDs using Maximum Likelihood Estimation
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Perform inference using Variable Elimination
    inference = VariableElimination(model)

    # Prepare data for inference
    true_values = data[target_node].values
    pred_values = np.zeros(len(data))

    computation_time = time.time()

    for n, row in tqdm(
        data.iterrows(), desc="benchmarking df pgmpy...", total=len(data)
    ):
        evidence = {
            feat: row[feat_idx]
            for feat_idx, feat in enumerate(data.columns)
            if feat != target_node
        }

        query_result = inference.map_query(
            variables=[target_node], evidence=evidence, show_progress=False
        )
        pred_values[n] = query_result[target_node]

    computation_time = time.time() - computation_time

    # Report results
    report_metrics_as_latex(
        task_name,
        pred_values,
        true_values,
        discrete_target,
        computation_time,
        xlsx_path=f"{file_path}/pgmpy.xlsx",
        txt_path=f"{file_path}/pgmpy.txt",
    )

    return pred_values


def benchmarking_df_pyagrum(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    target_node: str,
    discrete_target: bool,
    task_name: str = "task",
    file_path: str = "",
) -> np.ndarray:
    """# Convert NetworkX DAG to a pyAgrum Bayesian Network
    bn = gum.BayesNet(task_name)

    # Add nodes (discrete variables assumed)
    for node in data.columns:
        bn.add(gum.LabelizedVariable(node, node, len(data[node].unique())))

    # Add edges from DAG
    for parent, child in dag.edges:
        bn.addArc(parent, child)

    # Learn CPDs from data
    learner = gum.BNLearner(data)
    learner.useK2()
    bn = learner.learnParameters(bn)"""

    discretizer = Discretizer()
    bn = discretizer.discretizedTemplate(data)
    for arc in dag.edges():
        bn.addArc(arc[0], arc[1])
    learner = gum.BNLearner(data, bn)

    learner.useSmoothingPrior()
    learner.fitParameters(bn)
    ie = gum.LazyPropagation(bn)
    ie.makeInference()

    # Prepare data for inference
    true_values = data[target_node].values
    pred_values = np.zeros(len(data))

    computation_time = time.time()

    for n, row in tqdm(
        data.iterrows(), desc="benchmarking df pyagrum...", total=len(data)
    ):
        """evidence = {
            feat: [
                1.0 if value == row[feat_idx].item() else 0.0
                for value in sorted(data[feat].unique())
            ]
            for feat_idx, feat in enumerate(data.columns)
            if feat != target_node
        }"""
        evidence = {
            feat: str(row.iloc[feat_idx].item())
            for feat_idx, feat in enumerate(data.columns)
            if feat != target_node
        }
        ie.setEvidence(evidence)
        ie.makeInference()

        pred_values[n] = ie.posterior(target_node).argmax()[0][0][target_node]

    computation_time = time.time() - computation_time

    # Report results
    report_metrics_as_latex(
        task_name,
        pred_values,
        true_values,
        discrete_target,
        computation_time,
        xlsx_path=f"{file_path}/pyagrum.xlsx",
        txt_path=f"{file_path}/pyagrum.txt",
    )

    return pred_values
