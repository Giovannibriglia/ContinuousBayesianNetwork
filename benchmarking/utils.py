import time
from datetime import datetime
from typing import List, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
import pyAgrum as gum
import torch

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

    return lower_bound, upper_bound


def report_metrics_as_latex(
    task_name,
    y_pred,
    y_true,
    if_discrete,
    computation_time,
    file_path="report.txt",
    confidence_interval: float = 0.95,
):
    """
    Reports metrics for each task and saves them in LaTeX table format.

    Args:
        task_name (str): Name of the task (environment or experiment).
        y_pred (array-like): Predicted values.
        y_true (array-like): True values.
        if_discrete (bool): If discrete or continuous problem.
        computation_time (float): Time taken for inference in seconds.
        file_path (str): Path to save the report.
        confidence_interval (float): confidence interval.
    """
    # Compute metrics
    metrics = {}
    if not if_discrete:
        # Continuous case: Regression metrics
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["MSE"] = mean_squared_error(y_true, y_pred)
        metrics["R2"] = r2_score(y_true, y_pred)
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
        metrics[f"conf_int_{int(confidence_interval * 100)}"] = (
            compute_confidence_interval(y_true, y_pred, confidence_interval)
        )
    else:
        # Discrete case: Classification metrics

        if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
            # Binary classification
            metrics["Accuracy"] = accuracy_score(y_true, y_pred)

            unique_labels = np.unique(y_true)
            # Use the higher value in unique_labels as the positive label
            pos_label = max(unique_labels)

            metrics["Precision"] = precision_score(
                y_true, y_pred, pos_label=pos_label, zero_division=0
            )
            metrics["Recall"] = recall_score(y_true, y_pred, pos_label=pos_label)
            metrics["F1"] = f1_score(y_true, y_pred, pos_label=pos_label)
        else:
            # Multi-class classification
            metrics["Accuracy"] = accuracy_score(y_true, y_pred)
            metrics["F1 Micro"] = f1_score(y_true, y_pred, average="micro")
            metrics["F1 Macro"] = f1_score(y_true, y_pred, average="macro")

    # Add computation time
    metrics["Computation Time (s)"] = computation_time

    # Format as LaTeX
    latex_table = (
        "\\begin{{table}}[h!]\n\\centering\n\\begin{{tabular}}{{|c|c|}}\n\\hline\n"
    )
    latex_table += "Metric & Value \\\\\n\\hline\n"
    for metric, value in metrics.items():
        if isinstance(value, tuple):
            latex_table += f"{metric} & {[v.item() for v in value]} \\\\\n"
        else:
            latex_table += f"{metric} & {value:.4f} \\\\\n"
    latex_table += (
        "\\hline\n\\end{tabular}\n\\caption{" + task_name + "}\n\\end{table}\n"
    )

    # Save to file
    with open(file_path, "a") as f:
        f.write(f"\nTask: {task_name}\n")
        f.write(latex_table)
        f.write("\n\n")


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
    discrete_target: bool,
    batch_size: int = 64,
    task_name: str = "task",
    file_path: str = "",
):
    from cbn.base.bayesian_network import BayesianNetwork

    cbn = BayesianNetwork(dag, data)

    dict_values = {
        feat: torch.tensor(data[feat].values, device="cpu")
        for feat in data.columns
        if feat != target_node
    }

    true_values = data[target_node].values
    pred_values = np.zeros((len(data),))

    computation_time = time.time()
    for n in tqdm(range(0, len(data), batch_size), desc="benchmarking df..."):
        evidence = {
            feat: dict_values[feat][n : n + batch_size].unsqueeze(-1).to("cuda")
            for feat in data.columns
            if feat != target_node
        }

        """cpd, pdf, parameters_domain = cbn.get_cpd_and_pdf(
            target_node, evidence, normalize_pdf=True, N_max=64
        )"""

        inference_probabilities, domain_values = cbn.infer(
            target_node, evidence, plot_prob=False, N_max=batch_size
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

    computation_time = time.time() - computation_time

    report_metrics_as_latex(
        task_name,
        pred_values,
        true_values,
        discrete_target,
        computation_time,
        file_path=file_path + "/cbn",
    )


def benchmarking_df_pgmpy(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    target_node: str,
    discrete_target: bool,
    task_name: str = "task",
    file_path: str = "",
):
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
        file_path=file_path + "/pgmpy",
    )


def benchmarking_df_pyagrum(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    target_node: str,
    discrete_target: bool,
    task_name: str = "task",
    file_path: str = "",
):

    # Convert NetworkX DAG to a pyAgrum Bayesian Network
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
    bn = learner.learnParameters(bn)

    # Perform inference using pyAgrum's Variable Elimination
    ie = gum.LazyPropagation(bn)

    # Prepare data for inference
    true_values = data[target_node].values
    pred_values = np.zeros(len(data))

    computation_time = time.time()

    for n, row in tqdm(
        data.iterrows(), desc="benchmarking df pyagrum...", total=len(data)
    ):
        evidence = {
            feat: row[feat_idx]
            for feat_idx, feat in enumerate(data.columns)
            if feat != target_node
        }

        ie.setEvidence(evidence)
        ie.makeInference()

        pred_values[n] = ie.posterior(target_node).mode()

    computation_time = time.time() - computation_time

    # Report results
    report_metrics_as_latex(
        task_name,
        pred_values,
        true_values,
        discrete_target,
        computation_time,
        file_path=file_path + "/pyagrum",
    )
