import os
import time
from datetime import datetime
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarking.base import BaseBayesianNetwork, BaseBenchmarkingEnvs
from benchmarking.bayesian_networks import BN_LIBRARIES
from benchmarking.environment_suites import ENVIRONMENT_SUITES
from benchmarking.utils import get_bn_combinations


class Benchmarking:
    def __init__(self, envs_suites: List[str], bn_libraries: List[str], **kwargs):
        self.envs_suites = envs_suites
        self.bn_combinations = get_bn_combinations(bn_libraries)

        self.device = kwargs.get("device", "cpu")
        self.dir_saving = None

    @staticmethod
    def _generate_simulation_name(prefix: str = "test") -> str:
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

    @staticmethod
    def setup_bayesian_network(
        bn_lib: str,
        prob_config: Dict,
        infer_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ) -> BaseBayesianNetwork:

        return BN_LIBRARIES[bn_lib](
            prob_config, infer_config, dag, data, target_feature, **kwargs
        )

    @staticmethod
    def choose_base_env_suite(env_suite) -> BaseBenchmarkingEnvs:
        return ENVIRONMENT_SUITES[env_suite](env_suite)

    def run(
        self,
        n_seeds: int = 1,
        test_size: float = 0.2,
        n_steps: int = int(1e4),
        batch_size: int = 256,
    ):
        print("\n")
        self.dir_saving = "benchmarks/" + self._generate_simulation_name()
        os.makedirs(self.dir_saving, exist_ok=True)

        for env_suite in self.envs_suites:
            base_env_suite = self.choose_base_env_suite(env_suite)

            envs = base_env_suite.get_envs_names()

            for seed in range(n_seeds):
                for env_id in envs:
                    data, target_feature, do_key, is_discrete = (
                        base_env_suite.collect_data(env_id, n_steps, seed)
                    )

                    # data.drop_duplicates(inplace=True, ignore_index=True)

                    dag, observation_features, intervention_features = (
                        base_env_suite.define_dag(data, target_feature, do_key)
                    )

                    data_train, data_test = train_test_split(
                        data, test_size=test_size, random_state=seed
                    )
                    data_train = data_train.reset_index(drop=True)
                    data_test = data_test.reset_index(drop=True)

                    targets = data_test[target_feature]

                    for comb in self.bn_combinations:

                        bn_lib = comb["bn_library"]
                        prob_config = comb["prob_config"]
                        infer_config = comb["infer_config"]

                        prob_estimator = prob_config["estimator_name"]
                        infer_obj = infer_config["inference_obj"]

                        print(
                            f"Seed: {seed} | Env: {env_id} | Bayesian network from: {bn_lib} | Parameter Learning: {prob_estimator} | Inference: {infer_obj}"
                        )

                        simulation_info = {
                            "env_suite": env_suite,
                            "env_name": env_id,
                            "bn_lib": bn_lib,
                            "prob_estimator": prob_estimator,
                            "infer_obj": infer_obj,
                            "seed": seed,
                            "test_size": test_size,
                        }

                        training_time = time.time()

                        train_ok = False
                        try:
                            bn = self.setup_bayesian_network(
                                bn_lib,
                                prob_config,
                                infer_config,
                                dag,
                                data_train,
                                target_feature,
                            )
                            training_time = time.time() - training_time
                            train_ok = True
                        except Exception as e:
                            print(f"training problem: {e}")
                            simulation_info["training_problem"] = str(e)
                            training_time = -1

                        if train_ok:
                            try:
                                inference_time = time.time()
                                predictions = bn.benchmarking_df(
                                    data_test, batch_size=batch_size
                                )
                                inference_time = time.time() - inference_time
                            except Exception as e2:
                                print(f"inference problem: {e2}")
                                simulation_info["inference_problem"] = str(e2)
                                predictions = np.full((len(data_test),), np.nan)
                                inference_time = -1
                        else:
                            predictions = np.full((len(data_test),), np.nan)
                            inference_time = -1

                        simulation_info["training_time"] = training_time
                        simulation_info["inference_time"] = inference_time

                        self._store_metrics(
                            predictions, targets, simulation_info, is_discrete
                        )
                        print("\n")

    def _store_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        metrics_dict: Dict,
        is_discrete: bool,
        name: str = "results",
        confidence_interval: float = 0.95,
    ):

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

        # Create a mask to ignore NaNs in y_true or y_pred
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

        metrics_dict["predicted"] = f"{np.sum(mask).item()}/{len(y_pred)}"

        # Apply mask before calling mean_absolute_error
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if not np.isnan(y_pred).all():
            if not is_discrete:
                # Continuous (Regression) metrics
                mae_val = mean_absolute_error(y_true, y_pred)
                mse_val = mean_squared_error(y_true, y_pred)
                r2_val = r2_score(y_true, y_pred)
                mape_val = mean_absolute_percentage_error(y_true, y_pred)
                conf_int_val = self._compute_confidence_interval(
                    np.array(y_true), np.array(y_pred), confidence_interval
                )  # returns a tuple

                metrics_dict["MAE"] = mae_val
                metrics_dict["MSE"] = mse_val
                metrics_dict["R2"] = r2_val
                metrics_dict["MAPE"] = mape_val
                metrics_dict[f"low_conf_int_{int(confidence_interval * 100)}"] = (
                    conf_int_val[0]
                )
                metrics_dict[f"up_conf_int_{int(confidence_interval * 100)}"] = (
                    conf_int_val[1]
                )
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
                        precision_score(
                            y_true, y_pred, pos_label=pos_label, zero_division=0
                        )
                        * 100
                    )
                    # Check if pos_label actually appears in y_true
                    if pos_label in y_true:
                        metrics_dict["Recall"] = (
                            recall_score(
                                y_true, y_pred, pos_label=pos_label, zero_division=1
                            )
                            * 100
                        )
                    else:
                        metrics_dict["Recall"] = 0.0

                    metrics_dict["F1"] = (
                        f1_score(y_true, y_pred, pos_label=pos_label, zero_division=1)
                        * 100
                    )

                    metrics_dict[f"conf_int_{int(confidence_interval * 100)}"] = (
                        np.nan
                    )  # returns a tuple
                    metrics_dict[f"low_conf_int_{int(confidence_interval * 100)}"] = (
                        np.nan
                    )
                    metrics_dict[f"up_conf_int_{int(confidence_interval * 100)}"] = (
                        np.nan
                    )

                else:
                    # Multiclass classification
                    metrics_dict["Accuracy"] = accuracy_score(y_true, y_pred) * 100
                    metrics_dict["Precision"] = (
                        precision_score(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                        * 100
                    )
                    metrics_dict["Recall"] = (
                        recall_score(y_true, y_pred, average="macro", zero_division=0)
                        * 100
                    )
                    metrics_dict["F1"] = (
                        f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
                    )
                    metrics_dict[f"low_conf_int_{int(confidence_interval * 100)}"] = (
                        np.nan
                    )
                    metrics_dict[f"up_conf_int_{int(confidence_interval * 100)}"] = (
                        np.nan
                    )

        # ---- 2) Read or create the DataFrame from disk  ----
        if os.path.exists(f"{self.dir_saving}/{name}.xlsx"):
            df = pd.read_excel(f"{self.dir_saving}/{name}.xlsx")
        else:
            df = pd.DataFrame()

        # Create the row as a DataFrame (1 row)
        row_df = pd.DataFrame([metrics_dict])

        # ---- 4) Append new row to the DataFrame  ----
        df = pd.concat([df, row_df], ignore_index=True)

        # ---- 5) Write the updated DataFrame to Excel  ----
        df.to_excel(f"{self.dir_saving}/{name}.xlsx", index=False)

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
        with open(f"{self.dir_saving}/{name}.txt", "w", encoding="utf-8") as f:
            f.write("All tasks, aggregated metrics\n\n")
            f.write(latex_table)
            f.write("\n")

        """print(f"Updated Excel file: {xlsx_path}")
        print(f"Updated LaTeX table in: {txt_path}")"""

    @staticmethod
    def _compute_confidence_interval(y_true, y_pred, confidence: float = 0.95):
        from scipy.stats import t

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


def ask_list_from_user(prompt):
    print(f"Enter values for {prompt} (press Enter without typing anything to stop):")
    result = []
    while True:
        val = input(f"- {prompt} name: ").strip()
        if val == "":
            break
        result.append(val)
    return result


if __name__ == "__main__":
    envs_suites = [
        "cause_effect_pairs"
    ]  # ask_list_from_user("envs_suites")  # vmas, gymnasium "cause_effect_pairs"
    bn_libs = [
        "pyagrum",
        "my_bn",
        "pgmpy",
    ]  # ask_list_from_user("bn_libs")  # my_bn, pgmpy, pyagrum

    b = Benchmarking(envs_suites, bn_libs)

    n_seeds = 1  # int(input("Number of seeds: "))
    test_size = 0.2  # float(input("Test size as float: "))

    b.run(n_seeds, test_size)
