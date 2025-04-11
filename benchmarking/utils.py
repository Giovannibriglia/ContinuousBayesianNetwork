from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def get_bn_combinations(bn_libraries: List[str]) -> List[Dict]:
    combinations = []

    bn_libraries = set(bn_libraries)

    for bn_lib in bn_libraries:
        if bn_lib == "pgmpy":
            c = get_pgmpy_combinations()
        elif bn_lib == "pyagrum":
            c = get_pyagrum_combinations()
        elif bn_lib == "my_bn":
            c = get_my_bn_combinations()
        else:
            raise ValueError(f"Unknown bn_lib: {bn_lib}")

        combinations += c

    return combinations


def get_pgmpy_combinations() -> List[Dict]:
    prob_estimators = [
        "MaximumLikelihoodEstimator",
        "BayesianEstimator",
        "ExpectationMaximization",
    ]
    inference_objs = [
        "VariableElimination",
        "ApproxInference",
    ]

    combs = []

    for prob in prob_estimators:
        for inf in inference_objs:
            comb = {
                "bn_library": "pgmpy",
                "prob_config": {"estimator_name": prob},
                "infer_config": {"inference_obj": inf},
            }
            combs.append(comb)

    return combs


def get_pyagrum_combinations() -> List[Dict]:

    prob_estimators = [
        "useSmoothingPrior",
    ]
    inference_objs = [
        "LazyPropagation",
    ]

    combs = []

    for prob in prob_estimators:
        for inf in inference_objs:
            comb = {
                "bn_library": "pyagrum",
                "prob_config": {"estimator_name": prob},
                "infer_config": {"inference_obj": inf},
            }
            combs.append(comb)

    return combs


def get_my_bn_combinations() -> List[Dict]:
    from cbn.inference import INFERENCE_OBJS
    from cbn.parameter_learning import ESTIMATORS

    prob_estimators = ESTIMATORS.keys()
    inference_objs = INFERENCE_OBJS.keys()

    combs = []

    for prob in prob_estimators:
        with open(f"../cbn/conf/parameter_learning/{prob}.yaml", "r") as file:
            prob_config = yaml.safe_load(file)

        for inf in inference_objs:
            with open(f"../cbn/conf/inference/{inf}.yaml", "r") as file:
                infer_config = yaml.safe_load(file)

            d = {
                "bn_library": "my_bn",
                "prob_config": prob_config,
                "infer_config": infer_config,
            }

            combs.append(d)

    return combs


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
