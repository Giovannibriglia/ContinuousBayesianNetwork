import csv
import inspect
import os
from typing import List, Tuple

import networkx as nx

import pandas as pd

from benchmarking.base import BaseBenchmarkingEnvs


class BenchmarkingCauseEffectPairs(BaseBenchmarkingEnvs):
    def __init__(self, suite_name):
        super().__init__(suite_name)

        # Gets the file where the class is defined
        file_path = inspect.getfile(BenchmarkingCauseEffectPairs)
        # Returns the folder containing the file

        self.folder_data = os.path.dirname(os.path.abspath(file_path)) + "/pairs"

    def get_envs_names(self) -> List[str]:
        # List all txt files that do NOT contain '_des'
        valid_files = [
            f.replace(".txt", "")
            for f in os.listdir(self.folder_data)
            if f.endswith(".txt") and "_des" not in f
        ]
        # valid_files = ["pair0052"]
        return sorted(valid_files)

    def collect_data(self, env_id: str, n_steps: int, seed: int):
        file_path = f"{self.folder_data}/{env_id}.txt"

        # 1. Read a small sample for sniffing
        with open(file_path, "r") as f:
            sample = f.read(2048)  # read 2KB, not entire file

        # 2. Try auto-sniff
        try:
            dialect = csv.Sniffer().sniff(sample)
            sniffed_sep = dialect.delimiter
        except csv.Error:
            sniffed_sep = None

        df = None

        # 3. Attempt read using sniffed_sep
        fallback_needed = False
        if sniffed_sep:
            df = pd.read_csv(file_path, sep=sniffed_sep, header=None)
            # If we only get 1–2 columns but the file looks multi-column, fallback
            if df.shape[1] < 8:  # or pick your own threshold
                fallback_needed = True
            # Also fallback if we see a bunch of NaNs
            elif df.isnull().any().any():
                fallback_needed = True
        else:
            fallback_needed = True

        # 4. Fallback if needed
        if fallback_needed:
            df = pd.read_csv(file_path, sep=r"[\t\s]+", engine="python", header=None)
            df.dropna(how="all", inplace=True)

        # 5. Name columns dynamically
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

        target_feature = df.columns[-1]
        do_key = None
        if_discrete = False

        return df, target_feature, do_key, if_discrete

    def define_dag(
        self,
        df: pd.DataFrame,
        target_feature: str,
        do_key: str = None,
    ) -> Tuple[nx.DiGraph, List, List]:

        observation_features = [s for s in df.columns if s != target_feature]

        dag = []
        for col in observation_features:
            dag.append((col, target_feature))

        intervention_features = []

        G = nx.DiGraph()
        G.add_edges_from(dag)

        return G, observation_features, intervention_features
