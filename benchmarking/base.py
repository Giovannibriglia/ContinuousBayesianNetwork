from abc import ABC, abstractmethod

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class BaseBenchmarkingEnvs(ABC):
    def __init__(self, suite_name: str, **kwargs):
        self.suite_name = suite_name

    @abstractmethod
    def get_envs_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def collect_data(
        self,
        env_id: str,
        n_steps: int,
        seed: int,
    ) -> Tuple[pd.DataFrame, str, str, bool]:
        raise NotImplementedError

    @abstractmethod
    def define_dag(
        self,
        df: pd.DataFrame,
        target_feature: str,
        do_key: str = None,
    ) -> Tuple[nx.DiGraph, List, List]:
        raise NotImplementedError


class BaseBayesianNetwork(ABC):
    def __init__(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        self.bn_library_name = None
        self.model = self._setup_model(
            parameter_learning_config,
            inference_config,
            dag,
            data,
            target_feature,
            **kwargs,
        )

    @abstractmethod
    def _setup_model(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def benchmarking_df(
        self, data: pd.DataFrame, batch_size: int = 128, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
