from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from cbn.base.bayesian_network import BayesianNetwork

from benchmarking.base import BaseBayesianNetwork


class MyCBN(BaseBayesianNetwork):
    def __init__(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        super().__init__(
            parameter_learning_config,
            inference_config,
            dag,
            data,
            target_feature,
            **kwargs,
        )

        self.bn_library_name = "my_bn"

    def _setup_model(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        self.target_feature = target_feature

        return BayesianNetwork(
            dag=dag,
            data=data,
            parameters_learning_config=parameter_learning_config,
            inference_config=inference_config,
            **kwargs,
        )

    def benchmarking_df(
        self, data: pd.DataFrame, batch_size: int = 128, **kwargs
    ) -> np.ndarray:
        return self.model.benchmarking_df(
            data, self.target_feature, batch_size, **kwargs
        )
