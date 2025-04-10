from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmarking.base import BaseBayesianNetwork


class PgmpyBN(BaseBayesianNetwork):
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

        self.bn_library_name = "pgmpy"

    def _setup_model(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        from pgmpy import estimators, inference
        from pgmpy.models import BayesianNetwork

        try:
            estimator_cls = getattr(
                estimators, parameter_learning_config["estimator_name"]
            )
        except AttributeError:
            raise ValueError(
                f"Estimator '{parameter_learning_config['estimator_name']}' not found in pgmpy.estimators"
            )

        inference_cls = getattr(inference, inference_config["inference_obj"])

        # Convert NetworkX DAG to pgmpy BayesianNetwork
        self.model = BayesianNetwork([tuple(edge) for edge in dag.edges])

        self.model.fit(data, estimator=estimator_cls)

        self.inference_cls = inference_cls
        self.target_feature = target_feature

        return self

    def benchmarking_df(
        self, data: pd.DataFrame, batch_size: int = 128, **kwargs
    ) -> np.ndarray:

        inference_obj = self.inference_cls(self.model)

        pred_values = np.zeros(len(data))

        for n, row in tqdm(
            data.iterrows(), desc="benchmarking df pgmpy...", total=len(data)
        ):
            evidence = {
                feat: row[feat_idx]
                for feat_idx, feat in enumerate(data.columns)
                if feat != self.target_feature
            }

            query_result = inference_obj.map_query(
                variables=[self.target_feature], evidence=evidence, show_progress=False
            )
            pred_values[n] = query_result[self.target_feature]

        return pred_values
