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
        from pgmpy.models import DiscreteBayesianNetwork

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
        model = DiscreteBayesianNetwork(dag)

        estimator = estimator_cls(model=model, data=data)
        # Estimate all the CPDs for `new_model`
        all_cpds = estimator.get_parameters()

        # Add the estimated CPDs to the model.
        model.add_cpds(*all_cpds)
        # model.check_model()
        self.inference_obj = inference_cls(model)

        self.target_feature = target_feature

        return self

    def benchmarking_df(
        self, data: pd.DataFrame, batch_size: int = 128, **kwargs
    ) -> np.ndarray:

        import logging

        logging.getLogger("pgmpy").setLevel(logging.ERROR)

        pred_values = np.zeros(len(data))

        for n, row in tqdm(
            data.iterrows(), desc="benchmarking df pgmpy...", total=len(data)
        ):
            evidence = {
                feat: row.iloc[feat_idx].item()
                for feat_idx, feat in enumerate(data.columns)
                if feat != self.target_feature
            }
            try:
                query_result = self.inference_obj.map_query(
                    variables=[self.target_feature],
                    evidence=evidence,
                    show_progress=False,
                )
                pred_values[n] = query_result[self.target_feature]
            except Exception:
                pred_values[n] = np.nan

        return pred_values
