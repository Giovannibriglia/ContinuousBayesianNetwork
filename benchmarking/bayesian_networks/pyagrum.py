from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import pyAgrum as gum
from pyAgrum.lib.discretizer import Discretizer
from tqdm import tqdm

from benchmarking.base import BaseBayesianNetwork


class PyAgrumBN(BaseBayesianNetwork):
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

        self.bn_library_name = "pyagrum"

    def _setup_model(
        self,
        parameter_learning_config: Dict,
        inference_config: Dict,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        target_feature: str,
        **kwargs,
    ):
        discretizer = Discretizer()
        self.bn = discretizer.discretizedTemplate(data)
        for arc in dag.edges():
            self.bn.addArc(arc[0], arc[1])
        learner = gum.BNLearner(data, self.bn)

        learner.useSmoothingPrior()
        learner.fitParameters(self.bn)

        self.target_feature = target_feature

    def benchmarking_df(
        self, data: pd.DataFrame, batch_size: int = 128, **kwargs
    ) -> np.ndarray:

        ie = gum.LazyPropagation(self.bn)
        ie.makeInference()

        pred_values = np.zeros(len(data))

        for n, row in tqdm(
            data.iterrows(), desc="benchmarking df pyagrum...", total=len(data)
        ):
            evidence = {
                feat: str(row.iloc[feat_idx].item())
                for feat_idx, feat in enumerate(data.columns)
                if feat != self.target_feature
            }
            ie.setEvidence(evidence)
            ie.makeInference()

            pred_values[n] = ie.posterior(self.target_feature).argmax()[0][0][
                self.target_feature
            ]

        return pred_values
