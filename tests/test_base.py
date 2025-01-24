import networkx as nx
import pandas as pd
import pytest
import torch
from cbn.base.bayesian_network import BayesianNetwork


@pytest.fixture
def data_and_dag():
    """Fixture to provide dataset and DAG for tests."""
    data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [0.5, 1.5, 2.5], "C": [0.2, 0.8, 1.2]}
    )
    dag = nx.DiGraph()
    dag.add_edges_from([("A", "C"), ("B", "C")])
    return data, dag


def test_bayesian_network_with_dataframe(data_and_dag):
    """Test Bayesian Network with a pandas DataFrame."""
    data, dag = data_and_dag
    bn = BayesianNetwork(dag=dag, data=data)

    evidence = {"A": 1.5, "B": 1.0}
    mean, covariance = bn.infer_cpds("C", evidence, uncertainty=10)

    assert mean is not None
    assert covariance is not None


def test_bayesian_network_with_numpy(data_and_dag):
    """Test Bayesian Network with a numpy array."""
    data, _ = data_and_dag
    dag = nx.DiGraph()
    dag.add_edges_from([("0", "2"), ("1", "2")])

    data_np = data.values
    bn = BayesianNetwork(dag=dag, data=data_np)

    evidence = {0: 1.5, 1: 1.0}
    mean, covariance = bn.infer_cpds("2", evidence, uncertainty=10)

    assert mean is not None
    assert covariance is not None


def test_bayesian_network_with_torch(data_and_dag):
    """Test Bayesian Network with a torch tensor."""
    data, _ = data_and_dag
    dag = nx.DiGraph()
    dag.add_edges_from([("0", "2"), ("1", "2")])

    data_torch = torch.tensor(data.values, dtype=torch.float32)
    bn = BayesianNetwork(dag=dag, data=data_torch)

    evidence = {0: 1.5, 1: 1.0}
    mean, covariance = bn.infer_cpds("2", evidence, uncertainty=10)

    assert mean is not None
    assert covariance is not None


def test_consistency_across_inputs(data_and_dag):
    """Test consistency of mean and covariance across DataFrame, numpy, and torch inputs."""
    data, dag = data_and_dag

    # DataFrame
    bn_df = BayesianNetwork(dag=dag, data=data)
    evidence_df = {"A": 1.5, "B": 1.0}
    mean_df, covariance_df = bn_df.infer_cpds("C", evidence_df, uncertainty=10)

    # Numpy
    dag_np = nx.DiGraph()
    dag_np.add_edges_from([("0", "2"), ("1", "2")])
    data_np = data.values
    bn_np = BayesianNetwork(dag=dag_np, data=data_np)
    evidence_np = {0: 1.5, 1: 1.0}
    mean_np, covariance_np = bn_np.infer_cpds("2", evidence_np, uncertainty=10)

    # Torch
    data_torch = torch.tensor(data.values, dtype=torch.float32)
    bn_torch = BayesianNetwork(dag=dag_np, data=data_torch)
    mean_torch, covariance_torch = bn_torch.infer_cpds("2", evidence_np, uncertainty=10)

    # Assert all results are equal
    assert torch.allclose(mean_df, mean_np, atol=1e-5)
    assert torch.allclose(mean_df, mean_torch, atol=1e-5)
    assert torch.allclose(covariance_df, covariance_np, atol=1e-5)
    assert torch.allclose(covariance_df, covariance_torch, atol=1e-5)
