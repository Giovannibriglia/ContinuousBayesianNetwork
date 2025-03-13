import networkx as nx
import pandas as pd

from benchmarking.utils import benchmarking_df

if __name__ == "__main__":
    data = pd.read_pickle("../../frozen_lake.pkl")
    data.columns = ["obs_0", "action", "reward"]

    dag = nx.DiGraph()
    dag.add_edges_from([("obs_0", "reward"), ("action", "reward")])

    benchmarking_df(dag, data, target_node="reward", task_name="frozen_lake")
