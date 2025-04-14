from typing import List, Tuple

import gymnasium
import networkx as nx

import numpy as np
import pandas as pd
from gymnasium import envs
from tqdm import tqdm

from benchmarking.base import BaseBenchmarkingEnvs


class BenchmarkingGymnasiumEnvs(BaseBenchmarkingEnvs):
    def __init__(self, suite_name):
        super().__init__(suite_name)

    def get_envs_names(self) -> List[str]:
        all_envs = envs.registry.keys()
        env_ids = self.get_latest_envs(all_envs)

        env_ids.remove("GymV21Environment-v0")
        env_ids.remove("GymV26Environment-v0")

        return env_ids

    @staticmethod
    def get_latest_envs(env_keys):
        env_dict = {}
        for env in env_keys:
            env_name, version = env.split("-v")
            version = int(version)
            if env_name not in env_dict or env_dict[env_name] < version:
                env_dict[env_name] = version
        return [f"{name}-v{version}" for name, version in env_dict.items()]

    def collect_data(
        self,
        env_id: str,
        n_steps: int,
        seed: int,
    ) -> Tuple[pd.DataFrame, str, str, bool]:

        if "FrozenLake" in env_id:
            env = gymnasium.make(env_id, is_slippery=False)
        else:
            env = gymnasium.make(env_id)

        # Determine action space size
        if isinstance(env.action_space, gymnasium.spaces.Discrete):
            action_size = 1
        elif isinstance(env.action_space, gymnasium.spaces.Box):
            action_size = env.action_space.shape[0]
        elif isinstance(env.action_space, gymnasium.spaces.MultiDiscrete):
            action_size = len(env.action_space.nvec)  # Number of discrete dimensions
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

        # Check if both action and observation spaces are discrete
        if_discrete = isinstance(
            env.action_space, gymnasium.spaces.Discrete
        ) and isinstance(env.observation_space, gymnasium.spaces.Discrete)

        if env_id == "Blackjack-v1":
            if_discrete = True

        # Reset the environment
        observation, info = env.reset(seed=seed)

        # Ensure observations are flattened correctly
        if isinstance(observation, tuple):
            observation = np.concatenate(
                [np.array(obs).flatten() for obs in observation]
            )
        else:
            observation = np.array(observation).flatten()

        obs_size = observation.shape[0]

        # Initialize storage for collected data
        data_dict = {f"obs_{n}": [] for n in range(obs_size)}

        # Define multiple action features if action_size > 1
        if action_size == 1:
            data_dict["action"] = []
        else:
            for i in range(action_size):
                data_dict[f"action_{i}"] = []

        data_dict["reward"] = []

        # Run the simulation
        for _ in tqdm(range(n_steps), desc=f"{env_id} - collecting data"):
            # Perform a random action
            action = env.action_space.sample()

            # Take a step in the environment
            next_observation, reward, done, truncated, info = env.step(action)

            # Ensure observations are flattened
            if isinstance(next_observation, tuple):
                next_observation = np.concatenate(
                    [np.array(obs).flatten() for obs in next_observation]
                )
            else:
                next_observation = np.array(next_observation).flatten()

            action = np.atleast_1d(action)  # Ensure action is always an array

            # Store observations
            for i, obs in enumerate(observation):
                data_dict[f"obs_{i}"].append(obs)

            # Store multiple actions if action_size > 1
            if action_size == 1:
                data_dict["action"].append(action[0])  # Store single action
            else:
                for i in range(action_size):
                    data_dict[f"action_{i}"].append(
                        action[i]
                    )  # Store each action separately

            data_dict["reward"].append(reward)

            # Reset if done
            if done or truncated:
                observation, info = env.reset()
                if isinstance(observation, tuple):
                    observation = np.concatenate(
                        [np.array(obs).flatten() for obs in observation]
                    )
                else:
                    observation = np.array(observation).flatten()
            else:
                observation = next_observation

        # Close environment
        env.close()

        # Convert collected data to DataFrame
        df = pd.DataFrame(data_dict)

        target_feature = "reward"
        do_key = "action"

        return df, target_feature, do_key, if_discrete

    def define_dag(
        self,
        df: pd.DataFrame,
        target_feature: str,
        do_key: str = None,
    ) -> Tuple[nx.DiGraph, List, List]:

        if do_key is not None:
            observation_features = [
                s for s in df.columns if do_key not in s and s != target_feature
            ]
            intervention_features = [
                s for s in df.columns if do_key in s and s != target_feature
            ]
        else:
            observation_features = [s for s in df.columns]
            intervention_features = []

        dag = [(feature, target_feature) for feature in observation_features]

        for int_feature in intervention_features:
            dag.append((int_feature, target_feature))

        G = nx.DiGraph()
        G.add_edges_from(dag)

        return G, observation_features, intervention_features
