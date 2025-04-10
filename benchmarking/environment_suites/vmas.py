from typing import List, Tuple

import pandas as pd
import torch
import vmas
from tqdm import tqdm
from vmas import make_env
from vmas.simulator.heuristic_policy import RandomPolicy

from benchmarking.base import BaseBenchmarkingEnvs


class BenchmarkingVmasEnvs(BaseBenchmarkingEnvs):
    def __init__(self, suite_name):
        super().__init__(suite_name)

        self.policy = RandomPolicy
        self.dict_actions = True
        self.dict_space = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_envs_names(self) -> List[str]:
        return vmas.scenarios

    def _select_actions(self, obs: torch.Tensor):
        actions = {} if self.dict_actions else []
        for n_agent, agent in enumerate(self.env.agents):
            if self.dict_space:
                actions_agent = self.policy.compute_action(
                    obs[agent.name], agent.u_range
                )
            else:
                actions_agent = self.policy.compute_action(obs[n_agent], agent.u_range)

            actions_agent.to(self.device)

            if self.dict_actions:
                actions[agent.name] = actions_agent
            else:
                actions.append(actions_agent)

        return actions

    def collect_data(
        self,
        env_id: str,
        n_steps: int,
        seed: int,
    ) -> Tuple[pd.DataFrame, str, str, bool]:

        env = make_env(
            scenario=env_id, device=self.device, dict_spaces=self.dict_space, num_envs=1
        )

        obs = env.reset(seed=seed)

        data_dict = {f"obs_{i}": [] for i in range(obs.shape[1])}
        data_dict["action"] = []
        data_dict["reward"] = []

        for _ in tqdm(range(n_steps), desc=f"{env_id} - collecting data"):
            actions = self._select_actions(obs)
            new_obs, rews, dones, info = env.step(actions)

            for obs_i in obs:
                data_dict["obs_" + str(obs_i)].append(obs_i.cpu().numpy())

            data_dict["action"].append(actions.cpu().numpy())
            data_dict["reward"].append(rews.cpu().numpy())

            obs = new_obs

        # Close environment
        env.close()

        # Convert collected data to DataFrame
        df = pd.DataFrame(data_dict)

        target_feature = "reward"
        do_key = "action"
        if_discrete = False

        return df, target_feature, do_key, if_discrete
