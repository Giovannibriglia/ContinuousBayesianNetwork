import os

import yaml
from gymnasium import envs

from benchmarking.utils import (
    benchmarking_df,
    # benchmarking_df_pyagrum,
    collect_data_gymnasium,
    define_dag,
    generate_simulation_name,
    get_latest_envs,
)


if __name__ == "__main__":
    """
    pip install "gymnasium[all]"
    pip install pgmpy
    pip install pyAgrum
    """

    target_feature = "reward"

    dir_saving = generate_simulation_name()
    os.makedirs(dir_saving, exist_ok=True)

    all_envs = envs.registry.keys()
    env_ids = get_latest_envs(all_envs)
    env_ids.remove("GymV21Environment-v0")
    env_ids.remove("GymV26Environment-v0")
    env_ids.remove("Ant-v5")
    env_ids.remove("Humanoid-v5")
    env_ids = ["HumanoidStandup-v5"]
    n_steps = 100

    for env_index, env_id in enumerate(env_ids):
        if (
            "phys2d" in env_id
            or "CarRacing" in env_id
            or "tabular" in env_id
            or "MountainCar" in env_id
        ):
            continue

        print(env_id)
        data, if_discrete = collect_data_gymnasium(env_id, n_steps)

        data.drop_duplicates(inplace=True, ignore_index=True)

        dag, observation_features, intervention_features = define_dag(
            data, target_feature
        )

        # Load the YAML file
        with open("../cbn/conf/parameter_learning/gp_gpytorch.yaml", "r") as file:
            parameters_learning_config = yaml.safe_load(file)

        # Load the YAML file
        with open("../cbn/conf/inference/exact.yaml", "r") as file:
            inference_config = yaml.safe_load(file)

        cbn_predictions = benchmarking_df(
            dag,
            data,
            target_feature,
            if_discrete,
            batch_size=512,
            task_name=env_id,
            file_path=f"{dir_saving}",
        )
        # data["cbn"] = cbn_predictions

        """try:
            pgmpy_predictions = benchmarking_df_pgmpy(
                dag,
                data,
                target_feature,
                if_discrete,
                task_name=env_id,
                file_path=f"{dir_saving}",
            )
            data["pgmpy"] = pgmpy_predictions
        except Exception as e:
            print("PGMPY: ", e)
            # print(f"{env_id} cannot be solved with pgmpy MLE and Variable Elimination")"""

        """try:
            pyagrum_predictions = benchmarking_df_pyagrum(
                dag,
                data,
                target_feature,
                if_discrete,
                task_name=env_id,
                file_path=f"{dir_saving}",
            )
            data["pyagrum"] = pyagrum_predictions
        except Exception as e:
            print("PyAgrum: ", e)"""

        # data.to_pickle(f"{dir_saving}/{env_id}.pkl")
        print("\n")
