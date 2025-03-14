import os

from gymnasium import envs

from benchmarking.utils import (
    benchmarking_df,
    benchmarking_df_pgmpy,
    benchmarking_df_pyagrum,
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
    # env_ids = [
    #    "FrozenLake-v1"
    # ]  # ["Walker2d-v5", "HumanoidStandup-v5", "HalfCheetah-v5", "Pusher-v5"]
    # env_ids = ["FrozenLake-v1", "Taxi-v3"]
    # env_ids = ["CliffWalking-v0"]
    # print(len(env_ids), env_ids)
    n_steps = 100000

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

        benchmarking_df(
            dag,
            data,
            target_feature,
            if_discrete,
            batch_size=512,
            task_name=env_id,
            file_path=f"{dir_saving}",
        )

        try:
            benchmarking_df_pgmpy(
                dag,
                data,
                target_feature,
                if_discrete,
                task_name=env_id,
                file_path=f"{dir_saving}",
            )
        except Exception as e:
            print("PGMPY: ", e)
            # print(f"{env_id} cannot be solved with pgmpy MLE and Variable Elimination")

        try:
            benchmarking_df_pyagrum(
                dag,
                data,
                target_feature,
                if_discrete,
                task_name=env_id,
                file_path=f"{dir_saving}",
            )
        except Exception as e:
            print("PyAgrum: ", e)

        print("\n")
