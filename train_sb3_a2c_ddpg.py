# train_sb3_a2c_ddpg.py

import os
import yaml
import pandas as pd
import numpy as np

from stable_baselines3 import A2C, DDPG
from stable_baselines3.common.monitor import Monitor

from env.stockEnv import StockEnv
from utils.create_dataset import data_split


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_algorithm(
    algo_name: str,
    algo_cls,
    config: dict,
    dataset: pd.DataFrame,
    env_configs: dict,
    run_save_path: str,
    model_save_root: str,
):
    """
    Train a single SB3 algorithm (A2C or DDPG) for multiple runs.

    Parameters
    ----------
    algo_name : str
        e.g. 'a2c' or 'ddpg'
    algo_cls  : SB3 class
        e.g. A2C, DDPG
    config    : dict
        Loaded hyperparameters from YAML (config[algo_name])
    dataset   : DataFrame
        Processed dataset for StockEnv
    env_configs : dict
        Config dictionary passed to StockEnv
    run_save_path : str
        Directory where Monitor CSVs are saved
    model_save_root : str
        Root directory where models/<algo_dir>/run.zip are saved
    """
    algo_cfg = config[algo_name].copy()  # e.g. config["a2c"]

    # Extract control parameters from config
    total_timesteps = algo_cfg.pop("total_timesteps", 150_000)
    n_runs = algo_cfg.pop("n_runs", 5)
    policy = algo_cfg.pop("policy", "MlpPolicy")

    # DDPG-specific: set action noise if specified
    if algo_name.lower() == "ddpg":
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env_configs.get("n_stock", 1)  # default to 1 if not specified
        action_noise_std = algo_cfg.pop("action_noise_std", 0.1)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions),
        )
        algo_cfg["action_noise"] = action_noise

    # Where SB3 models will be saved, e.g. models/sb3_a2c
    algo_id = f"sb3_{algo_name.lower()}"
    model_dir = os.path.join(model_save_root, algo_id)
    os.makedirs(model_dir, exist_ok=True)

    # Ensure run log directory exists
    os.makedirs(run_save_path, exist_ok=True)

    print(f"Training {algo_id} with {n_runs} runs, {total_timesteps} timesteps each.")

    for run in range(n_runs):
        print(f"\n=== {algo_id} - Run {run} ===")

        # Create environment
        env = StockEnv(dataset, **env_configs)

        # Each run writes its own monitor CSV
        monitor_file = os.path.join(run_save_path, f"{algo_id}_{run}.csv")
        env = Monitor(env, monitor_file)

        # Instantiate SB3 model with given hyperparameters
        model = algo_cls(policy, env, **algo_cfg)

        # Train
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Close env to flush monitor file
        env.close()

        # Save model
        model_path = os.path.join(model_dir, f"{run}.zip")
        model.save(model_path)
        print(f"Saved {algo_id} model for run {run} to {model_path}")


if __name__ == "__main__":
    # 1. Load dataset and create train split (same range as train.py)
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, "2013-01-01", "2014-01-01")

    # 2. Load env configs
    env_configs = load_yaml("configs/env_configs.yaml")

    # 3. Paths
    run_save_path = "runs/stockEnv"
    model_save_root = "models"

    # 4. Load algorithm configs
    a2c_config = load_yaml("configs/a2c_configs.yaml")
    ddpg_config = load_yaml("configs/ddpg_configs.yaml")

    # 5. Train A2C
#    train_algorithm(
#        algo_name="a2c",
#        algo_cls=A2C,
#        config=a2c_config,
#        dataset=dataset,
#        env_configs=env_configs,
#        run_save_path=run_save_path,
#        model_save_root=model_save_root,
#    )

    # 6. Train DDPG
    train_algorithm(
        algo_name="ddpg",
        algo_cls=DDPG,
        config=ddpg_config,
        dataset=dataset,
        env_configs=env_configs,
        run_save_path=run_save_path,
        model_save_root=model_save_root,
    )
