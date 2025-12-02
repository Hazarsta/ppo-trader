"""
Train two SB3 PPO *signal* agents:

- LongSignalEnv  (actions in [0, 1])
- ShortSignalEnv (actions in [-1, 0])

Both agents only see technical indicators, *not* portfolio state. They are later
plugged into TradingEnvWithSignals as fixed feature generators.
"""

import os
import yaml
import pandas as pd
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor

from utils.create_dataset import data_split
from env.signal_envs import LongSignalEnv, ShortSignalEnv


def make_signal_envs(dataset: pd.DataFrame, env_configs: dict, train: bool = True):
    """
    Build a pair of (long_env, short_env) over a date range determined from env_configs.
    We reuse the same date ranges as the original project (train/test split).
    """
    if train:
        start, end = "2010-01-01", "2018-01-01"
    else:
        start, end = "2018-01-01", "2020-01-01"

    split = data_split(dataset, start, end)
    nb_stock = env_configs["nb_stock"]

    # The dataframe index is assumed to be the integer "day" used in StockEnv
    start_day = int(split.index.min())
    end_day = int(split.index.max())

    long_env = LongSignalEnv(split, nb_stock=nb_stock, start_day=start_day, end_day=end_day)
    short_env = ShortSignalEnv(split, nb_stock=nb_stock, start_day=start_day, end_day=end_day)
    return long_env, short_env


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")

    with open(os.path.join(configs_dir, "env_configs.yaml"), "r") as f:
        env_configs = yaml.safe_load(f)

    # You can either load the preprocessed dataset or call create_dataset yourself.
    data_path = os.path.join(root_dir, "processed_dataset.csv")
    dataset = pd.read_csv(data_path)

    # 1. Build environments
    long_env, short_env = make_signal_envs(dataset, env_configs, train=True)

    os.makedirs(os.path.join(root_dir, "models", "signal_agents"), exist_ok=True)

    # 2. Wrap with Monitor to log rewards/etc.
    long_env = Monitor(long_env)
    short_env = Monitor(short_env)

    # 3. Train long-signal PPO
    long_model = SB3PPO(
        "MlpPolicy",
        long_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    long_model.learn(total_timesteps=300_000, progress_bar=True)
    long_model.save(os.path.join(root_dir, "models", "signal_agents", "ppo_long_signal.zip"))

    # 4. Train short-signal PPO
    short_model = SB3PPO(
        "MlpPolicy",
        short_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    short_model.learn(total_timesteps=300_000, progress_bar=True)
    short_model.save(os.path.join(root_dir, "models", "signal_agents", "ppo_short_signal.zip"))


if __name__ == "__main__":
    main()
