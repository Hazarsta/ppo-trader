"""
Train the *final* SB3 PPO trading agent that takes as input:

    [StockEnv state, long_signals, short_signals]

and outputs continuous trading actions in [-1, 1] (same as original StockEnv).
"""

import os
import yaml
import pandas as pd

from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor

from utils.create_dataset import data_split
from env.stockEnv import StockEnv
from env.trading_env_with_signals import TradingEnvWithSignals


def make_trading_env_with_signals(dataset: pd.DataFrame, env_configs: dict, long_model, short_model):
    """
    Build a wrapped StockEnv -> TradingEnvWithSignals for the training period.
    """
    # Same date split convention as original project; adjust if needed
    train_set = data_split(dataset, start="2010-01-01", end="2018-01-01")

    # Note: StockEnv expects the full dataset and env_configs, including day index.
    # We set the starting day to the first day of the training set.
    env_kwargs = env_configs.copy()
    env_kwargs["day"] = int(train_set.index.min())

    base_env = StockEnv(train_set, **env_kwargs)

    wrapped_env = TradingEnvWithSignals(
        base_env=base_env,
        long_model=long_model,
        short_model=short_model,
        nb_stock=env_configs["nb_stock"],
    )
    return wrapped_env


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")

    with open(os.path.join(configs_dir, "env_configs.yaml"), "r") as f:
        env_configs = yaml.safe_load(f)

    data_path = os.path.join(root_dir, "processed_dataset.csv")
    dataset = pd.read_csv(data_path)

    # 1. Load pre-trained signal agents
    signal_model_dir = os.path.join(root_dir, "models", "signal_agents")
    long_model = SB3PPO.load(os.path.join(signal_model_dir, "ppo_long_signal.zip"))
    short_model = SB3PPO.load(os.path.join(signal_model_dir, "ppo_short_signal.zip"))

    # 2. Build wrapped multi-agent env for training the final trader
    env = make_trading_env_with_signals(dataset, env_configs, long_model, short_model)
    os.makedirs(os.path.join(root_dir, "runs"), exist_ok=True)
    env = Monitor(env, filename=os.path.join(root_dir, "runs", "multi_agent_trader.csv"))

    os.makedirs(os.path.join(root_dir, "models", "trader"), exist_ok=True)

    # 3. Train final PPO trading agent
    trader_model = SB3PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    trader_model.learn(total_timesteps=500_000, progress_bar=True)
    trader_model.save(os.path.join(root_dir, "models", "trader", "ppo_trader_multiagent.zip"))


if __name__ == "__main__":
    main()