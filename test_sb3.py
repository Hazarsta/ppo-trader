import os
import yaml
import yfinance as yf
import pickle as pkl  # not used but kept for symmetry with original test.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from datetime import datetime

import empyrical
from stable_baselines3 import PPO as BPPO

from utils.create_dataset import data_split
from env.stockEnv import StockEnv


def sb3_evaluate_episode(model, env, max_iter: int = 10_000) -> np.ndarray:
    """
    Run one evaluation episode with an SB3 PPO model on StockEnv.

    Returns
    -------
    balances : np.ndarray
        The sequence of portfolio values over the episode (env.asset_memory).
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0

    while not done and not truncated and steps < max_iter:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    # StockEnv keeps track of portfolio value over time
    return np.array(env.asset_memory, dtype=float)


def plot_portfolio_stats(
    dates,
    mean_asset_values_sb3: np.ndarray,
    std_error_sb3: np.ndarray,
    djia_price_rel: np.ndarray,
) -> None:
    """
    Plot portfolio statistics: mean SB3 asset values with ±2*std_error,
    and DJIA index normalized to 1e6 initial value.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure everything is the same length
    n = min(len(dates), len(mean_asset_values_sb3), len(djia_price_rel))
    dates_plot = pd.to_datetime(dates[:n])
    mean_sb3 = mean_asset_values_sb3[:n]
    se_sb3 = std_error_sb3[:n]
    djia_plot = djia_price_rel[:n]

    # SB3 mean + band
    ax.plot(dates_plot, mean_sb3, label="SB3 PPO", linewidth=2)
    ax.fill_between(
        dates_plot,
        mean_sb3 - 2 * se_sb3,
        mean_sb3 + 2 * se_sb3,
        alpha=0.3,
        label="SB3 PPO ± 2 SE",
    )

    # DJIA reference
    ax.plot(dates_plot, djia_plot, label="DJIA", linewidth=2)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Asset Value", fontsize=12)
    ax.legend(fontsize=10)

    ax.xaxis.set_major_locator(MonthLocator())
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def print_mean_statistics(sb3_lists_dict: dict) -> None:
    """
    Print mean statistics for SB3 PPO models.

    sb3_lists_dict keys:
        'balances', 'cum_returns', 'sharpe_ratio',
        'ann_return', 'ann_vol', 'max_dd'
    """
    mean_sharpe_ratio_sb3 = np.mean(sb3_lists_dict["sharpe_ratio"])
    mean_ann_return_sb3 = np.mean(sb3_lists_dict["ann_return"])
    mean_max_dd_sb3 = np.mean(sb3_lists_dict["max_dd"])
    mean_ann_vol_sb3 = np.mean(sb3_lists_dict["ann_vol"])
    mean_cum_returns_sb3 = np.mean(sb3_lists_dict["cum_returns"])

    data = {
        "Method": ["SB3 PPO"],
        "Mean Sharpe Ratio": [mean_sharpe_ratio_sb3],
        "Mean Annual Return": [mean_ann_return_sb3],
        "Mean Max Drawdown": [mean_max_dd_sb3],
        "Mean Annual Volatility": [mean_ann_vol_sb3],
        "Mean Cumulative Returns": [mean_cum_returns_sb3],
    }

    df = pd.DataFrame(data)
    print("Mean Statistics (SB3 only):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # 1. Load dataset and create test split
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, "2016-01-01", "2020-01-01")
    dates = dataset.date.drop_duplicates().values

    # 2. Load env configs
    with open("configs/env_configs.yaml", "r") as f:
        env_configs = yaml.safe_load(f)

    model_path = "models/"
    sb3_dir = os.path.join(model_path, "sb3_ppo")

    if not os.path.isdir(sb3_dir):
        raise FileNotFoundError(
            f"SB3 model directory not found: {sb3_dir}\n"
            f"Make sure train.py has created models in {sb3_dir}."
        )

    # 3. Create a base environment for evaluation
    env = StockEnv(dataset, **env_configs)

    # 4. Evaluate SB3 PPO models
    sb3_lists_dict = {
        "balances": [],
        "cum_returns": [],
        "sharpe_ratio": [],
        "ann_return": [],
        "ann_vol": [],
        "max_dd": [],
    }

    # sort files to have a stable order: 0.zip, 1.zip, ...
    model_files = sorted(
        f for f in os.listdir(sb3_dir) if f.endswith(".zip") and not f.startswith(".")
    )

    if not model_files:
        raise FileNotFoundError(
            f"No .zip SB3 models found in {sb3_dir}. "
            f"Make sure train.py has saved models there."
        )

    # how many evaluation episodes per model
    n_eval_episodes = 10

    for file in model_files:
        model_full_path = os.path.join(sb3_dir, file)
        print(f"Evaluating SB3 model: {model_full_path}")

        model = BPPO.load(model_full_path)

        for _ in range(n_eval_episodes):
            # For each episode, we use a fresh reset of the same env
            balances = sb3_evaluate_episode(model, env)
            sb3_lists_dict["balances"].append(balances)

            # daily returns
            daily_returns = np.diff(balances) / balances[:-1]

            # cumulative return over the episode
            cumulative_returns = balances[-1] / balances[0] - 1.0
            sb3_lists_dict["cum_returns"].append(cumulative_returns)

            # convert to pandas Series for empyrical
            ret_series = pd.Series(daily_returns)

            sharpe_ratio = empyrical.sharpe_ratio(ret_series)
            sb3_lists_dict["sharpe_ratio"].append(sharpe_ratio)

            annualized_return = empyrical.annual_return(ret_series)
            sb3_lists_dict["ann_return"].append(annualized_return)

            max_drawdown = empyrical.max_drawdown(ret_series)
            sb3_lists_dict["max_dd"].append(max_drawdown)

            annualized_volatility = empyrical.annual_volatility(ret_series)
            sb3_lists_dict["ann_vol"].append(annualized_volatility)

    # 5. Compute mean SB3 portfolio and volatility band
    n_samples = len(sb3_lists_dict["balances"])
    sb3_mean_asset = np.mean(sb3_lists_dict["balances"], axis=0)
    sb3_asset_std_error = np.std(sb3_lists_dict["balances"], axis=0) / np.sqrt(
        n_samples
    )

    # 6. Download DJIA as benchmark
    djia_data = yf.download("^DJI", start="2016-01-01", end="2020-01-01")
    first_close_price = djia_data["Close"].iloc[0]
    djia_price_rel = 1e6 * djia_data["Close"] / first_close_price
    djia_price_rel = djia_price_rel.values  # to numpy

    # 7. Plot SB3 vs DJIA
    plot_portfolio_stats(dates, sb3_mean_asset, sb3_asset_std_error, djia_price_rel)

    # 8. Print statistics
    print_mean_statistics(sb3_lists_dict)
