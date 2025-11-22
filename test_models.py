# test_models.py

import os
from typing import Dict, List

import yaml
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator

import empyrical
from stable_baselines3 import PPO, A2C, DDPG

from utils.create_dataset import data_split
from env.stockEnv import StockEnv


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_episode_sb3(model, env, max_iter: int = 10_000) -> np.ndarray:
    """
    Run one evaluation episode with an SB3 model on StockEnv.

    Returns
    -------
    balances : np.ndarray
        Sequence of portfolio values over the episode.
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0

    while not done and not truncated and steps < max_iter:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    return np.array(env.asset_memory, dtype=float)


def collect_metrics_for_algo(
    algo_name: str,
    model_cls,
    model_dir: str,
    env: StockEnv,
    n_eval_episodes: int = 10,
) -> Dict[str, List[float]]:
    """
    Evaluate all models for a given algorithm and collect performance metrics.

    Returns a dict with keys:
        'balances', 'cum_returns', 'sharpe_ratio',
        'ann_return', 'ann_vol', 'max_dd'
    """
    results = {
        "balances": [],
        "cum_returns": [],
        "sharpe_ratio": [],
        "ann_return": [],
        "ann_vol": [],
        "max_dd": [],
    }

    if not os.path.isdir(model_dir):
        print(f"[{algo_name}] Model directory not found: {model_dir} (skipping).")
        return results

    model_files = sorted(
        f for f in os.listdir(model_dir) if f.endswith(".zip") and not f.startswith(".")
    )

    if not model_files:
        print(f"[{algo_name}] No .zip models found in {model_dir} (skipping).")
        return results

    print(f"[{algo_name}] Evaluating {len(model_files)} models, "
          f"{n_eval_episodes} episodes each.")

    for file in model_files:
        model_path = os.path.join(model_dir, file)
        print(f"[{algo_name}] Loading model: {model_path}")
        model = model_cls.load(model_path)

        for _ in range(n_eval_episodes):
            balances = evaluate_episode_sb3(model, env)
            results["balances"].append(balances)

            # daily returns
            daily_returns = np.diff(balances) / balances[:-1]
            ret_series = pd.Series(daily_returns)

            # cumulative returns over the episode
            cumulative_returns = balances[-1] / balances[0] - 1.0
            results["cum_returns"].append(cumulative_returns)

            sharpe_ratio = empyrical.sharpe_ratio(ret_series)
            results["sharpe_ratio"].append(sharpe_ratio)

            annualized_return = empyrical.annual_return(ret_series)
            results["ann_return"].append(annualized_return)

            max_drawdown = empyrical.max_drawdown(ret_series)
            results["max_dd"].append(max_drawdown)

            annualized_volatility = empyrical.annual_volatility(ret_series)
            results["ann_vol"].append(annualized_volatility)

    return results


def plot_algorithms_vs_djia(
    dates,
    algo_curves: Dict[str, np.ndarray],
    djia_price_rel: np.ndarray,
    title: str = "SB3 Algorithms vs DJIA",
):
    """
    Plot mean asset value curves for multiple algorithms versus DJIA.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Make sure DJIA length is aligned to the minimal len among algos
    min_len = len(djia_price_rel)
    for vals in algo_curves.values():
        min_len = min(min_len, len(vals))

    dates_plot = pd.to_datetime(dates[:min_len])
    djia_plot = djia_price_rel[:min_len]

    # Plot each algo
    for algo_name, mean_vals in algo_curves.items():
        ax.plot(dates_plot, mean_vals[:min_len], label=algo_name, linewidth=2)

    # DJIA
    ax.plot(dates_plot, djia_plot, label="DJIA", linewidth=2, linestyle="--")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Asset Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    ax.xaxis.set_major_locator(MonthLocator())
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def print_summary_table(all_results: Dict[str, Dict[str, List[float]]]):
    """
    Print a summary stats table across algorithms.

    all_results: dict of
        algo_name -> metrics dict (like results for one algo)
    """
    rows = []
    for algo_name, metrics in all_results.items():
        if not metrics["sharpe_ratio"]:
            # No data for this algo
            continue
        row = {
            "Algorithm": algo_name,
            "Mean Sharpe Ratio": np.mean(metrics["sharpe_ratio"]),
            "Mean Annual Return": np.mean(metrics["ann_return"]),
            "Mean Max Drawdown": np.mean(metrics["max_dd"]),
            "Mean Annual Volatility": np.mean(metrics["ann_vol"]),
            "Mean Cumulative Returns": np.mean(metrics["cum_returns"]),
        }
        rows.append(row)

    if not rows:
        print("No statistics to display (no models evaluated).")
        return

    df = pd.DataFrame(rows)
    print("\n=== Summary statistics across algorithms ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # 1. Load dataset and create test split (2016-2020)
    dataset = pd.read_csv("processed_dataset.csv")
    dataset = data_split(dataset, "2016-01-01", "2020-01-01")
    dates = dataset.date.drop_duplicates().values

    # 2. Load env configs
    env_configs = load_yaml("configs/env_configs.yaml")

    # 3. Create base environment for evaluation
    env = StockEnv(dataset, **env_configs)

    # 4. Where models live
    model_root = "models"
    algo_specs = {
        "PPO": {
            "dir": os.path.join(model_root, "sb3_ppo"),
            "cls": PPO,
        },
        "A2C": {
            "dir": os.path.join(model_root, "sb3_a2c"),
            "cls": A2C,
        },
        "DDPG": {
            "dir": os.path.join(model_root, "sb3_ddpg"),
            "cls": DDPG,
        },
    }

    # 5. Evaluate each algorithm
    all_results = {}
    n_eval_episodes = 10

    for algo_pretty, spec in algo_specs.items():
        metrics = collect_metrics_for_algo(
            algo_name=algo_pretty,
            model_cls=spec["cls"],
            model_dir=spec["dir"],
            env=env,
            n_eval_episodes=n_eval_episodes,
        )
        all_results[algo_pretty] = metrics

    # 6. Compute mean curves for plotting
    algo_curves = {}
    for algo_pretty, metrics in all_results.items():
        balances_list = metrics["balances"]
        if not balances_list:
            continue
        n_samples = len(balances_list)
        mean_curve = np.mean(balances_list, axis=0)
        algo_curves[algo_pretty] = mean_curve

    # 7. Download DJIA as benchmark (same range)
    djia_data = yf.download("^DJI", start="2016-01-01", end="2020-01-01")
    first_close_price = djia_data["Close"].iloc[0]
    djia_price_rel = 1e6 * djia_data["Close"] / first_close_price
    djia_price_rel = djia_price_rel.values

    # 8. Plot algorithms vs DJIA
    if algo_curves:
        plot_algorithms_vs_djia(dates, algo_curves, djia_price_rel)
    else:
        print("No curves to plot (no models evaluated).")

    # 9. Print summary statistics
    print_summary_table(all_results)
