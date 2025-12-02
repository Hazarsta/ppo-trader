# test_models.py
#
# Compare SB3 A2C, DDPG, PPO and *multi-agent* PPO (PPO trader with
# long/short signal agents) against the DJIA benchmark on the same
# test period.

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
from env.trading_env_with_signals import TradingEnvWithSignals


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_test_split(dataset: pd.DataFrame, start: str = "2016-01-01", end: str = "2020-01-01"):
    """
    Return (test_df, dates) where:
      - test_df is the subset used by StockEnv
      - dates is a DatetimeIndex for plotting (one date per env step)
    """
    test_df = data_split(dataset, start, end)
    # We have multiple tickers per day; env steps once per unique date.
    unique_dates = test_df["date"].drop_duplicates().values
    dates = pd.to_datetime(unique_dates)
    return test_df, dates


def sb3_evaluate_episode(model, env) -> np.ndarray:
    """
    Run a single evaluation episode for a SB3 model
    and return the equity curve (asset value over time).
    """
    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    # StockEnv stores portfolio value in asset_memory
    if hasattr(env, "asset_memory"):
        balances = np.array(env.asset_memory, dtype=np.float64)
    elif hasattr(env, "base_env") and hasattr(env.base_env, "asset_memory"):
        balances = np.array(env.base_env.asset_memory, dtype=np.float64)
    else:
        raise AttributeError("Environment has no asset_memory attribute.")

    return balances


def compute_statistics_from_balances(balances: np.ndarray) -> Dict[str, float]:
    """
    Given an equity curve (portfolio value over time),
    compute standard performance statistics with empyrical.
    """
    # daily returns
    daily_returns = np.diff(balances) / balances[:-1]
    ret_series = pd.Series(daily_returns)

    stats = {
        "cum_returns": balances[-1] / balances[0] - 1.0,
        "sharpe_ratio": empyrical.sharpe_ratio(ret_series),
        "ann_return": empyrical.annual_return(ret_series),
        "ann_vol": empyrical.annual_volatility(ret_series),
        "max_dd": empyrical.max_drawdown(ret_series),
    }
    return stats


def aggregate_results(balances_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Aggregate statistics across multiple runs of the same algorithm.
    """
    if not balances_list:
        return {
            "cum_returns": np.nan,
            "sharpe_ratio": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_dd": np.nan,
        }

    stats_list = [compute_statistics_from_balances(bal) for bal in balances_list]
    keys = stats_list[0].keys()
    mean_stats = {k: float(np.mean([s[k] for s in stats_list])) for k in keys}
    return mean_stats


def plot_algorithms_vs_djia(
    dates: np.ndarray,
    algo_curves: Dict[str, np.ndarray],
    djia_price_rel: np.ndarray,
) -> None:
    """
    Plot each algorithm's mean equity curve vs DJIA.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Align lengths
    min_len = len(djia_price_rel)
    for curve in algo_curves.values():
        min_len = min(min_len, len(curve))

    dates_plot = pd.to_datetime(dates[:min_len])
    djia_plot = djia_price_rel[:min_len]

    # Plot each algo
    for algo_name, mean_vals in algo_curves.items():
        ax.plot(dates_plot, mean_vals[:min_len], label=algo_name, linewidth=2)

    # DJIA
    ax.plot(dates_plot, djia_plot, label="DJIA", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("A2C, DDPG, PPO, Multi-agent PPO vs DJIA")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(MonthLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def print_summary_table(all_results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a summary table of performance metrics for each algorithm.
    """
    rows = []
    for algo_name, stats in all_results.items():
        row = {
            "Method": algo_name,
            "Mean Sharpe Ratio": stats.get("sharpe_ratio", np.nan),
            "Mean Annual Return": stats.get("ann_return", np.nan),
            "Mean Max Drawdown": stats.get("max_dd", np.nan),
            "Mean Annual Volatility": stats.get("ann_vol", np.nan),
            "Mean Cumulative Returns": stats.get("cum_returns", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== Performance Summary ===")
    print(df.to_string(index=False))


# -------------------------------------------------------
# Main evaluation script
# -------------------------------------------------------

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load dataset and env configs
    dataset = pd.read_csv(os.path.join(root_dir, "processed_dataset.csv"))
    env_configs = load_yaml(os.path.join(root_dir, "configs", "env_configs.yaml"))

    # 2. Create test split and dates
    test_df, dates = make_test_split(dataset, start="2016-01-01", end="2020-01-01")

    # 3. Prepare model directories
    model_root = os.path.join(root_dir, "models")
    a2c_dir = os.path.join(model_root, "sb3_a2c")
    ddpg_dir = os.path.join(model_root, "sb3_ddpg")
    ppo_dir = os.path.join(model_root, "sb3_ppo")

    # Multi-agent components
    signal_model_dir = os.path.join(model_root, "signal_agents")
    long_signal_path = os.path.join(signal_model_dir, "ppo_long_signal.zip")
    short_signal_path = os.path.join(signal_model_dir, "ppo_short_signal.zip")
    multiagent_trader_path = os.path.join(model_root, "trader", "ppo_trader_multiagent.zip")

    # 4. Collect all *.zip models for each algo (except multi-agent which is single)
    def list_model_paths(path: str) -> List[str]:
        if not os.path.isdir(path):
            return []
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".zip")
        ]

    a2c_paths = list_model_paths(a2c_dir)
    ddpg_paths = list_model_paths(ddpg_dir)
    ppo_paths = list_model_paths(ppo_dir)

    algo_curves: Dict[str, np.ndarray] = {}
    all_results: Dict[str, Dict[str, float]] = {}

    # 5. Evaluate A2C, DDPG, PPO in the original StockEnv
    algo_specs = [
        ("A2C", A2C, a2c_paths),
        ("DDPG", DDPG, ddpg_paths),
        ("PPO", PPO, ppo_paths),
    ]

    for algo_name, algo_cls, model_paths in algo_specs:
        if not model_paths:
            print(f"[WARN] No saved models found for {algo_name} in {model_root}. Skipping.")
            continue

        print(f"\n=== Evaluating {algo_name} on test period ===")
        balances_list: List[np.ndarray] = []

        for model_path in model_paths:
            print(f"  -> {algo_name} model: {model_path}")
            # Build a fresh env for each run
            env = StockEnv(test_df, **env_configs)
            model = algo_cls.load(model_path)
            balances = sb3_evaluate_episode(model, env)
            balances_list.append(balances)

        # Mean equity curve
        algo_curves[algo_name] = np.mean(balances_list, axis=0)
        # Aggregate stats
        all_results[algo_name] = aggregate_results(balances_list)

    # 6. Evaluate the multi-agent PPO trader:
    #    It uses TradingEnvWithSignals (StockEnv + pre-trained signal agents).
    from stable_baselines3 import PPO as SB3PPO  # just to make intent explicit

    if os.path.isfile(long_signal_path) and os.path.isfile(short_signal_path) and os.path.isfile(multiagent_trader_path):
        print("\n=== Evaluating Multi-agent PPO Trader ===")
        # Load frozen signal agents
        long_model = SB3PPO.load(long_signal_path)
        short_model = SB3PPO.load(short_signal_path)
        # Load final trading PPO
        trader_model = SB3PPO.load(multiagent_trader_path)

        # Build wrapped env
        base_env = StockEnv(test_df, **env_configs)
        wrapped_env = TradingEnvWithSignals(
            base_env=base_env,
            long_model=long_model,
            short_model=short_model,
            nb_stock=env_configs["nb_stock"],
        )

        balances_multi = sb3_evaluate_episode(trader_model, wrapped_env)
        algo_curves["Multi-agent PPO"] = balances_multi
        all_results["Multi-agent PPO"] = aggregate_results([balances_multi])
    else:
        print("\n[WARN] Multi-agent PPO or signal models not found; skipping its evaluation.")
        print(f"  Expected: {long_signal_path}")
        print(f"            {short_signal_path}")
        print(f"            {multiagent_trader_path}")

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


if __name__ == "__main__":
    main()
