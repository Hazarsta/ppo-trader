import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.stockEnv import StockEnv


def _extract_technical_features(data_slice):
    """
    Build the same technical-indicator vector used in the signal-envs:
    flat concatenation of macd, rsi, cci, adx across all stocks for a day.
    """
    macd = data_slice["macd"].values
    rsi = data_slice["rsi"].values
    cci = data_slice["cci"].values
    adx = data_slice["adx"].values
    features = np.concatenate([macd, rsi, cci, adx], axis=0).astype(np.float32)
    return features


class TradingEnvWithSignals(gym.Env):
    """
    Portfolio-aware trading environment:

    - Wraps a base StockEnv that already handles cash, inventory, transaction costs,
      turbulence, etc.
    - Augments the observation with:
        * long signals in [0, 1]^N  (from a pre-trained PPO signal agent)
        * short signals in [-1, 0]^N (from a pre-trained PPO signal agent)
        * current normalized portfolio value (total asset / initial_balance)
    - Re-defines the reward as *portfolio return* between two consecutive steps:

        r_t = (V_t - V_{t-1}) / V_{t-1}

      so that the agent explicitly optimizes percentage changes in total wealth.
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: StockEnv, long_model, short_model, nb_stock: int):
        super().__init__()
        self.base_env = base_env
        self.long_model = long_model
        self.short_model = short_model
        self.nb_stock = nb_stock

        # Initial balance for normalization
        self.initial_balance = float(getattr(self.base_env, "initial_balance", 1.0))

        # Action space: identical to underlying StockEnv
        self.action_space = self.base_env.action_space

        # ---- Observation space ------------------------------------------------
        assert isinstance(self.base_env.observation_space, spaces.Box), \
            "StockEnv observation_space must be Box"
        base_box: spaces.Box = self.base_env.observation_space

        # Signals: long in [0, 1], short in [-1, 0]
        signal_low = np.concatenate(
            [np.zeros(nb_stock, dtype=np.float32), -np.ones(nb_stock, dtype=np.float32)],
            axis=0,
        )
        signal_high = np.concatenate(
            [np.ones(nb_stock, dtype=np.float32), np.zeros(nb_stock, dtype=np.float32)],
            axis=0,
        )

        # One extra feature: normalized portfolio value V_t / V_0
        pv_low = np.array([-np.inf], dtype=np.float32)
        pv_high = np.array([np.inf], dtype=np.float32)

        low = np.concatenate([base_box.low, signal_low, pv_low], axis=0)
        high = np.concatenate([base_box.high, signal_high, pv_high], axis=0)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Cache last signals for edge cases
        self._last_signals = np.zeros(2 * nb_stock, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _compute_signals(self) -> np.ndarray:
        """
        Use the pre-trained long & short models to compute signals given
        the current technical indicators extracted from base_env.dataframe.
        """
        df = self.base_env.dataframe
        day = self.base_env.day
        data_slice = df.loc[day, :]

        tech_obs = _extract_technical_features(data_slice)

        # Each model outputs an action vector of length nb_stock
        long_signal, _ = self.long_model.predict(tech_obs, deterministic=True)
        short_signal, _ = self.short_model.predict(tech_obs, deterministic=True)

        long_signal = np.asarray(long_signal, dtype=np.float32).reshape(self.nb_stock)
        short_signal = np.asarray(short_signal, dtype=np.float32).reshape(self.nb_stock)

        signals = np.concatenate([long_signal, short_signal], axis=0)
        self._last_signals = signals
        return signals

    def _get_portfolio_value(self) -> float:
        """
        Get current total portfolio value V_t.
        Use StockEnv.asset_memory if possible, otherwise reconstruct from state.
        """
        if hasattr(self.base_env, "asset_memory") and len(self.base_env.asset_memory) > 0:
            return float(self.base_env.asset_memory[-1])

        # Fallback: compute from state: cash + sum(price * shares)
        state = np.asarray(self.base_env.state, dtype=np.float32)
        cash = state[0]
        prices = state[1 : 1 + self.nb_stock]
        holdings = state[1 + self.nb_stock : 1 + 2 * self.nb_stock]
        total_asset = float(cash + np.sum(prices * holdings))
        return total_asset

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        # Signals
        if self.long_model is None or self.short_model is None:
            signals = self._last_signals
        else:
            signals = self._compute_signals()

        # Normalized portfolio value
        current_value = self._get_portfolio_value()
        norm_pv = current_value / self.initial_balance if self.initial_balance > 0 else 0.0
        pv_feature = np.array([norm_pv], dtype=np.float32)

        return np.concatenate(
            [base_obs.astype(np.float32), signals.astype(np.float32), pv_feature], axis=0
        )

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        # After reset, StockEnv.asset_memory is typically [initial_balance]
        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, info

    def step(self, action):
        """
        Step the underlying StockEnv, then:
        - Recompute reward as portfolio return:

              r_t = (V_t - V_{t-1}) / V_{t-1}

        - Augment the observation with signals + normalized portfolio value.
        """
        base_obs, _, terminated, truncated, info = self.base_env.step(action)

        # Compute portfolio-return reward using asset_memory
        reward = 0.0
        if hasattr(self.base_env, "asset_memory") and len(self.base_env.asset_memory) >= 2:
            prev_v, curr_v = self.base_env.asset_memory[-2], self.base_env.asset_memory[-1]
            if prev_v != 0:
                reward = (curr_v - prev_v) / prev_v

        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, reward, terminated, truncated, info

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def close(self):
        if hasattr(self.base_env, "close"):
            self.base_env.close()
