import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _extract_technical_features(data_slice):
    """
    Helper to build a flat technical-indicator vector from a slice of the
    dataframe corresponding to a single 'day' (multiple tickers).
    Expects columns: macd, rsi, cci, adx.
    """
    macd = data_slice["macd"].values
    rsi = data_slice["rsi"].values
    cci = data_slice["cci"].values
    adx = data_slice["adx"].values
    features = np.concatenate([macd, rsi, cci, adx], axis=0).astype(np.float32)
    return features


class BaseSignalEnv(gym.Env):
    """
    Single-agent environment that learns a *signal* (position strength)
    from technical indicators, without handling cash / inventory explicitly.

    The agent observes technical indicators and outputs a per-asset signal
    each step. Reward is based on realized next-period returns.

    mode='long'  : actions in [0, 1], reward ≈ long PnL
    mode='short' : actions in [-1, 0], reward ≈ short PnL
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe,
        nb_stock: int,
        start_day: int = 0,
        end_day: int | None = None,
        mode: str = "long",
    ) -> None:
        super().__init__()

        assert mode in {"long", "short"}
        self.mode = mode

        self.dataframe = dataframe
        self.nb_stock = nb_stock

        # We assume the dataframe index is an integer "day" index as in StockEnv
        self.start_day = int(start_day)
        self.end_day = int(end_day) if end_day is not None else int(dataframe.index.max())
        self.day = self.start_day

        # --- Observation space: technical indicators for all stocks
        # shape = 4 * nb_stock  (macd, rsi, cci, adx)
        example_slice = dataframe.loc[self.day, :]
        obs_example = _extract_technical_features(example_slice)
        low = np.full_like(obs_example, -np.inf, dtype=np.float32)
        high = np.full_like(obs_example, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Action space: per-stock signal strength
        if mode == "long":
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(nb_stock,), dtype=np.float32
            )
        else:  # short
            self.action_space = spaces.Box(
                low=-1.0, high=0.0, shape=(nb_stock,), dtype=np.float32
            )

        self._last_obs = None

    def _get_obs(self) -> np.ndarray:
        data_slice = self.dataframe.loc[self.day, :]
        obs = _extract_technical_features(data_slice)
        self._last_obs = obs
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = self.start_day
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Clip to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current_day = self.day
        next_day = current_day + 1

        # If we've reached or passed the end, terminate immediately.
        if next_day > self.end_day:
            # No more data; zero reward on final dummy step
            obs = self._last_obs if self._last_obs is not None else self._get_obs()
            return obs, 0.0, True, False, {}

        current_slice = self.dataframe.loc[current_day, :]
        next_slice = self.dataframe.loc[next_day, :]

        # Next-day simple returns: (P_{t+1} / P_t) - 1
        current_prices = current_slice["adjcp"].values
        next_prices = next_slice["adjcp"].values
        returns = (next_prices / current_prices) - 1.0  # shape (nb_stock,)

        if self.mode == "long":
            # Long: reward ~ position * return
            per_asset_reward = action * returns
        else:
            # Short: exposure magnitude = -action (>= 0), PnL ~ -exposure * return
            exposure = -action  # in [0, 1]
            per_asset_reward = -exposure * returns

        reward = float(np.mean(per_asset_reward))

        # Advance to next day
        self.day = next_day
        terminated = self.day >= self.end_day
        truncated = False

        obs = self._get_obs()
        info = {
            "returns": returns,
            "per_asset_reward": per_asset_reward,
        }
        return obs, reward, terminated, truncated, info


class LongSignalEnv(BaseSignalEnv):
    def __init__(self, dataframe, nb_stock: int, start_day: int = 0, end_day: int | None = None):
        super().__init__(dataframe, nb_stock, start_day, end_day, mode="long")


class ShortSignalEnv(BaseSignalEnv):
    def __init__(self, dataframe, nb_stock: int, start_day: int = 0, end_day: int | None = None):
        super().__init__(dataframe, nb_stock, start_day, end_day, mode="short")
