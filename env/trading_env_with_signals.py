import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.stockEnv import StockEnv  # type hint / consistency


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
    Wrapper around the existing StockEnv that augments the observation with
    *pre-trained* long and short PPO signal agents.

    - Inner environment: StockEnv handling cash / holdings / transaction costs.
    - Long signal model: SB3 PPO trained on LongSignalEnv (actions in [0, 1])
    - Short signal model: SB3 PPO trained on ShortSignalEnv (actions in [-1, 0])

    The wrapped env's action space is identical to StockEnv (typically [-1, 1]^N),
    but observations are [stockenv_obs, long_signals, short_signals].
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: StockEnv, long_model, short_model, nb_stock: int):
        super().__init__()
        self.base_env = base_env
        self.long_model = long_model
        self.short_model = short_model
        self.nb_stock = nb_stock

        # Action space: same as underlying trading env
        self.action_space = self.base_env.action_space

        # Observation space: concatenate base_obs with 2 * nb_stock signal values
        assert isinstance(self.base_env.observation_space, spaces.Box), \
            "StockEnv observation_space must be Box"
        base_box: spaces.Box = self.base_env.observation_space

        signal_low = np.concatenate(
            [np.zeros(nb_stock, dtype=np.float32), -np.ones(nb_stock, dtype=np.float32)], axis=0
        )
        signal_high = np.concatenate(
            [np.ones(nb_stock, dtype=np.float32), np.zeros(nb_stock, dtype=np.float32)], axis=0
        )

        low = np.concatenate([base_box.low, signal_low], axis=0)
        high = np.concatenate([base_box.high, signal_high], axis=0)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._last_signals = np.zeros(2 * nb_stock, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_signals(self) -> np.ndarray:
        """
        Use the pre-trained long & short models to compute signals given
        the current technical indicators extracted from base_env.dataframe.
        """
        # We assume base_env has attributes `dataframe`, `day`, and `nb_stock`
        df = self.base_env.dataframe
        day = self.base_env.day
        data_slice = df.loc[day, :]

        tech_obs = _extract_technical_features(data_slice)

        # Each model outputs an action vector of length nb_stock
        long_signal, _ = self.long_model.predict(tech_obs, deterministic=True)
        short_signal, _ = self.short_model.predict(tech_obs, deterministic=True)

        # Ensure correct shape
        long_signal = np.asarray(long_signal, dtype=np.float32).reshape(self.nb_stock)
        short_signal = np.asarray(short_signal, dtype=np.float32).reshape(self.nb_stock)

        signals = np.concatenate([long_signal, short_signal], axis=0)
        self._last_signals = signals
        return signals

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        if self.long_model is None or self.short_model is None:
            # If for some reason models are missing, just use zeros for signals
            signals = self._last_signals
        else:
            signals = self._compute_signals()

        return np.concatenate([base_obs, signals.astype(np.float32)], axis=0)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, info

    def step(self, action):
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        aug_obs = self._augment_obs(np.asarray(base_obs, dtype=np.float32))
        return aug_obs, reward, terminated, truncated, info

    def render(self):
        # Delegate to underlying environment if it has a render method
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def close(self):
        if hasattr(self.base_env, "close"):
            self.base_env.close()
