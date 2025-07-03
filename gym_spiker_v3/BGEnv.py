import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from sklearn.metrics import mean_squared_error
from importlib.resources import files
import pickle
from omegaconf import DictConfig, OmegaConf
from mfm import MFM
from utils import elapsed


class BGEnv(gym.Env):
    """Basal-Ganglia mean-field environment with continuous DBS control."""

    def __init__(self, config: DictConfig = None, stim_limits=(-10, 10, 60, 600, 30, 200), reward_target=25):
        super().__init__()

        if config is None:
            config = OmegaConf.load("config/mfm_config.yaml")
        self.cfg = config

        with files('gym_spiker_v1.envs').joinpath('trialPSTH.pickle').open('rb') as f:
            self.target_psth = np.asarray(pickle.load(f))

        (self.stim_low, self.stim_high, self.pw_low, self.pw_high,
         self.freq_low, self.freq_high) = stim_limits

        low = np.array([self.stim_low, self.pw_low, self.freq_low], dtype=np.float32)
        high = np.array([self.stim_high, self.pw_high, self.freq_high], dtype=np.float32)
        self.action_space = Box(low, high, dtype=np.float32)
        self.observation_space = Box(-1e4, 1e4, shape=(1025,), dtype=np.float32)

        self.reward_target = reward_target
        self._reset_state()

    def _reset_state(self):
        """Reset environment state"""
        self.history = dict(obs={}, reward={}, params={})
        self.counter = 0
        self.done = False
        self.reward = 0.0
        self.obs = np.zeros(self.observation_space.shape, self.observation_space.dtype)
        self.info = {}

    def _clip(self, a: np.ndarray) -> np.ndarray:
        """Replace NaNs with 0 and clip to valid range."""
        a = np.nan_to_num(a)
        return np.clip(a, [self.stim_low, self.pw_low, self.freq_low],
                       [self.stim_high, self.pw_high, self.freq_high])

    @staticmethod
    def _score(tgt, run, mode=4):
        """Compute reward given target and run PSTH"""
        if mode == 1:  # weighted corr + RMSE
            seg = slice(39, 90)
            corr = np.corrcoef(tgt[seg], run[seg])[0, 1]
            rmse = mean_squared_error(tgt[seg], run[seg]) ** 0.5
            return (1 - rmse) * corr

        if mode == 2:
            return mean_squared_error(tgt, run)

        if mode == 3:
            seg = slice(25, 71)
            return 1.0 / mean_squared_error(tgt[seg], run[seg])

        if mode == 4:  # percent deviation (beta-band)
            seg = slice(25, 71)
            err = np.mean((tgt[seg] - run[seg]) / tgt[seg]) * 100
            return abs(err) if err < 0 else 0

        raise ValueError(f'Unknown score mode {mode}')

    def step(self, action):
        """Execute one environment step"""
        action = self._clip(action)

        mfm = MFM(config=self.cfg,
                  cDBS=True,
                  cDBS_amp=float(action[0]),
                  cDBS_width=float(action[1]),
                  cDBS_f=float(action[2]))

        with elapsed("mfm.run"):
            mfm.run()

        pxx = mfm.power_spectrum

        self.reward = self._score(self.target_psth, pxx, mode=4)
        self.done = self.reward >= self.reward_target
        self.obs = pxx.astype(self.observation_space.dtype)

        idx = str(self.counter)
        self.history['obs'][idx] = self.obs
        self.history['reward'][idx] = self.reward
        self.history['params'][idx] = action
        self.counter += 1

        if self.cfg.simulation.verbose:
            print(self.reward)

        return self.obs, self.reward, self.done, False, self.info

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self._reset_state()
        return self.obs, self.info
