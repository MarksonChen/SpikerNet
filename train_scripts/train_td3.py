import gymnasium as gym, torch, gym_spiker
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

MAX_EP_LEN  = 100
TOTAL_STEPS = 300_000
TOT_EPOCHS  = 50
REWARD_THR  = 1.5
N_ENVS      = 8                     # TODO: check if can be larger
LOG_DIR     = "/workspace/SpikerNet/logs/run"
MODEL_PATH  = "/workspace/SpikerNet/models/run"
SEED        = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
set_random_seed(SEED)

def make_env(rank: int):
    def _init():
        env = gym.wrappers.TimeLimit(
            gym.make("spiker-v1", use_cuda=True),  # <- if you ported SpikerNet to CUDA
            MAX_EP_LEN
        )
        env.reset(seed=SEED + rank)
        env.action_space.seed(SEED + rank)
        env.observation_space.seed(SEED + rank)
        return env
    return _init

if __name__ == "__main__":
    np.random.seed(seed); torch.manual_seed(seed)
    env = VecMonitor(
        AsyncVectorEnv([make_env(i) for i in range(N_ENVS)]),
        LOG_DIR
    )
    logger  = configure(LOG_DIR, ["tensorboard", "stdout"])
    model   = TD3(
        "MlpPolicy",
        env,
        device="cuda",                # now using RTX A5000
        batch_size=512,               # TODO: check if can be larger,
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=SEED,
    )
    model.set_logger(logger)
    
    model.learn(total_timesteps=TOTAL_STEPS, log_interval=10)
    model.save(MODEL_PATH)
    env.close()
