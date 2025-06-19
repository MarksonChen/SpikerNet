import gymnasium as gym, torch, gym_spiker
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

MAX_EP_LEN  = 100
TOTAL_STEPS = 300_000
N_ENVS      = 8                     # TODO: check if can be larger
LOG_DIR     = "./logs/gpu_run"
MODEL_PATH  = "./models/gpu_run"

def make_env():
    return gym.wrappers.TimeLimit(gym.make("spiker-v0"), MAX_EP_LEN)

if __name__ == "__main__":
    env     = VecMonitor(SubprocVecEnv([make_env]*N_ENVS), LOG_DIR)
    logger  = configure(LOG_DIR, ["stdout", "tensorboard", "csv"])
    model   = TD3(
        "MlpPolicy",
        env,
        device="cuda",                # now using RTX A5000
        batch_size=512,               # TODO: check if can be larger
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),
        ),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )
    model.set_logger(logger)
    model.learn(total_timesteps=TOTAL_STEPS, log_interval=1_000)
    model.save(MODEL_PATH)
    env.close()
