from genrl.agents import TD3
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer

env = VectorEnv("MountainCarContinuous-v0")
agent = TD3("mlp", env)
trainer = OffPolicyTrainer(agent, env, max_timesteps=40000,render=False)
trainer.train()
trainer.evaluate()
