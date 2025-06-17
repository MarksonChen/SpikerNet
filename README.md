Forked from https://github.com/bscoventry/DeepRLClosedLoopControl

To install the SpikerNet gym: 

```
conda create -n spikernet python=3.10 -y
conda activate spikernet
pip install -e BGTC_SpikerNet
```

Try this:

```
import gymnasium as gym
import gym_spiker
env = gym.make("spiker-v0", render_mode=None)
obs, _ = env.reset()
print(obs.shape)
```
