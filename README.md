Forked from https://github.com/bscoventry/DeepRLClosedLoopControl.

A lightweight, Gymnasium-compatible environment that emulates Basal-Ganglia 
Thalamo-Cortical (BGTC) neural dynamics, plus training scripts from SpikerNet

Clone this repo:

```bash
git clone https://github.com/MarksonChen/SpikerNet.git
cd SpikerNet
```

Setup:

```bash
python -m pip install --upgrade build pip
conda create -n spikernet python=3.10 -y
conda activate spikernet
```

To install the SpikerNet gym only:

```bash
pip install -e .
```

To install the SpikerNet gym + dependencies for the training scripts:

```bash
pip install -e .[train]
```

Smoke test:

```python
import gymnasium as gym
import gym_spiker
env = gym.make("spiker-v0")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}") # Expected: (1025,)
```
