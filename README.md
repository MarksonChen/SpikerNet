Forked from: https://github.com/bscoventry/DeepRLClosedLoopControl, the repo for paper "Closed-Loop Reinforcement Learning Based Deep Brain Stimulation Using SpikerNet: A Computational Model" ([Coventry et al.](https://ieeexplore.ieee.org/document/10123797))

SpikerNet's model is from [ModelDB 247310](https://modeldb.science/247310?tab=1): Adaptive dual control of deep brain stimulation in Parkinsons disease simulations ([Grado et al 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006606))

---

GymSpiker v0: See [bscoventry/DeepRLClosedLoopControl/BGTC_SpikerNet
/gym_spiker/](https://github.com/bscoventry/DeepRLClosedLoopControl/tree/main/BGTC_SpikerNet/gym_spiker)

GymSpiker v1: Updated to use Python 3.10 & OpenAI's gymnasium

GymSpiker v2: Removed codes irrelevant for RL training

GymSpiker v3: Significantly updated to use modern python practices, most 
importantly [OmegaConf](https://github.com/omry/omegaconf)

---

Clone this repo:

```bash
git clone https://github.com/MarksonChen/SpikerNet.git
cd SpikerNet
```

Setup:

```bash
conda create -n spikernet python=3.10 -y
conda activate spikernet
python -m pip install --upgrade build pip
```

To install the SpikerNet gym only:

```bash
pip install -e .
```

To install the SpikerNet gym + dependencies for the training scripts:

```bash
pip install -e .[train]
```
