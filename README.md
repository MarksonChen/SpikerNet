# SpikerNet_StarProtocols
![Screenshot of SpikerNet Setup.](https://github.com/bscoventry/SpikerNet_StarProtocols/blob/main/Figure1.png)
This repository is a supplement to Coventry and Bartlett's tutorial on using Deep Reinforcement Learning for Closed-loop Neural Control, an algorithm we call SpikerNet.
# Initialization and Setup
To setup SpikerNet, both for computational models and in vivo runs, first download the anaconda environment: https://www.anaconda.com/
Open the environment and initialize a specific SpikerNet environment by running:
```
conda create --name SpikerNet python=3.6.8
```
Currently, SpikerNet is stable for this version of Python. Will extend to later versions later on.
Next, install SpikerNet requirements by running
```
python setup.py install
```
or
```
python3 setup.py install
```
Alternatively, each package can be installed individually.
If using a TDT interface for in vivo control, run
```
pip install tdtpy
```
