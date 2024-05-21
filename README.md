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
# Running the Basal-Ganglia Thalamocortical Model
This github profile comes packaged with a SpikerNet implementation of control of a DBS model of basal-ganglia thalamocortical control. This is detailed in the following paper:
doi.org//10.1109/NER52421.2023.10123797. This should work out of the box, though this has not been tested on OSX operating systems. This is also a good example for how to design SpikerNet
environments for custom in vivo runs. To run this model, navigate to the BGTC folder in Anaconda by typing
```
cd BGTC_SpikerNet
```
Ensure that the SpikerNet environment is activated by typing
```
conda activate SpikerNet
```
SpikerNet can be run on the model by simply typing:
```
python SpikerNet_Main.py
```
Sit back and enjoy the run! Output files can be visualized using the SpikerNet_Plotter.py program.
# Running SpikerNet in In Vivo applications
