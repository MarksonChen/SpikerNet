# A Protocol for Artificial Intelligence-Guided Neural Control Using Deep Reinforcement Learning and Infrared Neural Stimulation (Coventry and Bartlett, STAR Protocols 2025 6, 103496)
![Screenshot of SpikerNet Setup.](https://github.com/bscoventry/SpikerNet_StarProtocols/blob/main/Figure1.png)
This repository is a supplement to Coventry and Bartlett's tutorial on using Deep Reinforcement Learning for Closed-loop Neural Control, an algorithm we call SpikerNet.
Protocol link: https://doi.org/10.1016/j.xpro.2024.103496
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
Sit back and enjoy the run! Output files can be visualized using the SpikerNet_Plotter.py program by running:
```
python SpikerNet_Plotter.py
```
in the directory containing the results (ie the directory where SpikerNet_Main.py was run.)
# Running SpikerNet in In Vivo applications
As every experiment will have widely varying control and sensing mechanisms, SpikerNet must be customized to each experiment. To do so, we highly recommend learning the structure of OpenAI gym environments.
To do so, see the following tutorial: https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
In short, gym environments require the following file-structure, which is preconfigured in this repository:
```
SpikerNet/
  gym_spiker/
    __init__.py                 #Defines environment names
    envs/
      __init__.py
      #Various helper files
      spiker_env.py              #Name of the environment in genrl call
```
All Gym environments need to include 4 key functions in their Python class declaration. This is defined in the spiker_env.py file.
```
class SpikerEnv(gym.Env):
  def __init__(self, #Any necessary inputs)
    #This is used to define any class variables necessary to run SpikerNet. See Code setup for examples
  def step(self, stimSet):
    #This determines what happens in each iteraction of RL. Inputs need to be self, action space change variable.
  def reset(self):
    #This determines what happens when a new episode/epoc is started.
  def render(self, mode='human', close=False):
    # This is an optional function used to define step by step readout. I use it to track learning during RL.
```
The class also includes class functions to help facilitate multiunit recording and detection. The environment and class RL functions are dependent on the experiment being run. The included example shows how to faciliate 
neural recording through TDT circuits and the INSight stimulation system. Generally, the stimulation system is broadly defined to output an analog waveform which can be transduced by a stimulation system which has an Analog-to-Digital
conversion option, like much of the TDT hardware. 
# Example TDT Circuits
We initially built SpikerNet to interface with TDT amplifiers, notably the RZ-2 for observation space neural recordings and an RX-7 for action-space stimulation. This is facilitated through TDTPy (https://tdtpy.readthedocs.io/en/latest/)
and custom circuits. We provided templates that can be updated in OpenEx/Synapse and serve as stimulation and recording examples. 
```
RX-7 Stimulation template: SpikerNet/OstimTestPulse_Reinforce_RX7.rcx
RZ-2 Recording template: SpikerNet/OstimTestPulse_Reinforce_RZ2.rcx
```
# Running SpikerNet
Once Gym environments and stim/recording setups are running and validated, running SpikerNet is as simple as:
```
cd SpikerNet
conda activate SpikerNet
python SpikerNet_Main.py
```
# Citing SpikerNet
If this protocol is utilized, please cite as:
Coventry BS and Bartlett EL (2025). Protocol for artificial intelligence-guided neural control using deep reinforcement learning and infrared neural stimulation. STAR Protocols 6, 103496. DOI: https://doi.org/10.1016/j.xpro.2024.103496

# License
This software can be used under a CC-BY-NC 4.0 License.
