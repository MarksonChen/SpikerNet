#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 12/8/20
# Purpose: This is the main function for running reinformcement learning on computational Basal-Ganglia Thalamocortical models
# Revision History: 12/16/20 - Added vars for control of time steps and total epochs. Number of agents set to 1, for real-time operation
# Notes:
# Toolboxes Necessary: Standard scipy (numpy, scipy, etc), Matplotlib, GenRL, Gym, torch
#--------------------------------------------------------------------------------------------------------------------------------------------------------
""" Imports """
import numpy as np
# import gym
import gymnasium as gym
import gym_spiker     #Import the spiking environment
# import genrl
# from genrl.environments import VectorEnv
# from genrl.agents import TD3
# from genrl.trainers import OffPolicyTrainer

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

"""
This program is the main "runme" program for SpikerNet in the Basal-Ganglia Thalamocortical model. To run SpikerNet, use anaconda to activate the 
environment. The environment holds all packages necessary at runtime. See Github tutorial for setting up the environment. Then, once the anaconda 
environment has been started, run
>> python SpikerNet_Main.py
This will run the program out.
"""
if __name__ == '__main__':
    """Setup simulation variables"""
    envRender = True                             #Set to true to use Open-AI Gym plot feature. This renders the environment, as specified by the user in the Gym setup. 
    envLogMode = ['stdout','tensorboard','csv']          #This outputs evaluation types at the end of run. stdout is through command line, tensorboard and csv output readable files in their respective formats
    save_intervals = 100                        #How often do you want to save? Every 100 steps is default
    envLogDir = "./logs/20221026/samplerun"          #Tells the program where you want to save. .logs needs to be intialized. I like to have a date code followed by a runtime name.
    saveLog = "./models/20221026/samplerun"
    maxTimeSteps = 10                    #Dependent on your environment. See OpenAI Gym setup for full description
    max_ep_len = 100                     #How many samples do you want in each epoch? Dependent on environment
    tot_epochs = 10                      #10 epochs is good to visualize. May need more if evaluating a more complex environment. 

    """Setup simulation: Setup as a TD3 agent"""
    # env = VectorEnv('spiker-v0',n_envs=1)#("spiker-v0",n_envs=1)      #This tells Gym what environment to use. The name is defined in the Gym init in the config folder
    env = DummyVecEnv([lambda: gym.make("spiker-v0")])
    agent = TD3("mlp",env)                          #Type of RL to do and what deep neural network to use. TD3 for continuous spaces, mlp -multilayer perceptron. Good for timeseries. CNN - Good for images
    trainer = OffPolicyTrainer(agent,env,max_timesteps = maxTimeSteps,render = envRender,log_mode = envLogMode, log_interval = save_intervals, logdir = envLogDir, save_model = saveLog, max_ep_len = int(max_ep_len),epochs=tot_epochs)
    #Trainer initializes the environment. See documentation for genrl OffPolicyTrainer (model-free) for full documentation.
    """Run the simulation"""
    trainer.train()             #This step runs RL
    trainer.evaluate()          #This evaluates performance after run