#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 12/8/20
# Purpose: This is the main function for running reinformcement learning on in vivo stimulation preparations
# Revision History: 12/16/20 - Added vars for control of time steps and total epochs. Number of agents set to 1, for real-time operation
# Notes:
# Toolboxes Necessary: Standard scipy (numpy, scipy, etc), Matplotlib, GenRL, Gym, torch
#--------------------------------------------------------------------------------------------------------------------------------------------------------
""" Imports """
import numpy as np
import gym
import genrl
import gym_spiker     #Import the spiking environment
from genrl.environments import VectorEnv
from genrl.agents import TD3
from genrl.trainers import OffPolicyTrainer
if __name__ == '__main__':
    """Setup simulation variables"""
    envRender = True
    envLogMode = ['stdout','tensorboard','csv']
    save_intervals = 100
    envLogDir = "./logs/INS2015_12_15_20"
    saveLog = "./models/INS2015_12_15_20"
    maxTimeSteps = 30
    max_ep_len = 30
    tot_epochs = 10

    """Setup simulation: Setup as a TD3 agent"""
    env = VectorEnv("spiker-v0",n_envs=1)
    agent = TD3("mlp",env) 
    trainer = OffPolicyTrainer(agent,env,max_timesteps = maxTimeSteps,render = envRender,log_mode = envLogMode, log_interval = save_intervals, logdir = envLogDir, save_model = saveLog, max_ep_len = max_ep_len,epochs=tot_epochs)

    """Run the simulation"""
    trainer.train()
    trainer.evaluate()