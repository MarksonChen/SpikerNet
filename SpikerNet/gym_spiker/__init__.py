#----------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry         CAP Lab Purdue University Weldon School of BME
# Date: 09/02/20
# Revision History: N/A
# Purpose: This is the setup file to get spiking reinforcement learning setup as a gym environment
# Hardware: This version implements real time reinforcement learning using Tucker-Davis Technologies (TDT)
# Sys3 hardware. Use of ActiveX or TDTPy is necessary. This can be modified for Blackrock, Plexon, 
# OpenEphys, etc. setups. 
# Note: Adapted from Ashish Poddar https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
#----------------------------------------------------------------------------------------------------------------------------
from gym.envs.registration import register

register(
    id='spiker-v0',              #This is the name you call the environment by
    entry_point='gym_spiker.envs:SpikerEnv',
)