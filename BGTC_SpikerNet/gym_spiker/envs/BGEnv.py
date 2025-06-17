#----------------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry         CAP Lab Purdue University Weldon School of BME
# Date: 09/02/20
# Revision History: 09/15/20 - Adaptation made for multiprocessor use. On a single RX processor, spike recordings
#                              and writing data to output buffer to be sent to the stimulator uses similar RAM space,
#                              causing memory overflows. Splitting recording and stimulation between two processors
#                              eases memory load and makes the data pipeline flow easier.
#                   12/13/20 - Initial trial runs and testing
#                   12/14/20 - Added additional MSE reward functions
#                   12/15/20 - First successful trial run. 
#                   12/16/20 - Refactoring of save states and total number of runs
# Purpose: This is the definition file of the spiker reinforcement learning framework.
# Hardware: This version implements real time reinforcement learning using Tucker-Davis Technologies (TDT)
# Sys3 hardware. Use of ActiveX or TDTPy is necessary. This can be modified for Blackrock, Plexon, 
# OpenEphys, etc. setups. This particular application uses the RX-7, but can be easily modified for other 
# DSP processors.
# Note: Adapted from Ashish Poddar https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
#----------------------------------------------------------------------------------------------------------------------------
#Begin with module imports
# import gym
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
from numpy import random
from .mfm import MFM
import torch

import pdb
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
from itertools import count
from PIL import Image
from time import sleep
import scipy as sci
import bisect
from sklearn.metrics import mean_squared_error
import scipy.signal as sig
from importlib.resources import files
import pickle

class BGEnv(gym.Env):
  metadata = {"render_modes": [None, "human"], "render_fps": 30}

  def __init__(self, chSel = 4, stimLimits = [-10,10,60,600,30,200], render_mode=None):
    """
    The init function will initialize our startup variables and begin the process of setting up the
    TDT hardware. This implementation will use the RX-7 for both stim and record.
    #Inputs: chSel = Channel over which reinforcement learning will be run
    #        stimLimits = Limits to place on estim parameters
    #        stimLimits = [stimLow,stimHigh,PWLow,PWHigh]
    """
    super().__init__()          #For inheritence reasons
    self.render_mode = render_mode
    with files('gym_spiker.envs').joinpath('trialPSTH.pickle').open('rb') as f:
      self.targetPSTH = pickle.load(f)
    self.targetPSTH = np.asarray(self.targetPSTH)
    
    self.obsStore = {}
    self.rewardHist = {}
    self.paramStore = {}
    self.spkStore = {}
    self.counter = 0
    self.stimLimists = stimLimits   #So we don't accidentally burn up our laser.
    self.done = False
    self.rewardTarget = 25#1.5
    self.obs = []
    self.info = {}
    self.stimHigh = stimLimits[1]
    self.stimLow = stimLimits[0]
    self.PWLow = stimLimits[2]
    self.PWHigh = stimLimits[3]
    self.freqLow = stimLimits[4]
    self.freqHigh = stimLimits[5]
    self.reward = 0                 #Reward when completing a state.
    self.animate = 1
    self.fs = 1000
    
    
    self.numStimTrials = 1             #Number of stimulation trials to generate PSTH
    self.datalen = 23193                #Corresponds to 950ms on time
    self.binSize = 5                    #Bin Size in ms
    self.numBins = int(np.ceil(self.datalen/(self.binSize*0.001*self.fs)))
    self.low = np.array([self.stimLow,self.PWLow,self.freqLow])
    self.high = np.array([self.stimHigh,self.PWHigh,self.freqHigh])
    self.action_space = gym.spaces.Box(self.low,self.high,dtype=np.float16)
    self.obsLow = -10000.*np.ones((1025,))
    self.obsHigh = 10000.*np.ones((1025,))
    self.observation_space = gym.spaces.Box(low = self.obsLow,high = self.obsHigh)
    self.spkDetectDeadTime = 0.003
    self.spkSearchTime = 0.002
    
    self. max_episode_len = 1000
    
  def step(self, stimSet):
    """Run one environment step."""
    # `stimSet` is a vector of action values with the form
    # [NumPulses, INSAmpVal, TmBetPulses, PulseWidth]
    stimSet = self.checkPulses(stimSet)                  #Make sure learned pulses aren't violating upper or lower bounds
              #Generate waveform to be sent to stimulator
    kwargs = {
        'cDBS': True,
        'cDBS_amp': stimSet[0],
        'cDBS_f': stimSet[2],
        'cDBS_width': stimSet[1]
    }
    def parse_kwargs(kwargs):
        args = {}
        for arg in kwargs:
            key,value = arg.split('=')
            if value.lower() == 'false': value = False
            elif value.lower() == 'true' : value = True
            else:
                try: value = int(value)
                except:
                    try: value = float(value)
                    except:
                        pass
            args[key] = value
        return args
    mfm = MFM(**kwargs)
    dataStore = {}
    for ck in range(self.numStimTrials):
        mfm.run()
        evokedPXX = mfm.getPXX()
    reward = self.getScores(self.targetPSTH,evokedPXX,scoreType=4)
    self.reward = reward
    # ensure observation dtype matches the space
    self.obs = np.asarray(evokedPXX, dtype=self.observation_space.dtype)
    self.obsStore[str(self.counter)] = evokedPXX
    self.rewardHist[str(self.counter)] = reward
    self.paramStore[str(self.counter)] = stimSet

    self.counter = self.counter + 1
    self.checkEnd()
    print(self.reward)
    # Gymnasium requires (obs, reward, terminated, truncated, info)
    return self.obs, self.reward, self.done, False, {}
  
  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.done = False
    self.reward = 0
    # return a valid observation for the Box space
    self.obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
    return self.obs, self.info
  
  def render(self, mode='human', close=False):
    if self.animate == 1:
        # plt.ion()
        # plt.figure(1)
        # fig, axs = plt.subplots(2, 1)
        # plt.show()
        # axs[0].plot(self.targetPSTH)
        # axs[1].plot(self.obs)
        # plt.draw()
        # plt.pause(0.001)
        try:
          ax1.clear()
          ax2.clear()
        except:
          pass
        plt.ion()
        plt.figure(1)
        ax1 = plt.subplot(211)
        plt.plot(self.targetPSTH)
        ax2 = plt.subplot(212)
        plt.plot(self.obs)
        
        plt.pause(0.001)
        plt.draw()

    print('Current Reward: ' + str(self.reward))
  def genPulses(self,numPulses,PulseVec,PWVec,ISIvec):
    """
    This function generates arbitrary pulse waveforms to learn arbitrary pulse functions
    """
    waveform = np.zeros((1,int(self.fs)))
    waveform = np.squeeze(waveform)
    startDel = 0.*self.fs*(0.001)
    curIndx = startDel
    for ck in range(int(numPulses)):
        curPulse = PulseVec
        curPW = PWVec
        curPWSamp = curPW*self.fs*0.001
        curISI = ISIvec*self.fs*0.001
        waveform[int(round(curIndx)):int(round(curIndx+curPWSamp))] = curPulse
        curIndx = curIndx+curPWSamp+curISI
    return waveform

  def checkPulses(self,stimSet):
    # if stimSet[0] < 1: 
    #     stimSet[0] = 1
    # stimSet[0] = int(stimSet[0])
    # pdb.set_trace()
    # for ck in range(int(stimSet[0])):
    #     #stimSet[0] = int(stimSet[0])        #Force an int here. Can't have fractional number of states.
    #     if stimSet[1][ck] < self.stimLow:
    #         stimSet[1][ck] = self.stimLow
    #     if stimSet[1][ck] >= self.stimHigh:
    #         stimSet[1][ck] = self.stimHigh
    #     if stimSet[2][ck] < self.tBetPulsesLim:
    #         stimSet[2][ck] = self.tBetPulsesLim
    #     if stimSet[3][ck] < self.PWLow:
    #         stimSet[3][ck] = self.PWLow
    #     if stimSet[3][ck] > self.PWHigh:
    #         stimSet[3][ck] = self.PWHigh
    
    if np.isnan(stimSet[0]):
       stimSet[0] = 0
    if np.isnan(stimSet[1]):
       stimSet[1] = 0  
    if np.isnan(stimSet[2]):
       stimSet[2] = 0
    if stimSet[0] < self.stimLow:
        stimSet[0] = self.stimLow
    if stimSet[0] > self.stimHigh:
        stimSet[0] = self.stimHigh
    if stimSet[1] < self.PWLow:
        stimSet[1] = self.PWLow
    if stimSet[1] > self.PWHigh:
        stimSet[1] = self.PWHigh
    if stimSet[2] < self.freqLow:
        stimSet[2] = self.freqLow
    if stimSet[2] > self.freqHigh:
        stimSet[2] = self.freqHigh
    return stimSet

  def getPSTH(self,data):
    dataSize = len(data)
    numTrials = dataSize
    psthKey = np.zeros((self.numBins,))
    bSize = round(self.binSize*0.001*self.fs)
    sumVal = 0
    psth = np.zeros((self.numBins,))
    for jj in range(self.numBins):
      psthKey[jj] = bSize+sumVal
      sumVal = sumVal + bSize
    for ck in range(numTrials):
      curTrial = str(ck)
      curSpikes = data[curTrial]
      for bc in range(len(curSpikes)):
        curSpike = curSpikes[bc]
        idx = bisect.bisect(psthKey,curSpike)
        try:
          psth[idx] = psth[idx] + 1
        except:
          pass
    psth = psth/(numTrials*0.001*self.binSize)
    return psth, psthKey

  def getScores(self,targetPSTH,runPSTH,scoreType=2):
    if scoreType == 1:
       MSE = mean_squared_error(targetPSTH[39:90],runPSTH[39:90])
       funcHolder = (1-np.power(MSE,0.5))
       corCoeff = np.corrcoef(targetPSTH[39:90],runPSTH[39:90])
       corCoeff = corCoeff[0,1]
       reward = funcHolder*corCoeff
    elif scoreType == 2:
       reward = mean_squared_error(targetPSTH,runPSTH)
    elif scoreType == 3:
       reward = 1./mean_squared_error(targetPSTH[25:71],runPSTH[25:71])
    elif scoreType == 4:
       reward = np.mean(np.divide(targetPSTH[25:71] - runPSTH[25:71],targetPSTH[25:71])*100)
       if reward < 0:
         reward = np.abs(reward)
       elif reward >= 0:
         reward = 0
    else:
       raise ValueError("Score Type is not chosen")
    return reward

  def checkEnd(self):
    if self.reward >= self.rewardTarget:          #Changed to <= for Beta band reduction **********BRANDON CHECK IF THINGS AREN"T WORKING WELL***************************************************
      self.done = True
    if self.done == True:
        print('PSTH Matched')
        with open('obsB.pickle', 'wb') as handle:
            pickle.dump(self.obsStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('stimB.pickle', 'wb') as handle:
            pickle.dump(self.paramStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('rewardB.pickle', 'wb') as handle:
            pickle.dump(self.rewardHist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('spksB.pickle', 'wb') as handle:
            pickle.dump(self.spkStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('obsFinalB.pickle', 'wb') as handle:
            pickle.dump(self.obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('rewardFinalB.pickle', 'wb') as handle:
            pickle.dump(self.reward, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def spkDetect(self,data,detectEdge='pos'):
    # pdb.set_trace()
    # dataSize = np.shape(data)
    # self.spikeThreshStore = np.zeros(dataSize[0])
    # for ck in range(dataSize[0]):
    #     pdb.set_trace()
    #     curData = data[ck,:]
    #     madEstimate = sci.nanmedian(np.abs(curData))/0.6745
    #     threshold = self.stdmin * madEstimate
    #     treshMax = self.stdmax * madEstimate
    #     self.spikeThreshStore[ck] = threshold
    #     crossings = self.detect_thresh_cross(curData,threshold,self.spkDetectDeadTime)
    #     spks = self.align_to_minimum(curData, crossings, self.spkSearchTime)
    #     return spks
    dataSize = np.shape(data)
    self.spikeThreshStore = np.zeros(dataSize[0])
    
    curData = data
    madEstimate = sci.nanmedian(np.abs(curData))/0.6745
    threshold = self.stdmin * madEstimate
    treshMax = self.stdmax * madEstimate
    self.spikeThreshStore = threshold
    crossings = self.detect_thresh_cross(curData,threshold,self.spkDetectDeadTime)
    spks = self.align_to_minimum(curData, crossings, self.spkSearchTime)
    return spks
  def detect_thresh_cross(self,signal,threshold,dead_time = 0.003):
    dead_time_idx = dead_time*self.fs
    threshold_crossings = np.diff((signal <= threshold).astype(int)>0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        #Iteratively remove threshold crossings that lie within deadtime
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings)>=dead_time_idx,0,True)
    return threshold_crossings
  def get_next_minimum(self,signal,index,max_samples_to_search):
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx
  def align_to_minimum(self,signal,threshold_crossings,search_range):
    search_end = int(search_range*self.fs)
    aligned_spikes = [self.get_next_minimum(signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)
  
  def sigFilter(self,inSignal):
    filtSig = {}
    for rr in range(self.numStimTrials):
      curVal = inSignal[str(rr)]
      curfiltSig = sig.sosfiltfilt(self.secOrdS,curVal)
      curfiltSig = sig.sosfiltfilt(self.secOrd60,curfiltSig)
      curfiltSig = sig.sosfiltfilt(self.secOrd120,curfiltSig)
      filtSig[str(rr)] = curfiltSig
    return filtSig

  def getSpikes(self,data):
    spks = {}
    for tt in range(self.numStimTrials):
      curDat = data[str(tt)]
      curSpks = self.spkDetect(curDat)
      spks[str(tt)] = curSpks
    return spks
  
  def densityEstPSTH(self,PSTH):
    numtrials = self.numStimTrials
    PSTH = PSTH.tolist()         #Matlab vars do not like numpy arrays
    dat = matlab.double(PSTH)    #Convert to matlab data struct
    fit1 = self.eng.barsP(dat,matlab.double([0,1.]),numtrials)
    fit1 = np.asarray(fit1)
    fit1 = np.squeeze(fit1)
    return fit1
