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
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import random
import torch
import tdt
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
import pickle
import matlab.engine                #Not necessary for RL, but for PSTH estimation
class SpikerEnv(gym.Env):
  metadata = {'render.modes': ['human']}              #This intializes rendering so you can keep track during runs if your fits are heading in the right direction.

  def __init__(self, chSel = 4, stimLimits = [0,2.1,0.1,10,1,1,10]):
    """
    The init function will initialize our startup variables and begin the process of setting up the
    TDT hardware. This implementation will use the RX-7 for both stim and record.
    #Inputs: chSel = Channel over which reinforcement learning will be run
    #        stimLimits = Limits to place on laser parameters
    #        stimLimits = [stimLow,stimHigh,PWLow,PWHigh,timebetweenpulsesLow,numpulseslow,numpulseshigh]
    """
    super().__init__()          #For inheritence reasons
    with open('trialPSTH.pickle', 'rb') as f:
      self.targetPSTH = pickle.load(f)                #If template matching, load template.
    self.targetPSTH = np.asarray(self.targetPSTH)
    self.targetPSTH = self.targetPSTH[0:190]
    self.obsStore = {}
    self.rewardHist = {}            #Keep track of rewards in a Python Dictionary indexed by trial number.
    self.paramStore = {}            #Track parameters learned during RL
    self.spkStore = {}              #Hold Spike Traces indexed by trial number
    self.counter = 0                #Number of trials
    self.stimLimists = stimLimits   #So we don't accidentally burn up our laser.
    self.done = False                #Done flag is used to signal completion of epocs in RL.
    self.rewardTarget = 10
    self.obs = []                     #Observation space
    self.stimHigh = stimLimits[1]      #Give class access to stimulation limits.
    self.stimLow = stimLimits[0]
    self.PWLow = stimLimits[2]
    self.PWHigh = stimLimits[3]
    self.tBetPulsesLim = stimLimits[4]
    self.numPulsesLow = stimLimits[5]
    self.numPulsesHigh = stimLimits[6]
    self.reward = 0                 #Reward when completing a state.
    self.animate = 1                #Animate flag to signal when to display render.
    projectRX7 = tdt.DSPProject()      #Here we will setup stimulation for the RX-7
    projectRZ2 = tdt.DSPProject()      #Set up recording with the RZ-2
    self.circuitRX7 = projectRX7.load_circuit('OstimTestPulse_Reinforce_RX7_Delay.rcx', 'RX7')      #Use the circuit in first argument, processor in second.
    self.circuitRZ2 = projectRZ2.load_circuit('OstimTestPulse_Reinforce_RZ2.rcx', 'RZ2')
    tdt.util.connect_zbus()         #Connect to device I/O trigger. Here we use Z-bus but software triggers can also be used.
    self.circuitRX7.start()              #Start up RX7 circuit
    self.circuitRZ2.start()              #Start up RZ2 circuit
    self.circuitRZ2.set_tags(ChSel = chSel)        #Initialize the stimulation parameters
    self.fs = self.circuitRZ2.fs
    self.numStimTrials = 10             #Number of stimulation trials to generate PSTH
    self.datalen = 23193                #Corresponds to 950ms on time
    self.binSize = 5                    #Bin Size in ms
    self.numBins = int(np.ceil(self.datalen/(self.binSize*0.001*self.fs)))        #This is a helper for binning spike counts.
    self.low = np.array([self.numPulsesLow,self.stimLow,self.PWLow,self.tBetPulsesLim])  #List of stim limits low
    self.high = np.array([self.numPulsesHigh,self.stimHigh,self.PWHigh,200])      #List of stim limits high
    self.action_space = gym.spaces.Box(self.low,self.high,dtype=np.float16)
    self.obsLow = np.zeros((1,self.numBins))                  #Need to bound observation space for Box. We choose an ultra high number that is not achievable by spikes so that all spike counts can be seen.
    self.obsHigh = 10000.*np.ones((1,self.numBins))
    self.observation_space = gym.spaces.Box(low = self.obsLow,high = self.obsHigh)     #Initialize the observation space, which is spike counts in this implementation.
    self.spkDetectDeadTime = 0.003                        #Set refractory period for multiunit activity.
    self.spkSearchTime = 0.002
    #Define second order system filters. SecOrds - Bandpass 300-5000 #These are for spike filters and noise rejection.
    self.secOrdS = np.ndarray((19,6))
    self.secOrdS[:,0] = np.ones((19,))
    self.secOrdS[:,1] = [-0.557083233271196, -1.99406431868261, -0.555625692265288, -1.99407367452313, -0.551515284301230, -1.99409995772041, -0.541366484563647, -1.99416421929473, -0.516729767562496, -1.99431656727504, -0.456583674780893, -1.99466811085462, -0.306970000879784, -1.99543444855932, 0.0760708921716276, -1.99689385705562, 1.01135110986488, -1.99889903613736, 0]
    self.secOrdS[:,2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]
    self.secOrdS[:,3] = np.ones((19,))
    self.secOrdS[:,4] = [-0.560425969776873, -1.99401853059226, -0.561532701062491, -1.99394298852273, -0.564809074460413, -1.99377728857157, -0.572890552819328, -1.99338405289123, -0.592145129380278, -1.99243032200788, -0.636984669997328, -1.99006818024634, -0.736847768135986, -1.98395460754674, -0.937254399773507, -1.96648234775056, -1.25916802743759, -1.90547533499573, -1.65634142507340]
    self.secOrdS[:,5] = [0.999703892797184, 0.999976204887386, 0.998870670727159, 0.999909128094374, 0.997120121860118, 0.999767474120971, 0.993046464567097, 0.999433858332703, 0.983460327041947, 0.998626068054824, 0.961313456378720, 0.996627490637250, 0.912826163313502, 0.991466132704467, 0.820586489431352, 0.976806531598616, 0.705604525011879, 0.926817689762213, 0.755132573502125]
    self.secOrd60 = np.ndarray((4,6))
    self.secOrd60[:,0] = np.ones((4,))
    self.secOrd60[:,1] = [-1.999757336673731,-1.999768993687961,-1.999749127490619,-1.999776552841887]
    self.secOrd60[:,2] = np.ones((4,))
    self.secOrd60[:,3] = np.ones((4,))
    self.secOrd60[:,4] = [-1.996532024319651,-1.997182800632475,-1.999298467554908,-1.999437190667969]
    self.secOrd60[:,5] = [0.996818865786395,0.997377661196112,0.999576259089062,0.999638905524204]
    self.secOrd120 = np.ndarray((8,6))
    self.secOrd120[:,0] = np.ones((8,))
    self.secOrd120[:,2] = np.ones((8,))
    self.secOrd120[:,3] = np.ones((8,))
    self.secOrd120[:,1] = [-1.998966482169534,-1.999109073452369,-1.998984435104890,-1.999093323348585,-1.999018866633316,-1.999061504162885,-1.998959903818727,-1.999114708554225]
    self.secOrd120[:,4] = [-1.998699079356691,-1.998908288258769,-1.997997159246140,-1.998325873902175,-1.993780988613506,-1.994788307399206,-1.998895322518691,-1.999074937645729]
    self.secOrd120[:,5] = [0.999756408223522,0.999778948015547,0.999076067299233,0.999178572372702,0.994894772101863,0.995611105319687,0.999946326543450,0.999950995741130]
        #circuit.trigger(trigger='A',mode='pulse')
    self.eng = matlab.engine.start_matlab()       #Start matlab engine. Only used for Bayesian Adaptive Regression Splines for Density estimation.
    print('TDT Circuits Loaded')                 #We will only reach this if TDT circuits are loaded. An error will be thrown in TDT circuits do not compile.
    print('Learning on Ch: ' + str(self.circuitRZ2.get_tag('ChSel')))     #Confirmation message for user checks.
    self. max_episode_len = 1000          #RL episodic max length. Will probably end before hand. Determined by TD3.
    self.stdmin = 3.8                   #Minimum detection constant for spike detection
    self.stdmax = 50                  #Maximum detection constant for throwing out artifacts in spike detection.
    #self.detectWindow = [39:90]
  def step(self, stimSet):            #This determines what happens in each trial in the observation space. Setup step by step observations of spikes and the action space to take in each step.
    """
    This is the step command in which we adjust stimulation parameters and run the stimulation set.
    Inputs: stimSet = A vector containing the current action values. Has form:
                      stimSet = [NumPulses,INSAmpVal,TmBetPulses, PulseWidth]
    """
        #return self.obs, self.reward, self.done, {}
    stimSet = self.checkPulses(stimSet)                  #Make sure learned pulses aren't violating upper or lower bounds
    Waveform = self.genPulses(stimSet[0],stimSet[1],stimSet[3],stimSet[2])          #Generate waveform to be sent to stimulator
    #dataStore = np.zeros((self.numStimTrials+1,self.datalen))
    dataStore = {}
    for ck in range(self.numStimTrials):
        self.circuitRX7.trigger(trigger=1)              #Set Z-bus triggers for TDT hardware.
        self.circuitRZ2.trigger(trigger=1)
        laser_buffer = self.circuitRX7.get_buffer('Waveform', 'w')       #Prepare the buffer 
        sleep(1)                       #Give some time for the buffer to clear for writing. Variable based on computer. Since timing is self contained in its own thread, will not contradict environment.
        Waveform = Waveform[0:self.datalen-1] #This just gets it in "pythonic" form.
        laser_buffer.write(Waveform)       #Send action space waveform to the laser.
        SpkBuf = self.circuitRZ2.get_buffer('SpkDat', 'r')    #Record resulting multiunit activity.
        data = SpkBuf.acquire('A', 'running', False)
        dataStore[str(ck)] = data[0][0][:]          #Store the raw results.
    filtSignal = self.sigFilter(dataStore)          #Filter data for multiunit activity.
    spks = self.getSpikes(filtSignal)
    psth,psthKey = self.getPSTH(spks)               #Calculate PSTH online.
    try:
      psth = self.densityEstPSTH(psth)              #This should work. Inform user if there is issues with density estimation.
    except:
      print('barsP error')
    reward = self.getScores(self.targetPSTH,psth)    #Calculate the reward for taking action a in observation state o.
    self.reward = reward
    self.obs = psth 
    self.obsStore[str(self.counter)] = psth          #TD3 does this internally, but we need to store results explicitly for offline analysis.
    self.rewardHist[str(self.counter)] = reward
    self.paramStore[str(self.counter)] = Waveform
    self.spkStore[str(self.counter)] = spks
    self.counter = self.counter + 1
    self.checkEnd()                                   #Check if TD3 has detemined an end of an episode.
    print(self.reward)                               #Just to let you know what the reward is in the command line.
    return self.obs, self.reward, self.done, {}
  def reset(self):                                   #Reset state to reinitialize an episode.
    self.done = False
    self.reward = 0
    self.obs = 0
  def render(self, mode='human', close=False):
    """
    This function is defined by the user as to how they would like the render of the environment to be displayed.
    In our case, we are interested in comparing target response to measured response. So this amounts to just a few plots.
    However, you can be as creative on this as you want.
    -- mode: Human. Keyword for Gym. Basically view as a human with eyes. You'll always use this.
    -- close: Setting to True will autoclose the figure. Generally, this causes the figure to close prematurely, so I prefer to do it myself.
    """
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
        plt.plot(self.targetPSTH) #Plot target PSTH
        ax2 = plt.subplot(212)
        plt.plot(self.obs)      #Plot observed PSTH.
        plt.ylim([0, 20])
        plt.pause(0.001)
        plt.draw()

    print('Current Reward: ' + str(self.reward))
  def genPulses(self,numPulses,PulseVec,PWVec,ISIvec):
    """
    This function generates arbitrary pulse waveforms to learn arbitrary pulse functions for the laser system.
    Will need to be configured for whatever stimulator is used. Auditory, for example, will be changed to derive
    auditory stimuli to feed to the TDT system. This should be similar to how it is done in OpenEx.
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
    """
    This function checks if the stimulation parameters are above the user thresholds, then truncates them to max or min respectively. Necessary to
    prevent neural damage.
    """
    if stimSet[0] < 1:
        stimSet[0] = 1
    stimSet[0] = int(stimSet[0])
    if stimSet[1] < self.stimLow:
        stimSet[1] = self.stimLow
    if stimSet[1] > self.stimHigh:
        stimSet[1] = self.stimHigh
    if stimSet[2] < self.tBetPulsesLim:
        stimSet[2] = self.tBetPulsesLim
    if stimSet[3] < self.PWLow:
        stimSet[3] = self.PWLow
    if stimSet[3] > self.PWHigh:
        stimSet[3] = self.PWHigh
    return stimSet

  def getPSTH(self,data):
    """
    This is a simple online PSTH calculator.
    """
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
    """
    Here is where you define your reward function. This can take the form of any reasonable reward function and is customized completely 
    to the experiment.
    """
    if scoreType == 1:
       MSE = mean_squared_error(targetPSTH[39:90],runPSTH[39:90])
       funcHolder = (1-np.power(MSE,0.5))
       corCoeff = np.corrcoef(targetPSTH[39:90],runPSTH[39:90])
       corCoeff = corCoeff[0,1]
       reward = funcHolder*corCoeff
    elif scoreType == 2:
       reward = mean_squared_error(targetPSTH[39:90],runPSTH[39:90])
    else:
       raise ValueError("Score Type is not chosen")
    return reward

  def checkEnd(self):
    """
    This is a helper function to TD3 to tell RL when to start a new episode. I've modified this to also dump recorded data as needed.
    """
    if self.reward <= self.rewardTarget:
      self.done = True
    if self.done == True:
        print('PSTH Matched')
        with open('obs.pickle', 'wb') as handle:
            pickle.dump(self.obsStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('stim.pickle', 'wb') as handle:
            pickle.dump(self.paramStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('reward.pickle', 'wb') as handle:
            pickle.dump(self.rewardHist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('spks.pickle', 'wb') as handle:
            pickle.dump(self.spkStore, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('obsFinal.pickle', 'wb') as handle:
            pickle.dump(self.obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('rewardFinal.pickle', 'wb') as handle:
            pickle.dump(self.reward, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def spkDetect(self,data,detectEdge='pos'):
    """
    This is a simple threshold online spike detector for multiunit activity.
    """
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
    """
    This is a helper function to do signal filtering and conditioning after recording.
    """
    filtSig = {}
    for rr in range(self.numStimTrials):
      curVal = inSignal[str(rr)]
      curfiltSig = sig.sosfiltfilt(self.secOrdS,curVal)
      curfiltSig = sig.sosfiltfilt(self.secOrd60,curfiltSig)
      curfiltSig = sig.sosfiltfilt(self.secOrd120,curfiltSig)
      filtSig[str(rr)] = curfiltSig
    return filtSig

  def getSpikes(self,data):
    """
    This is a helper function for running multiunit threshold detection.
    """
    spks = {}
    for tt in range(self.numStimTrials):
      curDat = data[str(tt)]
      curSpks = self.spkDetect(curDat)
      spks[str(tt)] = curSpks
    return spks
  
  def densityEstPSTH(self,PSTH):
    """
    This is a helper function to run the BARS density estimation using the Matlab engine.
    """
    numtrials = self.numStimTrials
    PSTH = PSTH.tolist()         #Matlab vars do not like numpy arrays
    dat = matlab.double(PSTH)    #Convert to matlab data struct
    fit1 = self.eng.barsP(dat,matlab.double([0,1.]),numtrials)
    fit1 = np.asarray(fit1)
    fit1 = np.squeeze(fit1)
    return fit1



