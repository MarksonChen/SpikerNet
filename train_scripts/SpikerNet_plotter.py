import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
obs1 = pd.read_pickle('C://CodeRepos//spikerNet//obsB.pickle')
reward1 = pd.read_pickle('C://CodeRepos//spikerNet//rewardB.pickle')
stim1 = pd.read_pickle('C://CodeRepos//spikerNet//stimB.pickle')
obs1Val = obs1['0']
target = pd.read_pickle('trialPSTHB.pickle')
f = pd.read_pickle('C://CodeRepos//spikerNet//Pxxf.pickle')
#plt.plot(f,target,f,obs1Val,'--')
#plt.show()



plt.plot(f[0:410],target[0:410],f[0:410],obs1Val[0:410],'--')

plt.show()
pdb.set_trace()
