import numpy as np
from sklearn.metrics import mean_squared_error
def getScores(self,targetPSTH,runPSTH,scoreType=1):
    if scoreType == 1:
       MSE = mean_squared_error(targetPSTH,runPSTH)
       print('MSE' + str(MSE))
       funcHolder = (1-np.power(MSE,0.5))
       print('funcHolder' + str(funcHolder))
       corCoeff = np.corrcoef(targetPSTH,runPSTH)
       print('Corr' + str(corCoeff))
       corCoeff = corCoeff[0,1]
       reward = funcHolder*corCoeff
    else:
       raise ValueError("Score Type is not chosen")
    return reward
corvals = np.linspace(-1,1,100)
MSEVals = np.linspace(0,4,100)
reward = np.zeros((len(MSEVals),len(corvals)))
for ck in range(len(MSEVals)):
    for bc in range(len(corvals)):
        MSE = MSEVals[ck]
        funcHolder = funcHolder = (1-np.power(MSE,.5))
        cor = corvals[bc]
        if cor >= 0:
            reward[ck,bc] = funcHolder*cor
        else:
            reward[ck,bc] = -1*funcHolder*cor
import matplotlib.pyplot as plt
X, Y = np.meshgrid(MSEVals, corvals)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, reward, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('MSE')
ax.set_ylabel('Correlation')
plt.show()