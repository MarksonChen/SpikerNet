import matlab.engine  
import numpy as np 
import matplotlib.pyplot as plt
import pickle
targetPSTH = np.array([2, 4, 6,	4,	8,	3,	2,	4,	3,	5,	2,	3,	1,	5,	5,	2,	4,	4,	3,	5,	3,	2,	1,	1,	5,	4,	3,	4,	2,	3,	4,	5,	4,	5,	4,	5,	3,	5,	5,	6,	7,	4,	6,	1,	7,	10,	21,	18,	11,	11,	11,	10,	3,	5,	8,	11,	9,	8,	9,	11,	8,	12,	8,	3,	9,	5,	4,	3,	7,	11,	8,	4,	6,	7,	6,	3,	8,	7,	8,	10,	7,	5,	9,	6,	10,	5,	5,	4,	4,	4,	5,	5,	3,	8,	6,	2,	5,	6,	5,	3,	9,	5,	4,	2,	5,	5,	3,	4,	4,	4,	3,	1,	1,	3,	5,	0,	3,	2,	3,	5,	5,	4,	3,	5,	5,	2,	5,	3,	1,	1,	3,	5,	2,	2,	2,	1,	4,	3,	2,	0,	2,	4,	0,	1,	5,	1,	1,	4,	3,	3,	3,	2,	2,	3,	4,	1,	3,	3,	2,	2,	3,	5,	1,	3,	5,	0,	2,	3,	3,	4,	2,	4,	1,	2,	3,	1,	5,	5,	3,	1,	1,	1,	1,	5,	3,	1,	2,	3,	1,	2,	2,	3,	0,	3,	5,	1,	2,	3,	2,	2])
targetPSTH = targetPSTH.tolist()
dat = matlab.double(targetPSTH)
eng = matlab.engine.start_matlab()
fit1 = eng.barsP(dat,matlab.double([0,1.]),10)
fit1 = np.asarray(fit1)
fit1 = np.squeeze(fit1)
print(fit1)
fit2 = fit1
margin= 3*np.std(fit1)
margWhere = np.where(fit1<margin)
#fit1[margWhere] = 0
plt.plot(np.linspace(0,199,200),fit2)
plt.show()
fit1 = np.asarray(fit1)
print(fit1)
#with open('trialPSTH.pickle','wb') as f:
    #pickle.dump(fit1 ,f)