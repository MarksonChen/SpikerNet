import numpy as np
from dbs import cDBS

class MFM(object):
    def __init__(self,**kwargs):
        self.t = 0                  #internal time counter
        self.i = 0                  #internal index

        self._load_params(kwargs)
        self._set_MFM_params()
        self._set_DBS()

        self.S = np.zeros((self.params['N'],20))
        self.S[0,:] = [ 43.74102506,  -1.15197439,    6.96276347,  -22.25852135,    7.19671392,
                        -28.57548512,  17.26916297,  132.89911127,    9.71319243,   67.69191101,
                        9.57769785,  -21.11198645,    5.33943222,  -22.62375016,    0.54172422,
                        -22.23467637,   6.76173506,   143.5386694,    7.49756915,   23.59983148]

        self.memory = {
            'stim'  : np.zeros(self.params['N']),
        }

    def _load_params(self,kwargs):
        def process_kwargs(kwargs):
            for key,value in kwargs.items():
                if key not in self.params:
                    print('Invalid keyword argument {}'.format(key))
                else:
                    value = type(self.params[key])(value) #cast value to that of params[key]
                    self.params[key] = value

        self.params = {}

        self.params['verbose'] = True

        #General parameters
        self.params['dt']         = 1e-3        # s
        self.params['stim_start'] = 0.0         # s
        self.params['tstop']      = 50.0        # s
        self.params['RunID']      = -1

        #DD parameters
        self.params['DD'] = True

        #Stimulation parameters
        self.params['stim_target'] = 'STN'
        self.params['Cm']          = 1e-4       # F

        #cDBS parameters
        self.params['cDBS']        = False
        self.params['cDBS_f']      = 130.       # (Hz)
        self.params['cDBS_amp']    = 3.0        # (mV)
        self.params['cDBS_width']  = 60.        # (us)
        self.params['state_target'] = 'p1'

        process_kwargs(kwargs)

        #Set parameters dependent on other parameters
        self.params['N'] = int(np.ceil(self.params['tstop']/self.params['dt']))
        self.params['fs'] = 1./self.params['dt']

    def _set_MFM_params(self):
        self.phin = 15
        self.noiseAmp = 0.03
        self.re = 80 #mm
        self.gammae = 125 #s^-1
        self.alpha = 160 #s^-1
        self.beta = 640 #s^-1
        self.gammasq=self.gammae^2
        self.alphabeta=self.alpha*self.beta
        self.aPb = (self.alpha+self.beta)#/alphabeta
        self.alphagamma=self.alpha*self.gammae

        #Axonal Delays: from second to first (1st population type is postsynaptic)
        #(ms)
        self.taues   = int(0.035/self.params['dt'])
        self.tauis   = int(0.035/self.params['dt'])
        self.taud1e  = int(0.002/self.params['dt'])
        self.taud2e  = int(0.002/self.params['dt'])
        self.taud1s  = int(0.002/self.params['dt'])
        self.taud2s  = int(0.002/self.params['dt'])
        self.taup1d1 = int(0.001/self.params['dt'])
        self.taup1p2 = int(0.001/self.params['dt'])
        self.taup1ST = int(0.001/self.params['dt'])
        self.taup2d2 = int(0.001/self.params['dt'])
        self.taup2ST = int(0.001/self.params['dt'])
        self.tauSTe  = int(0.001/self.params['dt'])
        self.tauSTp2 = int(0.001/self.params['dt'])
        self.tause   = int(0.050/self.params['dt'])
        self.taure   = int(0.050/self.params['dt'])
        self.tausp1  = int(0.003/self.params['dt'])
        self.tausr   = int(0.002/self.params['dt'])
        self.taurs   = int(0.002/self.params['dt'])
        self.taud2d1 = int(0.001/self.params['dt'])

        #Threshold spread (mV)
        self.sigmaprime = 3.8

        #Connection strength (mVs)
        self.vee   =  1.6   #1.6 1.4 is DD
        self.vie   =  1.6   #1.6 1.4 is DD
        self.vei   = -1.9   #-1.9-1.6 is DD
        self.vii   = -1.9   #-1.9 -1.6 is DD
        self.ves   =  0.4
        self.vis   =  0.4
        self.vd1e  =  1.0   #1 #1 is normal, .5 is DD
        self.vd1d1 = -0.3
        self.vd1s  =  0.1
        self.vd2e  =  0.7   #0.7 1.4 is DD
        self.vd2d2 = -0.3
        self.vd2s  =  0.05
        self.vp1d1 = -0.1
        self.vp1p2 = -0.03
        self.vp1ST =  0.3
        self.vp2d2 = -0.3   #-0.3 -0.5 is DD
        self.vp2p2 = -0.1   #-0.1 -0.07 is DD
        self.vp2ST =  0.3
        self.vSTe  =  0.1
        self.vSTp2 = -0.04
        self.vse   =  0.8
        self.vsp1  = -0.03
        self.vsr   = -0.4
        self.vsn   =  0.5
        self.vre   =  0.15
        self.vrs   =  0.03
        self.vd2d1 =  0

        #Sigmoids
        #Maximum Firing Rates (s^-1)
        self.Qe = 300
        self.Qi = 300
        self.Qd1 = 65
        self.Qd2 = 65
        self.Qp1 = 250
        self.Qp2 = 300
        self.QST = 500
        self.Qs = 300
        self.Qr = 500

        #Firing Thresholds (mV)
        self.thetae = 14
        self.thetai = 14
        self.thetad1 = 19
        self.thetad2 = 19
        self.thetap1 = 10
        self.thetap2 = 9     #9 8 is DD
        self.thetaST = 10    #10 9 is DD
        self.thetas = 13
        self.thetar = 13

        self.phie     = 0
        self.phie_dot = 1
        self.Ve       = 2
        self.Ve_dot   = 3
        self.Vi       = 4
        self.Vi_dot   = 5
        self.Vd1      = 6
        self.Vd1_dot  = 7
        self.Vd2      = 8
        self.Vd2_dot  = 9
        self.Vp1      = 10
        self.Vp1_dot  = 11
        self.Vp2      = 12
        self.Vp2_dot  = 13
        self.VST      = 14
        self.VST_dot  = 15
        self.Vs       = 16
        self.Vs_dot   = 17
        self.Vr       = 18
        self.Vr_dot   = 19

        self.struct={'e'   :2,
                     'i'   :4,
                     'd1'  :6,
                     'd2'  :8,
                     'p1'  :10,
                     'p2'  :12,
                     'STN' :14,
                     's'   :16,
                     'r'   :18}

        #Set parkinsonian parameters
        if self.params['DD']:
            self.vee = 1.4
            self.vie = 1.4
            self.vei = -1.6
            self.vii = -1.6
            self.vd1e = 0.5
            self.vd2e = 1.4
            self.vp2d2 = -0.5
            self.vp2p2 = -0.07
            self.thetap2 = 8
            self.thetaST = 9

    def _set_DBS(self):
        if self.params['cDBS']:
            self.cDBS = cDBS(dt       = self.params['dt'],
                             f        = self.params['cDBS_f'],
                             stim_amp = self.params['cDBS_amp'],
                             width    = self.params['cDBS_width'],
                             tstart   = self.params['stim_start'])
        else:
            self.cDBS = None

    def advance(self):
        def sigmoid(V,Q,theta):
            return Q/(1+np.exp(-(V-theta)/3.8))

        i = self.i

        dSdt = np.zeros(20)

        dSdt[self.phie]= self.S[i,self.phie_dot]
        dSdt[self.phie_dot] = self.gammasq*(sigmoid(self.S[i,self.Ve], self.Qe, self.thetae)- \
                                            self.S[i,self.phie])-2*self.gammae*self.S[i,self.phie_dot]

        dSdt[self.Ve] = self.S[i,self.Ve_dot]
        dSdt[self.Ve_dot] = self.alphagamma*(self.vee*self.S[i,self.phie]+ \
                                             self.vei*sigmoid(self.S[i,self.Vi],self.Qi, self.thetai)+ \
                                             self.ves*sigmoid(self.S[i-self.taues,self.Vs], self.Qs, self.thetas)- \
                                             self.S[i,self.Ve])- \
                            self.aPb*self.S[i,self.Ve_dot]

        dSdt[self.Vi] = self.S[i,self.Vi_dot]
        dSdt[self.Vi_dot] = self.alphagamma*(self.vii*sigmoid(self.S[i,self.Vi], self.Qi, self.thetai)+ \
                                             self.vie*self.S[i,self.phie]+ \
                                             self.vis*sigmoid(self.S[i-self.tauis,self.Vs], self.Qs, self.thetas)- \
                                             self.S[i,self.Vi])- \
                            self.aPb*self.S[i,self.Vi_dot]

        dSdt[self.Vd1] = self.S[i,self.Vd1_dot]
        dSdt[self.Vd1_dot] = self.alphabeta*(self.vd1e*self.S[i-self.taud1e,self.phie]+ \
                                             self.vd1s*sigmoid(self.S[i-self.taud1s,self.Vs], self.Qs, self.thetas)+ \
                                             self.vd1d1*sigmoid(self.S[i,self.Vd1], self.Qd1, self.thetad1)- \
                                             self.S[i,self.Vd1])- \
                             self.aPb*self.S[i,self.Vd1_dot] #Add in SNc

        dSdt[self.Vd2] = self.S[i,self.Vd2_dot]
        dSdt[self.Vd2_dot] = self.alphabeta*(self.vd2e*self.S[i-self.taud2e, self.Ve]+ \
                                             self.vd2d1*sigmoid(self.S[i-self.taud2d1,self.Vd1], self.Qd1, self.thetad1)+ \
                                             self.vd2s*sigmoid(self.S[i-self.taud2s,self.Vs], self.Qs, self.thetas)+ \
                                             self.vd2d2*sigmoid(self.S[i,self.Vd2], self.Qd2, self.thetad2)- \
                                             self.S[i,self.Vd2])- \
                             self.aPb*self.S[i,self.Vd2_dot] #Add in the SNc

        dSdt[self.Vp1] = self.S[i,self.Vp1_dot]
        dSdt[self.Vp1_dot] = self.alphabeta*(self.vp1d1*sigmoid(self.S[i-self.taup1d1,self.Vd1], self.Qd1, self.thetad1)+ \
                                             self.vp1p2*sigmoid(self.S[i-self.taup1p2,self.Vp2], self.Qp2, self.thetap2)+ \
                                             self.vp1ST*sigmoid(self.S[i-self.taup1ST,self.VST], self.QST, self.thetaST)- \
                                             self.S[i,self.Vp1])- \
                             self.aPb*self.S[i,self.Vp1_dot]

        dSdt[self.Vp2] = self.S[i,self.Vp2_dot]
        dSdt[self.Vp2_dot] = self.alphabeta*(self.vp2d2*sigmoid(self.S[i-self.taup2d2,self.Vd2], self.Qd2, self.thetad2)+ \
                                             self.vp2p2*sigmoid(self.S[i,self.Vp2], self.Qp2, self.thetap2)+ \
                                             self.vp2ST*sigmoid(self.S[i-self.taup2ST,self.VST], self.QST, self.thetaST)- \
                                             self.S[i,self.Vp2])- \
                             self.aPb*self.S[i,self.Vp2_dot]

        dSdt[self.VST] = self.S[i,self.VST_dot]
        dSdt[self.VST_dot] = self.alphabeta*(self.vSTp2*sigmoid(self.S[i-self.tauSTp2,self.Vp2], self.Qp2, self.thetap2)+ \
                                             self.vSTe*self.S[i-self.tauSTe,self.phie]- \
                                             self.S[i,self.VST])- \
                             self.aPb*self.S[i,self.VST_dot]

        dSdt[self.Vs] = self.S[i,self.Vs_dot]
        dSdt[self.Vs_dot] = self.alphabeta*(self.vsp1*sigmoid(self.S[i-self.tausp1,self.Vp1], self.Qp1, self.thetap1)+ \
                                            self.vse*self.S[i-self.tause,self.phie]+ \
                                            self.vsr*sigmoid(self.S[i-self.tausr,self.Vr], self.Qr, self.thetar)+ \
                                            self.phin- \
                                            self.S[i,self.Vs])- \
                            self.aPb*self.S[i,self.Vs_dot]

        dSdt[self.Vr] = self.S[i,self.Vr_dot]
        dSdt[self.Vr_dot] = self.alphabeta*(self.vre*self.S[i-self.taure,self.phie]+ \
                                            self.vrs*sigmoid(self.S[i-self.taurs,self.Vs], self.Qs, self.thetas)- \
                                            self.S[i,self.Vr])- \
                            self.aPb*self.S[i,self.Vr_dot]

        #DBS
        #====================================================================================
        if self.params['cDBS']:
            cDBS_C = self.cDBS.advance()
            self.S[i,self.struct[self.params['stim_target']]] += cDBS_C/self.params['Cm']
            if cDBS_C != 0: self.memory['stim'][i+1] = cDBS_C

        #Advance
        #====================================================================================
        self.S[i+1,:] = self.S[i,:]+self.params['dt']*dSdt

        #Noise
        #====================================================================================
        self.S[i+1,self.Ve] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qe*(1-sigmoid(self.S[i+1,self.Ve], 1, self.thetae))*sigmoid(self.S[i,self.Ve], 1, self.thetae)

        self.S[i+1,self.Vi] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qi*(1-sigmoid(self.S[i+1,self.Vi], 1, self.thetai))*sigmoid(self.S[i,self.Vi], 1, self.thetai)

        self.S[i+1,self.Vd1] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qd1*(1-sigmoid(self.S[i+1,self.Vd1], 1, self.thetad1))*sigmoid(self.S[i,self.Vd1], 1, self.thetad1)

        self.S[i+1,self.Vd2] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qd2*(1-sigmoid(self.S[i+1,self.Vd2], 1, self.thetad2))*sigmoid(self.S[i,self.Vd2], 1, self.thetad2)

        self.S[i+1,self.Vp1] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qp1*(1-sigmoid(self.S[i+1,self.Vp1], 1, self.thetap1))*sigmoid(self.S[i,self.Vp1], 1, self.thetap1)

        self.S[i+1,self.Vp2] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qp2*(1-sigmoid(self.S[i+1,self.Vp2], 1, self.thetap2))*sigmoid(self.S[i,self.Vp2], 1, self.thetap2)

        self.S[i+1,self.VST] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.QST*(1-sigmoid(self.S[i+1,self.VST], 1, self.thetaST))*sigmoid(self.S[i,self.VST], 1, self.thetaST)

        self.S[i+1,self.Vs] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qs*(1-sigmoid(self.S[i+1,self.Vs], 1, self.thetas))*sigmoid(self.S[i,self.Vs], 1, self.thetas)

        self.S[i+1,self.Vr] += self.noiseAmp*np.random.normal(0,1)*np.sqrt(self.params['dt'])*self.Qr*(1-sigmoid(self.S[i+1,self.Vr], 1, self.thetar))*sigmoid(self.S[i,self.Vr], 1, self.thetar)

        self.i += 1

    def run(self):
        while self.i < self.params['N'] - 1:
            self.advance()

    def getPXX(self):
        from scipy import signal

        t = np.arange(self.params['N']) * self.params['dt']
        f,Pxx = signal.welch(self.S[:,self.struct[self.params['state_target']]],1/self.params['dt'],nperseg=2048)
        Pxx = 10*np.log10(Pxx)
        return Pxx

    def getLFP(self):
        return self.S[:,self.struct[self.params['state_target']]]


