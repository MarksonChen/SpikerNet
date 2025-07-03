import numpy as np
from omegaconf import DictConfig, OmegaConf
from dbs import cDBS


class Idx:
    phie, phie_dot, Ve, Ve_dot, Vi, Vi_dot, \
        Vd1, Vd1_dot, Vd2, Vd2_dot, \
        Vp1, Vp1_dot, Vp2, Vp2_dot, \
        VST, VST_dot, Vs, Vs_dot, Vr, Vr_dot = range(20)

class DelayIdx:
    taues, tauis, taud1e, taud2e, taud1s, taud2s, \
        taup1d1, taup1p2, taup1ST, taup2d2, taup2ST, \
        tauSTe, tauSTp2, tause, taure, tausp1, \
        tausr, taurs, taud2d1 = range(19)


class MFM:
    def __init__(self, config: DictConfig = None, **kwargs):
        self.t = 0
        self.i = 0

        # Load configuration (must exist)
        if config is None:
            config = OmegaConf.load("config/mfm_config.yaml")
        self.cfg = config

        self._process_kwargs(kwargs)
        self._set_MFM_params()
        self._set_derived_params()
        self._set_delays_vectorized()
        self._set_noise_params()
        self._set_DBS()

        self.S = np.zeros((self.cfg.simulation.N, 20))
        self.S[0, :] = self.cfg.initial_state
        self.memory = {'stim': np.zeros(self.cfg.simulation.N)}

    def _process_kwargs(self, kwargs):
        """Process keyword arguments for backward compatibility"""
        for key, value in kwargs.items():
            if key in ['dt', 'tstop', 'stim_start', 'verbose', 'DD', 'RunID']:
                self.cfg.simulation[key] = value
            elif key == 'cDBS':
                self.cfg.cDBS.cDBS = value
            elif key in ['cDBS_f', 'cDBS_amp', 'cDBS_width', 'stim_target', 'state_target']:
                self.cfg.cDBS[key] = value
            elif key in self.cfg.system:
                self.cfg.system[key] = value

    def _set_derived_params(self):
        """Set parameters derived from configuration"""
        self.cfg.simulation.N = int(np.ceil(self.cfg.simulation.tstop / self.cfg.simulation.dt))
        self.cfg.simulation.fs = 1.0 / self.cfg.simulation.dt

        # Cache frequently accessed values for performance
        self.dt = self.cfg.simulation.dt
        self.sqrt_dt = np.sqrt(self.dt)
        self.noise_factor = self.cfg.system.noiseAmp * self.sqrt_dt  # Cache combined noise factor
        self.Cm = self.cfg.system.Cm
        self.N = self.cfg.simulation.N
        self.fs = self.cfg.simulation.fs
        self.state_target_idx = self.struct[self.cfg.cDBS.state_target]
        self.stim_target_idx = None  # Will be set in _set_DBS if needed

    def _set_MFM_params(self):
        """Set MFM parameters from configuration"""
        # System parameters
        sys = self.cfg.system
        self.phin, self.noiseAmp, self.re = sys.phin, sys.noiseAmp, sys.re
        self.gammae, self.alpha, self.beta = sys.gammae, sys.alpha, sys.beta
        self.sigmaprime = sys.sigmaprime

        # Derived parameters
        self.gammasq = self.gammae ^ 2 # Don't Fix This
        self.alphabeta = self.alpha * self.beta
        self.aPb = self.alpha + self.beta
        self.alphagamma = self.alpha * self.gammae

        # Population parameters
        pop = self.cfg.populations
        self.Qe, self.Qi, self.Qd1, self.Qd2 = pop.Qe, pop.Qi, pop.Qd1, pop.Qd2
        self.Qp1, self.Qp2, self.QST, self.Qs, self.Qr = pop.Qp1, pop.Qp2, pop.QST, pop.Qs, pop.Qr
        self.thetae, self.thetai, self.thetad1, self.thetad2 = pop.thetae, pop.thetai, pop.thetad1, pop.thetad2
        self.thetap1, self.thetap2, self.thetaST = pop.thetap1, pop.thetap2, pop.thetaST
        self.thetas, self.thetar = pop.thetas, pop.thetar

        # Connection strengths - start with normal values
        for key, value in self.cfg.connections.items():
            setattr(self, key, value)

        # Apply Parkinsonian modifications if in DD state
        if self.cfg.simulation.DD:
            for key, value in self.cfg.parkinsonian.items():
                if key in ['thetap2', 'thetaST']:
                    # These are threshold modifications, handle separately
                    if key == 'thetap2':
                        self.thetap2 = value
                    elif key == 'thetaST':
                        self.thetaST = value
                else:
                    # These are connection strength modifications
                    setattr(self, key, value)

        # Structure mapping
        self.struct = {'e': Idx.Ve, 'i': Idx.Vi, 'd1': Idx.Vd1, 'd2': Idx.Vd2,
                       'p1': Idx.Vp1, 'p2': Idx.Vp2, 'STN': Idx.VST, 's': Idx.Vs, 'r': Idx.Vr}

    def _set_delays_vectorized(self):
        """Convert delay times to timesteps using vectorized approach"""
        delay_names = ['taues', 'tauis', 'taud1e', 'taud2e', 'taud1s', 'taud2s',
                       'taup1d1', 'taup1p2', 'taup1ST', 'taup2d2', 'taup2ST',
                       'tauSTe', 'tauSTp2', 'tause', 'taure', 'tausp1', 'tausr', 'taurs', 'taud2d1']

        delay_values = [getattr(self.cfg.delays, name) for name in delay_names]
        self.tau = (np.array(delay_values) / self.dt).astype(int)

    def _set_noise_params(self):
        """Set up vectorized noise parameters"""
        self._noise_idx = np.array([Idx.Ve, Idx.Vi, Idx.Vd1, Idx.Vd2,
                                    Idx.Vp1, Idx.Vp2, Idx.VST, Idx.Vs, Idx.Vr])
        self._Q = np.array([self.Qe, self.Qi, self.Qd1, self.Qd2,
                            self.Qp1, self.Qp2, self.QST, self.Qs, self.Qr])
        self._theta = np.array([self.thetae, self.thetai, self.thetad1, self.thetad2,
                                self.thetap1, self.thetap2, self.thetaST,
                                self.thetas, self.thetar])

    def _set_DBS(self):
        """Setup DBS stimulation"""
        if self.cfg.cDBS.cDBS:
            self.cDBS = cDBS(
                dt=self.cfg.simulation.dt,
                f=self.cfg.cDBS.cDBS_f,
                stim_amp=self.cfg.cDBS.cDBS_amp,
                width=self.cfg.cDBS.cDBS_width,
                tstart=self.cfg.simulation.stim_start
            )
            # Cache stimulation target index for performance
            self.stim_target_idx = self.struct[self.cfg.cDBS.stim_target]
        else:
            self.cDBS = None
            self.stim_target_idx = None

    def _sigmoid(self, V, Q, theta):
        """Sigmoid activation function"""
        return Q / (1 + np.exp(-(V - theta) / self.sigmaprime))

    def sigmoid(self, pop_name, V):
        """More concise notation"""
        if pop_name == 'e': return self._sigmoid(V, self.Qe, self.thetae)
        elif pop_name == 'i': return self._sigmoid(V, self.Qi, self.thetai)
        elif pop_name == 'd1': return self._sigmoid(V, self.Qd1, self.thetad1)
        elif pop_name == 'd2': return self._sigmoid(V, self.Qd2, self.thetad2)
        elif pop_name == 'p1': return self._sigmoid(V, self.Qp1, self.thetap1)
        elif pop_name == 'p2': return self._sigmoid(V, self.Qp2, self.thetap2)
        elif pop_name == 'ST': return self._sigmoid(V, self.QST, self.thetaST)
        elif pop_name == 's': return self._sigmoid(V, self.Qs, self.thetas)
        elif pop_name == 'r': return self._sigmoid(V, self.Qr, self.thetar)

    def second_order(self, x, y, I):
        """Helper for second-order ODE pattern: ẋ = y, ÿ = αβ(I - x) - (α+β)y"""
        return y, (self.alphabeta * (I - x) - self.aPb * y)

    def cortical_ode(self, phi, phi_dot, V):
        """Special case for cortical population with different dynamics"""
        return phi_dot, (self.gammasq * (self.sigmoid('e', V) - phi) - 2 * self.gammae * phi_dot)

    def apply_noise_vectorized(self, i):
        """Apply noise to all populations in one vectorized operation"""
        V_now = self.S[i, self._noise_idx]
        V_next = self.S[i + 1, self._noise_idx]
        sig_now = self._sigmoid(V_now, 1, self._theta)
        sig_next = self._sigmoid(V_next, 1, self._theta)
        ξ = np.random.normal(size=V_now.size)
        self.S[i + 1, self._noise_idx] += (self.noise_factor * ξ * self._Q *
                                           (1 - sig_next) * sig_now)

    def advance(self):
        """Advance simulation by one time step"""
        i = self.i
        dSdt = np.zeros(20)

        # Cortical populations (special dynamics)
        dSdt[Idx.phie], dSdt[Idx.phie_dot] = self.cortical_ode(
            self.S[i, Idx.phie], self.S[i, Idx.phie_dot], self.S[i, Idx.Ve])

        dSdt[Idx.Ve], dSdt[Idx.Ve_dot] = (
            self.S[i, Idx.Ve_dot],
            self.alphagamma * (
                    self.vee * self.S[i, Idx.phie] +
                    self.vei * self.sigmoid('i', self.S[i, Idx.Vi]) +
                    self.ves * self.sigmoid('s', self.S[i - self.tau[DelayIdx.taues], Idx.Vs]) -
                    self.S[i, Idx.Ve]
            ) - self.aPb * self.S[i, Idx.Ve_dot]
        )

        dSdt[Idx.Vi], dSdt[Idx.Vi_dot] = (
            self.S[i, Idx.Vi_dot],
            self.alphagamma * (
                    self.vii * self.sigmoid('i', self.S[i, Idx.Vi]) +
                    self.vie * self.S[i, Idx.phie] +
                    self.vis * self.sigmoid('s', self.S[i - self.tau[DelayIdx.tauis], Idx.Vs]) -
                    self.S[i, Idx.Vi]
            ) - self.aPb * self.S[i, Idx.Vi_dot]
        )

        # Striatal populations (second-order pattern)
        dSdt[Idx.Vd1], dSdt[Idx.Vd1_dot] = self.second_order(
            self.S[i, Idx.Vd1], self.S[i, Idx.Vd1_dot],
            self.vd1e * self.S[i - self.tau[DelayIdx.taud1e], Idx.phie] +
            self.vd1s * self.sigmoid('s', self.S[i - self.tau[DelayIdx.taud1s], Idx.Vs]) +
            self.vd1d1 * self.sigmoid('d1', self.S[i, Idx.Vd1])
        )

        dSdt[Idx.Vd2], dSdt[Idx.Vd2_dot] = self.second_order(
            self.S[i, Idx.Vd2], self.S[i, Idx.Vd2_dot],
            self.vd2e * self.S[i - self.tau[DelayIdx.taud2e], Idx.Ve] +
            self.vd2d1 * self.sigmoid('d1', self.S[i - self.tau[DelayIdx.taud2d1], Idx.Vd1]) +
            self.vd2s * self.sigmoid('s', self.S[i - self.tau[DelayIdx.taud2s], Idx.Vs]) +
            self.vd2d2 * self.sigmoid('d2', self.S[i, Idx.Vd2])
        )

        # Pallidal populations (second-order pattern)
        dSdt[Idx.Vp1], dSdt[Idx.Vp1_dot] = self.second_order(
            self.S[i, Idx.Vp1], self.S[i, Idx.Vp1_dot],
            self.vp1d1 * self.sigmoid('d1', self.S[i - self.tau[DelayIdx.taup1d1], Idx.Vd1]) +
            self.vp1p2 * self.sigmoid('p2', self.S[i - self.tau[DelayIdx.taup1p2], Idx.Vp2]) +
            self.vp1ST * self.sigmoid('ST', self.S[i - self.tau[DelayIdx.taup1ST], Idx.VST])
        )

        dSdt[Idx.Vp2], dSdt[Idx.Vp2_dot] = self.second_order(
            self.S[i, Idx.Vp2], self.S[i, Idx.Vp2_dot],
            self.vp2d2 * self.sigmoid('d2', self.S[i - self.tau[DelayIdx.taup2d2], Idx.Vd2]) +
            self.vp2p2 * self.sigmoid('p2', self.S[i, Idx.Vp2]) +
            self.vp2ST * self.sigmoid('ST', self.S[i - self.tau[DelayIdx.taup2ST], Idx.VST])
        )

        # STN (second-order pattern)
        dSdt[Idx.VST], dSdt[Idx.VST_dot] = self.second_order(
            self.S[i, Idx.VST], self.S[i, Idx.VST_dot],
            self.vSTp2 * self.sigmoid('p2', self.S[i - self.tau[DelayIdx.tauSTp2], Idx.Vp2]) +
            self.vSTe * self.S[i - self.tau[DelayIdx.tauSTe], Idx.phie]
        )

        # SNr and Reticular (second-order pattern)
        dSdt[Idx.Vs], dSdt[Idx.Vs_dot] = self.second_order(
            self.S[i, Idx.Vs], self.S[i, Idx.Vs_dot],
            self.vsp1 * self.sigmoid('p1', self.S[i - self.tau[DelayIdx.tausp1], Idx.Vp1]) +
            self.vse * self.S[i - self.tau[DelayIdx.tause], Idx.phie] +
            self.vsr * self.sigmoid('r', self.S[i - self.tau[DelayIdx.tausr], Idx.Vr]) +
            self.phin
        )

        dSdt[Idx.Vr], dSdt[Idx.Vr_dot] = self.second_order(
            self.S[i, Idx.Vr], self.S[i, Idx.Vr_dot],
            self.vre * self.S[i - self.tau[DelayIdx.taure], Idx.phie] +
            self.vrs * self.sigmoid('s', self.S[i - self.tau[DelayIdx.taurs], Idx.Vs])
        )

        # DBS stimulation
        if self.cDBS is not None:
            cDBS_C = self.cDBS.advance()
            self.S[i, self.stim_target_idx] += cDBS_C / self.Cm
            if cDBS_C != 0:
                self.memory['stim'][i + 1] = cDBS_C

        # Euler integration
        self.S[i + 1, :] = self.S[i, :] + self.dt * dSdt

        self.apply_noise_vectorized(i)

        self.i += 1

    def run(self):
        """Run the complete simulation"""
        while self.i < self.N - 1:
            self.advance()

    @property
    def power_spectrum(self):
        """Compute power spectral density (was getPXX)"""
        from scipy import signal
        f, Pxx = signal.welch(self.S[:, self.state_target_idx], self.fs, nperseg=2048)
        return 10 * np.log10(Pxx)

    @property
    def lfp(self):
        """Get local field potential (was getLFP)"""
        return self.S[:, self.state_target_idx]
