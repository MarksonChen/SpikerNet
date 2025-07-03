from __future__ import annotations
import numpy as np
import yaml
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Sequence, Tuple
from dbs import cDBS

# --------------------------------------------------------------------------- #
# 1.  Enumerations – no more manual indices
# --------------------------------------------------------------------------- #
class Var(IntEnum):
    """All state variables and their first derivatives (0‑based)."""
    phi_e, phi_e_dot, V_e, V_e_dot, V_i, V_i_dot, V_d1, V_d1_dot, \
        V_d2, V_d2_dot, V_p1, V_p1_dot, V_p2, V_p2_dot, V_STN, V_STN_dot, \
        V_s, V_s_dot, V_r, V_r_dot = range(20)


# Map names used in the YAML to Var indices
POP_IDX = {
    'e':    Var.phi_e,
    'i':    Var.V_i,
    'd1':   Var.V_d1,
    'd2':   Var.V_d2,
    'p1':   Var.V_p1,
    'p2':   Var.V_p2,
    'STN':  Var.V_STN,
    's':    Var.V_s,
    'r':    Var.V_r,
}


# --------------------------------------------------------------------------- #
# 2.  Parameter handling
# --------------------------------------------------------------------------- #
@dataclass
class ParameterSet:
    """Holds *all* user‑editable parameters."""
    raw: Dict[str, Any]

    # Convenience properties --------------------------------------------------
    @property
    def dt(self) -> float:         return self.raw['general']['dt']
    @property
    def tstop(self) -> float:      return self.raw['general']['tstop']
    @property
    def n_steps(self) -> int:      return int(np.ceil(self.tstop / self.dt))
    @property
    def fs(self) -> float:         return 1.0 / self.dt
    @property
    def disease(self) -> bool:     return self.raw['disease_model']['dopaminergic_depletion']
    @property
    def dbs_cfg(self) -> Dict:     return self.raw['dbs']

    # Loader ------------------------------------------------------------------
    @staticmethod
    def load(path: str | None = None, overrides: Dict[str, Any] | None = None) -> "ParameterSet":
        with open(path or "mfm_default.yaml", "r") as fh:
            data = yaml.safe_load(fh)
        overrides = overrides or {}
        # A *very* small, shallow merge so one can call `MFM(param_overrides=…)`
        for k, v in overrides.items():
            data.setdefault(k.split('.', 1)[0], {}).update(v)   # 1‑level deep
        return ParameterSet(data)


# --------------------------------------------------------------------------- #
# 3.  The actual model
# --------------------------------------------------------------------------- #
class MFM:
    """Minimal‑change refactor of the original Mattia‑Filippov model."""
    def __init__(self, param_file: str | None = None, **overrides):
        self.p = ParameterSet.load(param_file, overrides)
        self.i: int = 0                             # integration pointer
        self.t: float = 0.0                         # current time (s)

        # Pre‑compute secondary constants
        self.gamma_sq   = self.p.raw['gamma_e'] ** 2
        self.alpha_beta = self.p.raw['alpha'] * self.p.raw['beta']
        self.alpha_gamma = self.p.raw['alpha'] * self.p.raw['gamma_e']
        self.a_plus_b   = self.p.raw['alpha'] + self.p.raw['beta']

        # State tensor --------------------------------------------------------
        self.S = np.zeros((self.p.n_steps, len(Var)), dtype=float)
        self.S[0] = np.asarray(self.p.raw['initial_state'])

        # Fast helpers --------------------------------------------------------
        self._noise_table: Tuple[Tuple[Var, str], ...] = (
            (Var.V_e,  'e'  ), (Var.V_i,   'i'),
            (Var.V_d1, 'd1' ), (Var.V_d2,  'd2'),
            (Var.V_p1, 'p1' ), (Var.V_p2,  'p2'),
            (Var.V_STN,'STN'), (Var.V_s,   's' ),
            (Var.V_r,  'r'  ),
        )
        self._sigmoid_cache: Dict[str, Tuple[float, float]] = {
            name: (self.p.raw['sigmoid_Q_s_inv'][name], self.p.raw['thresholds_mV'][name])
            for name in self.p.raw['sigmoid_Q_s_inv']
        }

        # Parkinsonian tweaks -------------------------------------------------
        if self.p.disease:
            c = self.p.raw['connections_mV']
            θ = self.p.raw['thresholds_mV']
            c.update(dict(vee=1.4, vie=1.4, vei=-1.6, vii=-1.6,
                          vd1e=0.5, vd2e=1.4, vp2d2=-0.5, vp2p2=-0.07))
            θ.update(dict(p2=8, STN=9))

        # Pre‑compute discrete delays (samples) ------------------------------
        ms_to_step = lambda ms: int(np.round((ms * 1e-3) / self.p.dt))
        self.tau: Dict[str, int] = {name: ms_to_step(v)
                                    for name, v in self.p.raw['delays_ms'].items()}

        # DBS (kept minimal on purpose) -------------------------------------
        self.closed_loop = self.p.dbs_cfg['closed_loop']
        if self.closed_loop:
            self.cdbs = cDBS(dt=self.p.dt,
                             f = self.p.dbs_cfg['frequency'],
                             stim_amp = self.p.dbs_cfg['amplitude'],
                             width = self.p.dbs_cfg['width'],
                             tstart = self.p.raw['general']['stim_start'])
            self.stim_trace = np.zeros(self.p.n_steps)
        else:
            self.cdbs = None

    # --------------------------------------------------------------------- #
    #  Small utilities
    # --------------------------------------------------------------------- #
    def _sigmoid(self, V: float | np.ndarray, pop: str) -> np.ndarray:
        """Same shape as `V` (vectorised)."""
        Q, θ = self._sigmoid_cache[pop]
        σ = 1.0 / (1.0 + np.exp(-(V - θ) / self.p.raw['sigma_prime']))
        return Q * σ

# ------------------------------------------------------------------------- #
#  Replace the stub in class MFM with this implementation                   #
# ------------------------------------------------------------------------- #
def _rhs(self, S: np.ndarray) -> np.ndarray:
    """
    Compute dS/dt for the current integration index ``self.i``.

    Parameters
    ----------
    S : (20,) ndarray
        State vector at t = self.t.  Indices are given by ``Var`` enum.

    Returns
    -------
    dS : (20,) ndarray
        Time-derivative dS/dt evaluated at the same instant.
    """
    dS = np.zeros_like(S)

    # Short-hands -----------------------------------------------------------
    c   = self.p.raw['connections_mV']          # synaptic efficacies
    sig = self._sigmoid                         # local alias
    d   = self._delay                           # delayed state helper

    # ------------------------------------------------------------------ #
    #  φ-e (excitatory cortical firing-rate)                             #
    # ------------------------------------------------------------------ #
    dS[Var.phi_e]     = S[Var.phi_e_dot]
    dS[Var.phi_e_dot] = (
            self.gamma_sq * (sig(S[Var.V_e], 'e') - S[Var.phi_e])
            - 2 * self.p.raw['gamma_e'] * S[Var.phi_e_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-e  (cortical excitatory membrane potential)                     #
    # ------------------------------------------------------------------ #
    dS[Var.V_e] = S[Var.V_e_dot]
    dS[Var.V_e_dot] = (
            self.alpha_gamma *
            (
                    c['vee'] * S[Var.phi_e] +
                    c['vei'] * sig(S[Var.V_i], 'i') +
                    c['ves'] * sig(d(Var.V_s,  'tau_es'), 's') -
                    S[Var.V_e]
            )
            - self.a_plus_b * S[Var.V_e_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-i  (cortical inhibitory)                                        #
    # ------------------------------------------------------------------ #
    dS[Var.V_i] = S[Var.V_i_dot]
    dS[Var.V_i_dot] = (
            self.alpha_gamma *
            (
                    c['vii'] * sig(S[Var.V_i], 'i') +
                    c['vie'] * S[Var.phi_e] +
                    c['vis'] * sig(d(Var.V_s,  'tau_is'), 's') -
                    S[Var.V_i]
            )
            - self.a_plus_b * S[Var.V_i_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-d1 (direct-pathway MSNs)                                         #
    # ------------------------------------------------------------------ #
    dS[Var.V_d1] = S[Var.V_d1_dot]
    dS[Var.V_d1_dot] = (
            self.alpha_beta *
            (
                    c['vd1e']  * d(Var.phi_e, 'tau_d1e') +
                    c['vd1s']  * sig(d(Var.V_s, 'tau_d1s'), 's') +
                    c['vd1d1'] * sig(S[Var.V_d1], 'd1') -
                    S[Var.V_d1]
            )
            - self.a_plus_b * S[Var.V_d1_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-d2 (indirect-pathway MSNs)                                       #
    # ------------------------------------------------------------------ #
    dS[Var.V_d2] = S[Var.V_d2_dot]
    dS[Var.V_d2_dot] = (
            self.alpha_beta *
            (
                    c['vd2e']  * d(Var.V_e,  'tau_d2e') +
                    c['vd2d1'] * sig(d(Var.V_d1, 'tau_d2d1'), 'd1') +
                    c['vd2s']  * sig(d(Var.V_s,  'tau_d2s'),  's') +
                    c['vd2d2'] * sig(S[Var.V_d2], 'd2') -
                    S[Var.V_d2]
            )
            - self.a_plus_b * S[Var.V_d2_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-p1 (GPi / SNr, first segment)                                   #
    # ------------------------------------------------------------------ #
    dS[Var.V_p1] = S[Var.V_p1_dot]
    dS[Var.V_p1_dot] = (
            self.alpha_beta *
            (
                    c['vp1d1'] * sig(d(Var.V_d1, 'tau_p1d1'), 'd1') +
                    c['vp1p2'] * sig(d(Var.V_p2, 'tau_p1p2'), 'p2') +
                    c['vp1ST'] * sig(d(Var.V_STN,'tau_p1ST'), 'STN') -
                    S[Var.V_p1]
            )
            - self.a_plus_b * S[Var.V_p1_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-p2 (GPe, second segment)                                        #
    # ------------------------------------------------------------------ #
    dS[Var.V_p2] = S[Var.V_p2_dot]
    dS[Var.V_p2_dot] = (
            self.alpha_beta *
            (
                    c['vp2d2'] * sig(d(Var.V_d2, 'tau_p2d2'), 'd2') +
                    c['vp2p2'] * sig(S[Var.V_p2], 'p2') +
                    c['vp2ST'] * sig(d(Var.V_STN,'tau_p2ST'), 'STN') -
                    S[Var.V_p2]
            )
            - self.a_plus_b * S[Var.V_p2_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-STN (sub-thalamic nucleus)                                      #
    # ------------------------------------------------------------------ #
    dS[Var.V_STN] = S[Var.V_STN_dot]
    dS[Var.V_STN_dot] = (
            self.alpha_beta *
            (
                    c['vSTp2'] * sig(d(Var.V_p2, 'tau_STp2'), 'p2') +
                    c['vSTe']  * d(Var.phi_e,   'tau_STe')    -
                    S[Var.V_STN]
            )
            - self.a_plus_b * S[Var.V_STN_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-s  (thalamus)                                                   #
    # ------------------------------------------------------------------ #
    dS[Var.V_s] = S[Var.V_s_dot]
    dS[Var.V_s_dot] = (
            self.alpha_beta *
            (
                    c['vsp1'] * sig(d(Var.V_p1, 'tau_sp1'), 'p1') +
                    c['vse']  * d(Var.phi_e,   'tau_se') +
                    c['vsr']  * sig(d(Var.V_r,  'tau_sr'), 'r') +
                    self.p.raw['phi_n'] -
                    S[Var.V_s]
            )
            - self.a_plus_b * S[Var.V_s_dot]
    )

    # ------------------------------------------------------------------ #
    #  V-r  (cortical relay nuclei)                                      #
    # ------------------------------------------------------------------ #
    dS[Var.V_r] = S[Var.V_r_dot]
    dS[Var.V_r_dot] = (
            self.alpha_beta *
            (
                    c['vre'] * d(Var.phi_e, 'tau_re') +
                    c['vrs'] * sig(d(Var.V_s,'tau_rs'), 's') -
                    S[Var.V_r]
            )
            - self.a_plus_b * S[Var.V_r_dot]
    )

    return dS

    def _delay(self, var: Var, tau_name: str) -> np.ndarray:
        """Return `var` evaluated *tau* steps earlier (zero‑padded)."""
        k = self.tau[tau_name]
        j = max(self.i - k, 0)
        return self.S[j, var]

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def advance(self) -> None:
        """A single Euler step (dt is fixed)."""
        S_now = self.S[self.i]
        dSdt = self._rhs(S_now)

        # DBS (optional) ----------------------------------------------------
        if self.cdbs:
            I_stim = self.cdbs.advance()
            tgt_idx = POP_IDX[self.p.dbs_cfg['stim_target']]
            S_now[tgt_idx] += I_stim / self.p.dbs_cfg['cm']
            if I_stim != 0:
                self.stim_trace[self.i] = I_stim

        # Integrate ---------------------------------------------------------
        self.S[self.i + 1] = S_now + self.p.dt * dSdt

        # Add noise – vectorised loop over a small table --------------------
        ξ = np.random.normal(0.0, 1.0, len(self._noise_table))
        for n, (var, pop) in enumerate(self._noise_table):
            Q, θ = self._sigmoid_cache[pop]
            σ_pre  = self._sigmoid(self.S[self.i,   var], pop)
            σ_post = self._sigmoid(self.S[self.i+1, var], pop)
            self.S[self.i+1, var] += (
                    self.p.raw['noise_amp'] *
                    ξ[n] * np.sqrt(self.p.dt) *
                    Q * (1 - σ_post) * σ_pre
            )

        # Book‑keeping ------------------------------------------------------
        self.i += 1
        self.t += self.p.dt

    def run(self) -> None:
        """Integrate until `tstop` (inclusive)."""
        while self.i < self.p.n_steps - 1:
            self.advance()

    # -- Convenience post‑processing ----------------------------------------
    def lfp(self) -> np.ndarray:
        """Return the simulated LFP of the configured target population."""
        return self.S[:, POP_IDX[self.p.dbs_cfg['target_state']]]

    def power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.signal import welch
        f, Pxx = welch(self.lfp(), fs=self.p.fs, nperseg=2048)
        return f, 10 * np.log10(Pxx)
