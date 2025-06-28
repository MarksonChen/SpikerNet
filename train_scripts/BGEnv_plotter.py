import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gymnasium as gym

from gym_spiker.envs.BGEnv import BGEnv
from gym_spiker.envs.mfm   import MFM

np.random.seed(0)

actions = np.array([
    [-1.00,   60.0,   30.0],   # A (mA, µs, Hz)
    [ 0.11,   29.0,  124.0],   # B
    [ 7.38,  564.5,  166.8],   # C
], dtype=np.float32)
labels = ['A', 'B', 'C']

def simulate(env, action):
    kwargs = dict(cDBS=True,
                  cDBS_amp   = float(action[0]),
                  cDBS_width = float(action[1]),
                  cDBS_f     = float(action[2]))
    state = np.random.get_state()
    mfm   = MFM(**kwargs); mfm.run()
    np.random.set_state(state)

    obs, _ = env.reset(seed=0)
    obs, reward, *_ = env.step(action)
    psd = obs.copy()

    fs   = 1.0 / mfm.params['dt']  # 1000 Hz
    lfp  = mfm.getLFP()
    f, t_spec, Sxx = signal.spectrogram(lfp, fs=fs,
                                        nperseg=2048, noverlap=1536)
    bands = {'δ':(1,4), 'θ':(4,8), 'α':(8,12),
             'β':(12,35), 'γ':(35,80)}
    bp = {}
    for n,(f1,f2) in bands.items():
        idx = (f>=f1)&(f<f2)
        bp[n] = 10*np.log10(Sxx[idx].mean(0)+1e-20)

    return dict(psd=psd, bp=bp, t=t_spec)


env   = BGEnv(render_mode=None)
runs  = [simulate(env, a) for a in actions]
target_psd = env.targetPSTH.copy()


fs_target = 0.5                    # PSD has 0.5 Hz bins
freq_bins = np.arange(target_psd.size) * fs_target
bands = {'δ':(1,4), 'θ':(4,8), 'α':(8,12),
         'β':(12,35), 'γ':(35,80)}
target_bp = {}
for n,(f1,f2) in bands.items():
    idx = (freq_bins >= f1) & (freq_bins < f2)
    p_lin = 10**(target_psd[idx] / 10.0)
    target_bp[n] = 10*np.log10(p_lin.mean() + 1e-20)


plt.figure(figsize=(6,3), dpi=120, tight_layout=True)
plt.plot(target_psd[:512], label='Target', lw=2, c='k')
for k,r in enumerate(runs):
    plt.plot(r['psd'][:512], label=f"Action {labels[k]}", alpha=0.7)
plt.title("Figure 1 – Target PSD vs Actions (0–256 Hz)")
plt.xlabel("Frequency bin (0.5 Hz)"); plt.ylabel("dB/Hz")
plt.legend()


col = {'δ':'tab:blue', 'θ':'tab:orange', 'α':'tab:green',
       'β':'tab:red',  'γ':'tab:purple'}
for k,r in enumerate(runs):
    plt.figure(figsize=(8,4), dpi=120, tight_layout=True)
    for band,color in col.items():
        plt.plot(r['t'], r['bp'][band], color=color, label=band)
        plt.hlines(target_bp[band], r['t'][0], r['t'][-1],
                   color=color, linestyle='--', linewidth=1.2)
    plt.title(f"Figure {2+k} – Band-power Dynamics (Action {labels[k]})")
    plt.xlabel("Time [s]"); plt.ylabel("Power [dB]")
    plt.legend(ncol=5)

plt.show()
