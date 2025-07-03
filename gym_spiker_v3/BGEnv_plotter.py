import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from omegaconf import DictConfig, OmegaConf
from BGEnv import BGEnv
from mfm import MFM
from utils import elapsed

def simulate(env, action, config: DictConfig = None):
    if config is None:
        config = OmegaConf.load("config/mfm_config.yaml")

    state = np.random.get_state()

    mfm = MFM(config=config,
              cDBS=True,
              cDBS_amp=float(action[0]),
              cDBS_width=float(action[1]),
              cDBS_f=float(action[2]))
    mfm.run()
    np.random.set_state(state)

    obs, _ = env.reset(seed=0)
    with elapsed("env.step"):
        obs, reward, *_ = env.step(action)
    psd = obs.copy()

    fs = 1.0 / mfm.cfg.simulation.dt  # 1000 Hz
    lfp = mfm.lfp
    f, t_spec, Sxx = signal.spectrogram(lfp, fs=fs, nperseg=2048, noverlap=1536)

    bands = {'δ': (1, 4), 'θ': (4, 8), 'α': (8, 12), 'β': (12, 35), 'γ': (35, 80)}
    bp = {}
    for name, (f1, f2) in bands.items():
        idx = (f >= f1) & (f < f2)
        bp[name] = 10 * np.log10(Sxx[idx].mean(0) + 1e-20)

    return dict(psd=psd, bp=bp, t=t_spec, reward=reward)


def create_target_band_powers(target_psd):
    """Create target band powers from target PSD"""
    fs_target = 0.5  # PSD has 0.5 Hz bins
    freq_bins = np.arange(target_psd.size) * fs_target
    bands = {'δ': (1, 4), 'θ': (4, 8), 'α': (8, 12), 'β': (12, 35), 'γ': (35, 80)}

    target_bp = {}
    for name, (f1, f2) in bands.items():
        idx = (freq_bins >= f1) & (freq_bins < f2)
        p_lin = 10 ** (target_psd[idx] / 10.0)
        target_bp[name] = 10 * np.log10(p_lin.mean() + 1e-20)

    return target_bp


def main():
    """Main plotting function"""
    np.random.seed(0)

    # [amplitude, pulse_width, frequency]
    actions = np.array([
        [-1.00,  60.0,  30.0],   # A
        [ 0.11,  29.0, 124.0],   # B
        [ 7.38, 564.5, 166.8],   # C
        [ 0, 1, 1],   # D
    ], dtype=np.float32)
    labels = ['A', 'B', 'C', 'D']

    config = OmegaConf.load("config/mfm_config.yaml")

    env = BGEnv(config)

    print("Running simulations...")
    runs = [simulate(env, action, config) for action in actions]
    target_psd = env.target_psth.copy()
    target_bp = create_target_band_powers(target_psd)

    # Plot 1: Target PSD vs Actions
    plt.figure(figsize=(6, 3), dpi=120, tight_layout=True)
    plt.plot(target_psd[:512], label='Target', lw=2, c='k')
    for k, run in enumerate(runs):
        plt.plot(run['psd'][:512], label=f"Action {labels[k]}", alpha=0.7)
    plt.title("Figure 1 – Target PSD vs Actions (0–256 Hz)")
    plt.xlabel("Frequency bin (0.5 Hz)")
    plt.ylabel("dB/Hz")
    plt.legend()

    # Plot 2-4: Band-power dynamics for each action
    colors = {'δ': 'tab:blue', 'θ': 'tab:orange', 'α': 'tab:green',
              'β': 'tab:red', 'γ': 'tab:purple'}

    for k, run in enumerate(runs):
        plt.figure(figsize=(8, 4), dpi=120, tight_layout=True)
        for band, color in colors.items():
            plt.plot(run['t'], run['bp'][band], color=color, label=band)
            plt.hlines(target_bp[band], run['t'][0], run['t'][-1],
                       color=color, linestyle='--', linewidth=1.2)
        plt.title(f"Figure {2+k} – Band-power Dynamics (Action {labels[k]}) "
                  f"- Reward: {run['reward']:.2f}")
        plt.xlabel("Time [s]")
        plt.ylabel("Power [dB]")
        plt.legend(ncol=5)

    plt.show()


if __name__ == "__main__":
    main()
