#!/usr/bin/env python3
"""
store_psd.py – Generate and save reference PSD arrays for three DBS actions.

This script reproduces the PSD computation logic used in the analysis/plotting
notebook but omits all plotting.  The resulting file `reference_psd.npz`
contains three 1‑D float32 arrays keyed by 'A', 'B', and 'C'.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from BGEnv import BGEnv, MFM


@contextmanager
def elapsed(label: str = "elapsed"):
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        sec, ms = divmod(dt * 1000, 1000)
        print(f"{label}: {int(sec):d}s {int(ms):03d}ms")


# Reproducible RNG state
np.random.seed(0)

# (mA, µs, Hz)
ACTIONS = np.array(
    [
        [-1.00, 60.0, 30.0],   # A
        [0.11, 29.0, 124.0],  # B
        [7.38, 564.5, 166.8],  # C
    ],
    dtype=np.float32,
)
LABELS = ["A", "B", "C"]


def simulate(env: BGEnv, action: np.ndarray) -> np.ndarray:
    """Run BGEnv once and return the PSD observation array."""

    kwargs = dict(
        cDBS=True,
        cDBS_amp=float(action[0]),
        cDBS_width=float(action[1]),
        cDBS_f=float(action[2]),
        verbose=False,
    )

    # Make MFM deterministic w.r.t. global RNG
    state = np.random.get_state()
    mfm = MFM(**kwargs)
    mfm.run()
    np.random.set_state(state)

    # env.step returns (obs, reward, terminated, truncated, info)
    obs, _ = env.reset(seed=0)
    with elapsed("env.step"):
        obs, *_ = env.step(action)

    return obs.copy()


def main() -> None:
    env = BGEnv(render_mode=None)
    psd_arrays = {lbl: simulate(env, act) for lbl, act in zip(LABELS, ACTIONS)}

    out_path = Path(__file__).with_name("reference_psd.npz")
    np.savez(out_path, **psd_arrays)
    print(f"Saved PSD reference arrays → {out_path.resolve()}")


if __name__ == "__main__":
    main()
