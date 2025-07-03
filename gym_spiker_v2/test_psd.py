#!/usr/bin/env python3
"""
test_psd.py – Regression test that the PSD produced by BGEnv/MFM
matches the previously saved reference arrays.

Exit status:
    0 – success, all arrays matched (within tolerance)
    1 – at least one array differed
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from BGEnv import BGEnv, MFM

TOLERANCE = 1e-6

@contextmanager
def elapsed(label="time elapsed"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    sec = int(dt)
    ms  = int((dt - sec) * 1000)
    print(f"{label}: {sec}s {ms}ms")

np.random.seed(0)

ACTIONS = np.array([
    [-1.00,  60.0,  30.0],
    [ 0.11,  29.0, 124.0],
    [ 7.38, 564.5, 166.8],
], dtype=np.float32)

LABELS = ["A", "B", "C"]


def simulate(env: BGEnv, action: np.ndarray) -> np.ndarray:
    """Run BGEnv once with the given DBS action and return the PSD array."""
    kwargs = dict(
        cDBS=True,
        cDBS_amp=float(action[0]),
        cDBS_width=float(action[1]),
        cDBS_f=float(action[2]),
        verbose=False,
    )
    state = np.random.get_state()
    mfm = MFM(**kwargs)
    mfm.run()
    np.random.set_state(state)

    obs, _ = env.reset(seed=0)
    with elapsed():
        obs, *_ = env.step(action)

    return obs.copy()


def main() -> None:
    ref_path = Path(__file__).with_name("reference_psd.npz")
    if not ref_path.exists():
        print(f"Reference file {ref_path} not found.  Run store_psd.py first.")
        sys.exit(1)

    ref = np.load(ref_path)
    env = BGEnv()

    success = True
    for lbl, act in zip(LABELS, ACTIONS):
        current = simulate(env, act)
        if not np.allclose(current, ref[lbl], atol=TOLERANCE):
            diff = np.max(np.abs(current - ref[lbl]))
            print(
                f"Mismatch for action {lbl}: max |Δ| = {diff:.3e} "
                f"(tolerance {TOLERANCE})"
            )
            success = False

    if success:
        print("All PSD arrays match reference.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
