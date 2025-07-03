import time
from contextlib import contextmanager

@contextmanager
def elapsed(label="time elapsed"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    sec = int(dt)
    ms = int((dt - sec) * 1000)
    print(f"{label}: {sec}s {ms}ms")
