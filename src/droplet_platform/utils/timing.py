from __future__ import annotations

from contextlib import contextmanager
import time


@contextmanager
def timer(label: str = "elapsed"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.3f} s")
