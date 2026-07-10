"""prange overhead & scaling. On a 1-core CI box this can only show
correctness and overhead; run on a multi-core machine for real scaling."""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit, prange

A = np.random.default_rng(0).uniform(-1, 1, 8_000_000)

@jit
def serial(x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * x[i] + 0.5 * x[i]
    return s

def make(workers):
    @jit(workers=workers)
    def par(x):
        s = 0.0
        for i in prange(len(x)):
            s += x[i] * x[i] + 0.5 * x[i]
        return s
    return par

def best(f, *a):
    b = 9e9
    for _ in range(5):
        t0 = time.perf_counter(); r = f(*a); b = min(b, time.perf_counter()-t0)
    return r, b

r0, t0 = best(serial, A)
print(f"cores available: {os.cpu_count()}")
print(f"serial @jit:            {t0*1e3:7.2f} ms")
for w in (1, 2, 4, 8):
    p = make(w)
    r, t = best(p, A)
    assert abs(r - r0) < 1e-6 * abs(r0)
    print(f"prange workers={w}:      {t*1e3:7.2f} ms  ({t0/t:4.2f}x vs serial)")
