"""@jit(parallel=True): the outermost range loop auto-promotes to prange.
Tested for exact/approx equivalence with the serial oracle AND for actual
speedup on multiple cores. Where the loop shape isn't parallelizable, it
must compile serially and stay correct — parallel=True never changes
results, only scheduling."""
import os
import time
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")
CORES = os.cpu_count() or 1


def test_parallel_reduction_matches_serial():
    def work(n):
        acc = 0.0
        for i in range(n):
            x = i * 1e-6
            acc += x * x - 0.5 * x
        return acc

    ser = jit(work)
    par = jit(parallel=True)(work)
    n = 2_000_000
    # fp reduction is reassociated across chunks -> ~1e-12, not bit-exact
    assert par(n) == pytest.approx(ser(n), rel=1e-10)


def test_parallel_array_write_matches_serial():
    def fill(x, out):
        for i in range(len(x)):
            out[i] = x[i] * x[i] + 1.0
        return 0

    par = jit(parallel=True)(fill)
    a = np.random.default_rng(0).uniform(-1, 1, 500_000)
    o1 = np.zeros_like(a)
    par(a, o1)
    assert np.allclose(o1, a * a + 1.0)


@pytest.mark.skipif(CORES < 2, reason="needs >=2 cores")
def test_parallel_actually_speeds_up():
    def heavy(n):
        acc = 0.0
        for i in range(n):
            x = i * 1e-7
            # enough arithmetic that the loop, not overhead, dominates
            acc += (x * x + 0.3 * x - 0.1) * (x - 0.2) + x * x * x
        return acc

    ser = jit(heavy)
    par = jit(parallel=True, workers=CORES)(heavy)
    n = 8_000_000
    assert par(n) == pytest.approx(ser(n), rel=1e-9)

    def best(f):
        t = float("inf")
        for _ in range(3):
            t0 = time.perf_counter(); f(n); t = min(t, time.perf_counter()-t0)
        return t
    ts, tp = best(ser), best(par)
    assert tp < ts * 0.75, f"serial {ts*1e3:.1f}ms parallel {tp*1e3:.1f}ms"


def test_non_parallelizable_falls_back_to_serial_correct():
    """Two top-level loops -> not auto-parallelizable -> serial, correct."""
    def two_loops(n):
        a = 0
        for i in range(n):
            a += i
        b = 0
        for j in range(n):
            b += j * 2
        return a + b

    par = jit(parallel=True)(two_loops)
    assert par(1000) == two_loops(1000)


def test_parallel_equals_serial_on_many_sizes():
    def s(n):
        acc = 0
        for i in range(n):
            acc += (i % 7) - (i % 3)
        return acc
    par = jit(parallel=True)(s)
    for n in (0, 1, 2, 17, 1000, 99999):
        assert par(n) == s(n), n            # integer: exact
