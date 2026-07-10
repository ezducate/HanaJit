import warnings
import numpy as np
import pytest
from hanajit import jit, prange

warnings.filterwarnings("ignore")


def test_prange_reduction_matches_serial():
    @jit(workers=4)
    def psum(x):
        s = 0.0
        for i in prange(len(x)):
            s += x[i] * x[i] + 0.5 * x[i]
        return s

    @jit
    def ssum(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i] + 0.5 * x[i]
        return s

    a = np.random.default_rng(0).uniform(-1, 1, 100_003)
    assert getattr(psum, "parallel", False)
    assert psum(a) == pytest.approx(ssum(a), rel=1e-11)


def test_prange_array_writes_disjoint():
    @jit(workers=4)
    def scale(x, y):
        for i in prange(len(x)):
            y[i] = 3.0 * x[i] - 1.0
        return 0

    a = np.random.default_rng(1).uniform(-1, 1, 50_000)
    b = np.zeros_like(a)
    scale(a, b)
    assert np.allclose(b, 3.0 * a - 1.0)


def test_prange_two_bounds_and_return_expr():
    @jit(workers=3)
    def part(x, lo, hi):
        s = 0.0
        for i in prange(lo, hi):
            s += x[i]
        return s * 2.0 + lo

    a = np.arange(1000, dtype=np.float64)
    assert part(a, 100, 900) == pytest.approx(
        float(a[100:900].sum()) * 2.0 + 100)


def test_prange_int_reduction_exact():
    @jit(workers=4)
    def isum(n):
        s = 0
        for i in prange(n):
            s += i % 7
        return s

    assert isum(100_000) == sum(i % 7 for i in range(100_000))


def test_prange_nonliteral_init_and_seed():
    @jit(workers=4)
    def with_init(n):
        s = 100
        for i in prange(n):
            s += 1
        return s
    assert with_init(5000) == 5100  # initial value added exactly once


def test_unparallelizable_degrades_to_serial_compiled():
    @jit(verbose=False)
    def two_accs(n):
        a = 0
        b = 0
        for i in prange(n):
            a += i
            b += i * 2
        return a + b
    # two reductions -> serial path, still compiled & correct
    assert not getattr(two_accs, "parallel", False)
    assert two_accs(1000) == sum(range(1000)) * 3
    assert len(two_accs.cache) == 1


def test_prange_gil_released_in_chunks():
    import threading, time

    @jit(workers=1)
    def heavy(n):
        s = 0.0
        for i in prange(n):
            s += (i % 7) * 0.5
        return s
    heavy(10)
    t = threading.Thread(target=heavy, args=(400_000_000,))
    t.start()
    ticks = 0
    while t.is_alive():
        ticks += 1
    t.join()
    assert ticks > 1000  # main thread ran while chunk computed
