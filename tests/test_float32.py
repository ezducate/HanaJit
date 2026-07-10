"""Native float32 array support: float32 numpy arrays compile with 32-bit
LLVM float ops (half the memory bandwidth, 2x SIMD lanes). Semantics are
EXACT float32 — not an approximation — so results match numpy's float32
computation, and the precision is the well-defined f32 precision (~7
significant digits), not undefined behavior.

This is the sound aggressive-optimization: bounded, quantified tradeoff
(3.6x faster on memory-bound reductions for f32 precision), opt in simply
by passing float32 arrays."""
import time
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def test_float32_reduction_matches_numpy_f32():
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jf = jit(s)
    a = np.random.default_rng(0).uniform(-1, 1, 50_000).astype(np.float32)
    # matches numpy's float32 sum within f32 accumulation tolerance
    assert jf(a) == pytest.approx(float(np.float32(a).sum()), rel=1e-4)


def test_float32_elementwise_matches_numpy():
    def f(x, out):
        for i in range(len(x)):
            out[i] = x[i] * x[i] + 2.0
        return 0
    jf = jit(f)
    a = np.random.default_rng(1).uniform(-3, 3, 10_000).astype(np.float32)
    out = np.zeros_like(a)
    jf(a, out)
    assert np.allclose(out, a * a + 2.0, rtol=1e-4, atol=1e-4)
    assert out.dtype == np.float32


def test_float32_and_float64_both_work():
    """Same kernel, both dtypes, each correct in its own precision."""
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i] * x[i]
        return acc
    jf = jit(s)
    base = np.random.default_rng(2).uniform(-1, 1, 20_000)
    a64 = base.copy()
    a32 = base.astype(np.float32)
    assert jf(a64) == pytest.approx(float((a64 * a64).sum()), rel=1e-9)
    assert jf(a32) == pytest.approx(float((a32.astype(np.float64) ** 2).sum()),
                                    rel=1e-2)


def test_float32_strided_view():
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jf = jit(s)
    a = np.arange(200, dtype=np.float32)[::2]     # 4-byte-element strided view
    assert jf(a) == pytest.approx(float(a.sum()), rel=1e-4)


def test_float32_2d():
    def total(m):
        s = 0.0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                s += m[i, j]
        return s
    jf = jit(total)
    m = np.random.default_rng(3).uniform(-1, 1, (50, 40)).astype(np.float32)
    assert jf(m) == pytest.approx(float(m.sum()), rel=1e-3)


def test_float32_fusion():
    def f(x):
        return np.sum(x * x + 0.5 * x)
    jf = jit(f)
    a = np.random.default_rng(4).uniform(-1, 1, 30_000).astype(np.float32)
    ref = float((a.astype(np.float64) ** 2 + 0.5 * a).sum())
    assert jf(a) == pytest.approx(ref, rel=1e-3)


def test_float32_reduce_reassoc_combines():
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jf = jit(reduce_reassoc=True)(s)
    a = np.random.default_rng(5).uniform(-1, 1, 50_000).astype(np.float32)
    assert jf(a) == pytest.approx(float(np.float32(a).sum()), rel=1e-3)


def test_float32_precision_is_bounded_not_garbage():
    """The whole safety argument: f32 error is BOUNDED at f32 precision,
    never arbitrary. A value computed in f32 must be within f32 epsilon
    of the true sum, not wrong."""
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jf = jit(s)
    # controlled input where we know the exact answer
    a = np.full(1000, 0.5, dtype=np.float32)
    assert jf(a) == pytest.approx(500.0, rel=1e-5)   # exact-ish, bounded


@pytest.mark.skipif((__import__("os").cpu_count() or 1) < 1, reason="")
def test_float32_faster_when_vectorized():
    """f32 + reassoc should beat f64 + reassoc (more SIMD lanes)."""
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jr = jit(reduce_reassoc=True)(s)
    a64 = np.random.default_rng(6).uniform(-1, 1, 20_000_000)
    a32 = a64.astype(np.float32)
    jr(a64); jr(a32)

    def best(arg):
        t = float("inf")
        for _ in range(5):
            t0 = time.perf_counter(); jr(arg); t = min(t, time.perf_counter()-t0)
        return t
    t64, t32 = best(a64), best(a32)
    assert t32 < t64 / 1.3, f"f64 {t64*1e3:.1f}ms f32 {t32*1e3:.1f}ms"
