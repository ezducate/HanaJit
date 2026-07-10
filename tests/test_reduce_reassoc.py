"""reduce_reassoc=True: applies the `reassoc` fast-math flag ONLY to
reduction accumulators (float += and np.sum/dot/mean), letting LLVM
vectorize with parallel accumulators — the same reassociation numpy's
pairwise sum performs. This is the safe path to numpy-class reduction
speed without global fastmath.

Tested for: (1) accuracy stays within reassociation tolerance vs the exact
sum (NOT bit-exact — reassoc reorders, by design), (2) integer reductions
are untouched and stay bit-exact, (3) non-reduction arithmetic is
unaffected, (4) it actually vectorizes (speedup on a large reduction)."""
import time
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def test_float_reduction_within_reassoc_tolerance():
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jr = jit(reduce_reassoc=True)(s)
    a = np.random.default_rng(0).uniform(-1, 1, 100_000)
    exact = sum(float(v) for v in a)
    # reassoc reorders -> not bit-exact, but same order of magnitude as numpy
    assert abs(jr(a) - exact) < 1e-6 * max(1, abs(exact))
    assert abs(jr(a) - float(np.sum(a))) < 1e-6 * max(1, abs(exact))


def test_integer_reduction_stays_bit_exact():
    """reassoc is FP-only; integer reductions must be unchanged."""
    def s(x):
        acc = 0
        for i in range(len(x)):
            acc += x[i]
        return acc
    jr = jit(reduce_reassoc=True)(s)
    a = np.arange(-500, 5000, dtype=np.int64)
    assert jr(a) == int(a.sum())            # exact


def test_npsum_npdot_npmean_correct_with_reassoc():
    def f_sum(x): return np.sum(x)
    def f_dot(x, y): return np.dot(x, y)
    def f_mean(x): return np.mean(x)
    a = np.random.default_rng(1).uniform(-1, 1, 50_000)
    b = np.random.default_rng(2).uniform(-1, 1, 50_000)
    assert jit(reduce_reassoc=True)(f_sum)(a) == pytest.approx(
        float(np.sum(a)), rel=1e-9)
    assert jit(reduce_reassoc=True)(f_dot)(a, b) == pytest.approx(
        float(np.dot(a, b)), rel=1e-9)
    assert jit(reduce_reassoc=True)(f_mean)(a) == pytest.approx(
        float(np.mean(a)), rel=1e-9)


def test_non_reduction_arithmetic_unaffected():
    """A float += that ISN'T the only thing (still a reduction) vs general
    float math: general math must be byte-identical to default."""
    def compute(x):
        # not an accumulation loop — element transform stored to array
        return x * 2.0 + 1.0
    a = np.random.default_rng(3).uniform(-5, 5, 1000)
    plain = jit(compute)
    rr = jit(reduce_reassoc=True)(compute)
    # elementwise transform: reassoc changes nothing here
    assert np.array_equal(
        np.array([plain(v) for v in a]),
        np.array([rr(v) for v in a]))


def test_reduce_reassoc_off_by_default():
    """Default jit must NOT reassociate (bit-exact sequential sum)."""
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    a = np.random.default_rng(4).uniform(-1, 1, 10_000)
    seq = 0.0
    for v in a:
        seq += float(v)
    assert jit(s)(a) == seq                 # default is exact sequential


def test_reduce_reassoc_actually_speeds_up():
    """It must vectorize: a large reduction should be meaningfully faster
    than the sequential default."""
    def s(x):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i]
        return acc
    a = np.random.default_rng(5).uniform(-1, 1, 20_000_000)
    jd = jit(s)
    jr = jit(reduce_reassoc=True)(s)
    jd(a); jr(a)

    def best(f):
        t = float("inf")
        for _ in range(5):
            t0 = time.perf_counter(); f(a); t = min(t, time.perf_counter()-t0)
        return t
    td, tr = best(jd), best(jr)
    # expect ~1.5x from parallel SIMD accumulators; require >1.25x
    assert tr < td / 1.25, f"default {td*1e3:.1f}ms reassoc {tr*1e3:.1f}ms"
