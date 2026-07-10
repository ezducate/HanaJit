"""Differential testing OF evolved kernels: run the GA, then re-verify the
installed winner against the pure-Python/numpy oracle on fresh random
inputs — including the semantics traps (negative //, %, bool arithmetic)
and fused expressions. This guarantees the two dispatch states (pre- and
post-evolve) are equivalent, not just on evolve's own probes."""
import random
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")
random.seed(20260707)
GA = dict(generations=3, population=6, reps=2, seed=1)


def oracle_check(jf, pyfn, argsets, approx=False):
    for args in argsets:
        got, ref = jf(*args), pyfn(*args)
        if approx:
            assert got == pytest.approx(ref, rel=1e-9), args
        else:
            assert got == ref, (args, got, ref)


def test_evolved_semantics_negative_divmod():
    def f(a, b):
        return (a // b) * 100 + a % b + (a < b) + (a > 0 and b < 0)
    jf = jit(f)
    pairs = [(random.randint(-10**6, 10**6), random.choice([-7, -2, 3, 11]))
             for _ in range(40)]
    oracle_check(jf, f, pairs)                    # before GA
    rep = jf.evolve(-12345, 7, **GA)
    oracle_check(jf, f, pairs)                    # after GA: same oracle
    assert rep["speedup"] >= 1.0


def test_evolved_float_kernel_exact_without_fastmath():
    def f(n):
        acc = 0.0
        x = 0.5
        for i in range(n):
            x = 3.999 * x * (1.0 - x)
            acc += x % 0.7 - x // 0.3
        return acc
    jf = jit(f)
    refs = {n: f(n) for n in (10, 137, 5000)}
    jf.evolve(5000, **GA)                          # no fastmath: bit-exact
    for n, ref in refs.items():
        assert jf(n) == ref, n                     # == , not approx


def test_evolved_fusion_kernel_vs_numpy():
    def f(a, b):
        return np.sum(np.where(a > b, np.exp(-a * a), b * 0.5)
                      + np.clip(a * b, -0.4, 0.4))
    jf = jit(f)
    mk = lambda s: np.random.default_rng(s).uniform(-2, 2, 4001)
    a0, b0 = mk(1), mk(2)
    jf.evolve(a0, b0, **GA)
    for s in (3, 4, 5):                            # fresh, unseen inputs
        a, b = mk(s), mk(s + 100)
        assert jf(a, b) == pytest.approx(float(f.__wrapped__(a, b) if
                                         hasattr(f, "__wrapped__") else
                                         f(a, b)), rel=1e-9)


def test_evolved_fastmath_stays_within_tolerance_on_unseen_inputs():
    def f(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i] - 0.25 * x[i]
        return s
    jf = jit(f)
    a = np.random.default_rng(9).uniform(-1, 1, 100_000)
    rep = jf.evolve(a, allow_fastmath=True, **GA)
    for s in (11, 12, 13):                         # inputs the GA never saw
        u = np.random.default_rng(s).uniform(-3, 3, 50_000)
        assert jf(u) == pytest.approx(f(u), rel=1e-8), s
    assert rep["speedup"] >= 1.0


def test_evolve_then_new_specialization_still_correct():
    """A GA winner for (int,) must not disturb later (float,) dispatch."""
    def f(x):
        return x * 3 + (x // 2 if x > 0 else -x)
    jf = jit(f)
    jf.evolve(1000, **GA)                          # evolves the int path
    assert jf(7) == f(7)
    assert jf(7.5) == f(7.5)                       # fresh float path
    assert jf(-9) == f(-9) and jf(-9.5) == f(-9.5)
