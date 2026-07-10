"""Structural rewrite pass (@jit(rewrite=True)): pattern-matched algebraic
rewrites. Each must be EXACTLY equivalent to the un-rewritten kernel — a
rewrite that changes results is a bug, not an optimization. CPU only."""
import time
import warnings
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def test_closed_form_sum_exact():
    def gauss(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc
    jf = jit(rewrite=True)(gauss)
    for n in (0, 1, 2, 3, 10, 100, 1000, 100000):
        assert jf(n) == gauss(n), n


def test_closed_form_weighted_sum_exact():
    def w(n):
        acc = 0
        for i in range(n):
            acc += 3 * i
        return acc
    jf = jit(rewrite=True)(w)
    for n in (0, 1, 5, 999, 50000):
        assert jf(n) == w(n), n


def test_closed_form_affine_sum_exact():
    def aff(n):
        acc = 0
        for i in range(n):
            acc += 2 * i + 5
        return acc
    jf = jit(rewrite=True)(aff)
    for n in (0, 1, 2, 17, 100000):
        assert jf(n) == aff(n), n


def test_rewrite_matches_nonrewrite_version():
    """rewrite=True and rewrite=False must agree bit-for-bit (same i64
    wraparound semantics)."""
    def s(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc
    plain = jit(s)
    rw = jit(rewrite=True)(s)
    for n in (0, 1, 1000, 200000):
        assert plain(n) == rw(n), n


def test_rewrite_eliminates_loop():
    """The closed form is O(1): a huge n returns instantly."""
    def g(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc
    rw = jit(rewrite=True)(g)
    rw(10)
    t0 = time.perf_counter()
    rw(10_000_000)          # would take ms as a loop; O(1) closed form
    dt = time.perf_counter() - t0
    assert dt < 1e-3, f"loop not eliminated: {dt*1e3:.2f}ms"


def test_rewrite_leaves_nonmatching_loops_alone():
    """A loop that doesn't match a pattern must still compile and be
    correct (rewrite is additive, never harmful)."""
    def notclosed(n):
        acc = 0
        for i in range(n):
            acc += i * i        # not an affine body -> no closed-form rule
        return acc
    rw = jit(rewrite=True)(notclosed)
    for n in (0, 1, 10, 137):
        assert rw(n) == notclosed(n), n


def test_rewrite_off_by_default():
    """Without rewrite=True, no structural rewriting happens."""
    def g(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc
    plain = jit(g)
    assert plain(1000) == g(1000)
