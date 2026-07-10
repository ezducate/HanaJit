"""Helper inlining: cross-function calls between @jit functions become
in-line arithmetic. Tested for (a) mathematical equivalence with the
pure-Python oracle, and (b) that inlining actually happened — the call to
the helper must not survive into the compiled kernel (verified via IR),
and the previously-slow cross-call benchmark must now be fast."""
import time
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


@jit
def sq(x):
    return x * x


@jit
def cube(x):
    return x * x * x


@jit
def hyp2(a, b):
    return sq(a) + sq(b)          # helper call inside helper-friendly expr


def test_inlined_result_matches_python():
    def py_hyp2(a, b):
        return a * a + b * b
    for a, b in [(3.0, 4.0), (-1.5, 2.5), (0.0, 7.0)]:
        assert hyp2(a, b) == pytest.approx(py_hyp2(a, b), rel=1e-15)


def test_inlined_in_loop_matches_python():
    @jit
    def energy(x):
        s = 0.0
        for i in range(len(x)):
            s += sq(x[i]) + cube(x[i])
        return s

    def py(x):
        s = 0.0
        for v in x:
            s += v * v + v * v * v
        return s
    a = np.random.default_rng(0).uniform(-2, 2, 5000)
    assert energy(a) == pytest.approx(py(a), rel=1e-12)


def test_call_does_not_survive_in_ir():
    @jit(native_dispatch=False)   # Python dispatcher retains IR for inspection
    def uses_helpers(x):
        return sq(x) + cube(x) + sq(x + 1.0)
    uses_helpers(2.0)
    ir = uses_helpers.inspect_llvm()
    # no call instruction to a helper survives (fully inlined)
    assert "@sq" not in ir and "@cube" not in ir  # helpers fully inlined
    assert uses_helpers(3.0) == pytest.approx(3*3 + 3*3*3 + 4*4, rel=1e-12)


def test_nested_helper_inlining():
    @jit(native_dispatch=False)
    def quart(x):
        return sq(sq(x))          # helper-of-helper: must fully flatten
    for v in (2.0, -1.5, 3.0):
        assert quart(v) == pytest.approx(v**4, rel=1e-12)
    quart(2.0)
    assert " @sq" not in quart.inspect_llvm()


def test_recursion_still_dispatches_not_inlined():
    """Self-recursion must NOT be inlined (would be infinite); it still
    compiles and runs correctly via self-call."""
    @jit
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    assert fib(15) == 610
    assert fib(20) == 6765


def test_inlining_makes_cross_calls_fast():
    """The historical weakness: cross-function calls cost a dispatch.
    After inlining, a helper-heavy kernel should run within a small factor
    of the hand-inlined version (not the ~7x slower dispatch cost)."""
    @jit
    def viahelpers(x):
        s = 0.0
        for i in range(len(x)):
            s += sq(x[i]) + sq(x[i] + 1.0) + cube(x[i])
        return s

    @jit
    def handinlined(x):
        s = 0.0
        for i in range(len(x)):
            v = x[i]
            s += v * v + (v + 1.0) * (v + 1.0) + v * v * v
        return s
    a = np.random.default_rng(1).uniform(-1, 1, 1_000_000)
    viahelpers(a); handinlined(a)
    # correctness first
    assert viahelpers(a) == pytest.approx(handinlined(a), rel=1e-12)

    def best(f):
        t = float("inf")
        for _ in range(5):
            t0 = time.perf_counter(); f(a); t = min(t, time.perf_counter()-t0)
        return t
    th, ti = best(viahelpers), best(handinlined)
    # inlined helpers should be within 1.5x of hand-inlined (was ~7x via dispatch)
    assert th < ti * 1.5, f"helpers {th*1e3:.2f}ms vs hand {ti*1e3:.2f}ms"


def test_helper_with_loop_not_inlined_but_correct():
    """A helper containing a loop is NOT inlinable; it must still work
    (dispatched) and give the right answer."""
    @jit
    def sumto(n):
        s = 0
        for i in range(n):
            s += i
        return s

    @jit
    def uses_loopy(n):
        return sumto(n) + sumto(n)
    # sumto has a loop -> stays a call; result still correct
    assert uses_loopy(100) == 2 * sum(range(100))
