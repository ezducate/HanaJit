"""Accuracy contracts: operational and mathematical equivalence with
CPython/numpy, including INPUT domains and OUTPUT return types — not
performance. Where hanajit deliberately deviates (documented in
TESTING.md), the deviation itself is pinned by a test so it can never
change silently."""
import math
import random
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")
random.seed(7)


# ---------------- output types must match CPython exactly -------------------
def test_return_types_match_python():
    cases = [
        (lambda x: x * 2 + 1, 3, int),
        (lambda x: x * 2 + 1, 3.5, float),
        (lambda x: x / 2, 7, float),          # true division: int -> float
        (lambda x: x // 2, 7, int),
        (lambda x: x ** 2, 3, int),
        (lambda x: x ** 0.5, 9.0, float),
        (lambda x: x > 2, 5, bool),
        (lambda x: x == x, 1.5, bool),
    ]
    for fn, arg, ty in cases:
        def named(x, _f=fn):
            return _f(x)
        jf = jit(named)
        got, ref = jf(arg), fn(arg)
        assert type(got) is ty and type(ref) is ty, (arg, ty, type(got))
        assert got == ref


def test_reduction_return_types():
    ai = np.arange(-5, 20, dtype=np.int64)
    af = np.linspace(-1, 1, 25)

    @jit
    def s_i(x): return np.sum(x)
    @jit
    def s_f(x): return np.sum(x)
    @jit
    def am(x): return np.argmax(x)
    @jit
    def anyf(x): return np.any(x > 0.0)
    @jit
    def mn(x): return np.mean(x)
    @jit
    def cnz(x): return np.count_nonzero(x > 0.5)

    assert type(s_i(ai)) is int and s_i(ai) == int(ai.sum())
    assert type(s_f(af)) is float
    assert type(am(af)) is int and am(af) == int(np.argmax(af))
    assert type(anyf(af)) is bool
    assert type(mn(ai)) is float
    assert type(cnz(af)) is int


# ---------------- input domains ---------------------------------------------
def test_bool_inputs_behave_like_python_ints():
    @jit
    def f(x):
        return x * 2 + 1
    assert f(True) == 3 and type(f(True)) is int
    assert f(False) == 1


def test_numpy_scalar_inputs_match_python_semantics():
    def f(x):
        return x * 2 + 1
    jf = jit(f)
    for v in (np.float64(2.5), np.int64(4)):
        assert jf(v) == f(v)          # value-equivalent, however dispatched


def test_bigint_raises_loudly_never_silently_wrong():
    @jit
    def f(x):
        return x * 2 + 1
    f(3)                               # specialize the int path first
    with pytest.raises(OverflowError):
        f(2 ** 70)                     # Python would compute; we refuse
    assert f(5) == 11                  # dispatcher state uncorrupted


def test_special_float_values_propagate():
    @jit
    def f(x):
        return x * 2.0 + 1.0
    assert f(float("inf")) == float("inf")
    assert f(float("-inf")) == float("-inf")
    assert math.isnan(f(float("nan")))
    assert math.copysign(1.0, f(-0.5) + 0.0) == math.copysign(
        1.0, (-0.5) * 2.0 + 1.0)


def test_negative_zero_preserved():
    @jit
    def f(x):
        return x * 1.0
    assert math.copysign(1.0, f(-0.0)) == -1.0


# ---------------- mathematical exactness policy -----------------------------
def test_float_loop_bit_exact_vs_python_without_fastmath():
    """Same IEEE ops in the same order => identical bits, not 'close'."""
    def f(n):
        acc = 0.0
        x = 0.7
        for i in range(n):
            x = 3.7 * x * (1.0 - x)
            acc += x / (i + 1.5) - x * x
        return acc
    jf = jit(f)
    for n in (1, 13, 997, 20000):
        assert jf(n) == f(n), n        # exact ==, no tolerance


def test_sequential_sum_matches_python_order_exactly():
    """Our np.sum is a sequential loop: bit-identical to a Python loop.
    (numpy itself uses pairwise summation, so vs numpy the right standard
    is 1e-12 relative — which the fusion suite applies.)"""
    a = np.random.default_rng(3).uniform(-1, 1, 10001)

    @jit
    def s(x):
        return np.sum(x)
    py = 0.0
    for v in a:
        py += float(v)
    assert s(a) == py                  # exact
    assert s(a) == pytest.approx(float(np.sum(a)), rel=1e-12)


def test_nan_poisons_min_max_like_numpy():
    @jit
    def mn(x): return np.min(x)
    @jit
    def mx(x): return np.max(x)
    for arr in ([3.0, float("nan"), 1.0], [float("nan")],
                [1.0, 2.0, float("nan")], [2.0, 1.0, 3.0]):
        a = np.array(arr)
        for jf, npf in ((mn, np.min), (mx, np.max)):
            got, ref = jf(a), float(npf(a))
            assert (math.isnan(got) and math.isnan(ref)) or got == ref, arr


def test_randomized_divmod_grid_matches_python():
    @jit
    def dm(a, b):
        return (a // b) * 1000000 + a % b
    for _ in range(300):
        a = random.randint(-10**9, 10**9)
        b = random.choice([-97, -13, -3, -1, 1, 2, 7, 1000])
        assert dm(a, b) == (a // b) * 1000000 + a % b, (a, b)


def test_float_mod_floordiv_sign_matches_python():
    @jit
    def fm(a, b):
        return a % b + (a // b) * 1000.0
    for a in (-7.5, -0.3, 0.3, 7.5):
        for b in (-2.5, -0.7, 0.7, 2.5):
            assert fm(a, b) == pytest.approx(a % b + (a // b) * 1000.0,
                                             rel=1e-15), (a, b)


# ---------------- pinned, documented deviations ------------------------------
def test_documented_deviation_i64_wraparound():
    """Python ints are arbitrary precision; compiled kernels are i64 and
    wrap (like numba). Pinned so the contract can never drift silently."""
    @jit
    def sq(x):
        return x * x
    assert sq(2 ** 32) == 0            # 2**64 wraps; Python: 18446744073709551616
