"""Differential tests: every kernel runs both JIT-compiled and as pure
Python; results must match exactly. Random inputs make this a lightweight
property-based suite.

Known, documented semantic deviations from CPython (not tested here):
- i64 wraparound instead of arbitrary-precision ints
- division by zero traps instead of raising ZeroDivisionError
"""
import math
import random
import warnings
import pytest
from hanajit import jit

random.seed(20260706)
warnings.filterwarnings("ignore")


def check(pyfunc, *argsets, approx=False):
    jf = jit(pyfunc)
    for args in argsets:
        expect = pyfunc(*args)
        got = jf(*args)
        if approx:
            assert got == pytest.approx(expect, rel=1e-12, abs=1e-12), args
        else:
            assert got == expect, (args, got, expect)
    # a silent fallback would compare Python against Python — reject that
    assert len(jf.cache) >= 1, f"{pyfunc.__name__} was never JIT-compiled"


IPAIRS = [(a, b) for a in (-97, -13, -1, 1, 7, 254, 10**9)
          for b in (-11, -3, -1, 1, 5, 89)]
IPAIRS += [(random.randint(-10**6, 10**6),
            random.choice([-7, -2, 3, 11, 1000])) for _ in range(50)]
FPAIRS = [(random.uniform(-100, 100), random.uniform(0.1, 50) *
           random.choice([-1, 1])) for _ in range(50)]


# ---------------- integer arithmetic (incl. negative operands) -------------
def test_int_add_sub_mul():
    def f(a, b):
        return a + b - a * b + (a - b)
    check(f, *IPAIRS)


def test_int_floordiv_matches_python():
    def f(a, b):
        return a // b
    check(f, *IPAIRS)


def test_int_mod_matches_python():
    def f(a, b):
        return a % b
    check(f, *IPAIRS)


def test_int_divmod_identity():
    def f(a, b):
        return (a // b) * b + a % b
    check(f, *IPAIRS)


def test_bitwise_and_shifts():
    pairs = [(random.randint(0, 2**40), random.randint(0, 20))
             for _ in range(30)]
    def f(a, s):
        return ((a & 0xF0F0) | (a ^ 0x1234)) + (a << (s % 8)) + (a >> (s % 8))
    check(f, *pairs)


def test_unary_and_abs():
    def f(a, b):
        return abs(-a) + (-b) + (+a) - abs(b)
    check(f, *IPAIRS)


# ---------------- float arithmetic ------------------------------------------
def test_float_arith():
    def f(x, y):
        return x * y - x / y + (x - y) * 0.5
    check(f, *FPAIRS, approx=True)


def test_float_mod_matches_python():
    def f(x, y):
        return x % y
    check(f, *FPAIRS, approx=True)


def test_float_floordiv_matches_python():
    def f(x, y):
        return x // y
    check(f, *FPAIRS, approx=True)


def test_pow_float():
    pairs = [(random.uniform(0.1, 10), random.uniform(-3, 3))
             for _ in range(20)]
    def f(x, y):
        return x ** y
    check(f, *pairs, approx=True)


def test_int_float_promotion():
    def f(i, x):
        return i + x * 2 - i / 4
    check(f, *[(a, b) for (a, _), (b, __) in
               zip(IPAIRS[:20], FPAIRS[:20])], approx=True)


def test_true_division_always_float():
    def f(a, b):
        return a / b
    jf = jit(f)
    assert isinstance(jf(7, 2), float) and jf(7, 2) == 3.5
    assert len(jf.cache) == 1


# ---------------- comparisons / booleans ------------------------------------
def test_comparisons():
    def f(a, b):
        return ((a < b) + (a <= b) + (a == b) + (a != b) +
                (a > b) + (a >= b))
    check(f, *IPAIRS)


def test_mixed_type_compare():
    def f(i, x):
        return (i < x) + (x <= i) + (i == i)
    check(f, (3, 3.0), (5, 4.9), (-2, -1.5), (7, 7.5))


def test_boolean_ops_and_not():
    def f(a, b):
        return (a > 0 and b > 0) + (a < 0 or b < 0) + (not a > b)
    check(f, *IPAIRS[:30])


def test_ternary():
    def f(a, b):
        return (a if a > b else b) + (1.5 if a < 0 else 2.5)
    check(f, *IPAIRS[:30], approx=True)


# ---------------- control flow ----------------------------------------------
def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps


def test_collatz_many():
    check(collatz_steps, *[(n,) for n in range(2, 200)])


def gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a


def test_gcd():
    pairs = [(random.randint(1, 10**9), random.randint(1, 10**9))
             for _ in range(40)]
    check(gcd, *pairs)


def count_primes(n):
    count = 0
    for i in range(2, n):
        is_p = 1
        j = 2
        while j * j <= i:
            if i % j == 0:
                is_p = 0
                break
            j += 1
        count += is_p
    return count


def test_nested_loops_break():
    check(count_primes, (2,), (3,), (100,), (500,))


def step_range(n):
    s = 0
    for i in range(n, -n, -3):
        s += i
    for i in range(0, n, 7):
        s -= i
    return s


def test_range_variants():
    check(step_range, (0,), (1,), (50,), (997,))


def continue_break_mix(n):
    s = 0
    i = 0
    while i < n:
        i += 1
        if i % 3 == 0:
            continue
        if i > n - 5:
            break
        s += i
    return s


def test_continue_break():
    check(continue_break_mix, (0,), (7,), (100,))


def test_recursion_ackermann_small():
    def ack(m, n):
        if m == 0:
            return n + 1
        if n == 0:
            return ack(m - 1, 1)
        return ack(m - 1, ack(m, n - 1))
    check(ack, (0, 5), (1, 5), (2, 4), (3, 3))


def test_casts_int_float():
    def f(x):
        return int(x * 3.7) + float(int(x)) / 2.0
    check(f, *[(v,) for v, _ in FPAIRS[:20]], approx=True)


def test_float_accumulation_long_loop():
    def acc(n):
        s = 0.0
        for i in range(n):
            s += 1.0 / (i + 1.0)
        return s
    check(acc, (1,), (10,), (5000,), approx=True)


def test_mod_zero_test_peephole_negatives():
    """`x % y == 0` uses a fast path; must still match Python for
    negative operands."""
    def f(a, b):
        return (a % b == 0) + (a % b != 0) * 10
    check(f, *IPAIRS)


def test_mod_zero_reversed_operands():
    def f(a, b):
        return 1 if 0 == a % b else 2
    check(f, *IPAIRS)
