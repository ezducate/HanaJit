"""Differential tests for the lazy fusion engine: numpy-style expressions
compiled as single fused loops with zero temporaries. Every kernel is
checked against numpy and required to have actually compiled."""
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")
A = np.random.default_rng(10).uniform(-2, 2, 3001)
B = np.random.default_rng(11).uniform(0.1, 3, 3001)
IX = np.random.default_rng(12).integers(-50, 50, 2000).astype(np.int64)


def ok(jf, got, ref, rel=1e-9):
    assert got == pytest.approx(ref, rel=rel, abs=1e-12)
    assert len(jf.cache) >= 1, "fell back"


def test_elementwise_chain():
    @jit
    def f(a, b):
        return np.sum(a * b + 2.5 * a - b * b / 3.0)
    ok(f, f(A, B), float((A * B + 2.5 * A - B * B / 3.0).sum()))


def test_ufunc_composition():
    @jit
    def f(a, b):
        return np.sum(np.exp(-a * a) * np.sqrt(np.abs(b)) + np.sin(a))
    ok(f, f(A, B),
       float((np.exp(-A * A) * np.sqrt(np.abs(B)) + np.sin(A)).sum()))


def test_named_fused_expression():
    @jit
    def f(a, b):
        t = a * b - 1.0
        return np.mean(t) + np.max(t)
    t = A * B - 1.0
    ok(f, f(A, B), float(t.mean() + t.max()))


def test_comparisons_and_where():
    @jit
    def f(a, b):
        return np.sum(np.where(a > b, a, b * 0.5))
    ok(f, f(A, B), float(np.where(A > B, A, B * 0.5).sum()))


def test_boolean_reductions():
    @jit
    def f(a):
        return (np.count_nonzero(a > 0.0) + np.sum(a > 1.0)
                + (1 if np.any(a > 1.9) else 0)
                + (10 if np.all(a > -3.0) else 0))
    ref = (int(np.count_nonzero(A > 0)) + int((A > 1.0).sum())
           + (1 if (A > 1.9).any() else 0) + (10 if (A > -3.0).all() else 0))
    ok(f, f(A), ref)


def test_minimum_maximum_clip():
    @jit
    def f(a, b):
        return np.sum(np.minimum(a, b) + np.maximum(a, 0.0)
                      + np.clip(a, -1.0, 1.0))
    ref = float((np.minimum(A, B) + np.maximum(A, 0.0)
                 + np.clip(A, -1, 1)).sum())
    ok(f, f(A, B), ref)


def test_arange_linspace_virtual_arrays():
    @jit
    def f(n):
        return (np.sum(np.arange(n) ** 2)
                + np.sum(np.linspace(0.0, 1.0, n) * 2.0))
    n = 501
    ref = float((np.arange(n) ** 2).sum()
                + (np.linspace(0, 1, n) * 2.0).sum())
    ok(f, float(f(n)), ref)


def test_arange_start_stop_step():
    @jit
    def f(a, b, c):
        return np.sum(np.arange(a, b, c))
    for args in [(0, 50, 3), (5, 100, 7), (10, 11, 5)]:
        ok(f, f(*args), int(np.arange(*args).sum()))


def test_new_reductions_prod_arg_var_std():
    @jit
    def f(a):
        small = a * 0.01 + 1.0
        return (np.prod(small) + np.argmin(a) * 1000.0
                + np.argmax(a) + np.var(a) + np.std(a))
    small = A * 0.01 + 1.0
    ref = float(small.prod() + np.argmin(A) * 1000.0 + np.argmax(A)
                + A.var() + A.std())
    ok(f, f(A), ref, rel=1e-7)


def test_method_forms():
    @jit
    def f(a, b):
        return (a.sum() + b.mean() + (a * b).min()
                + a.argmax() + (a > 0.0).sum())
    ref = float(A.sum() + B.mean() + (A * B).min() + A.argmax()
                + (A > 0).sum())
    ok(f, f(A, B), ref)


def test_slice_store_fused():
    @jit
    def f(x, out):
        out[:] = np.exp(-x * x) * 2.0
        out[0:10] = 0.0
        return 0
    x = A.copy()
    out = np.zeros_like(x)
    f(x, out)
    ref = np.exp(-x * x) * 2.0
    ref[0:10] = 0.0
    assert np.allclose(out, ref)
    assert len(f.cache) == 1


def test_fusion_over_views():
    @jit
    def f(x):
        y = x[10:-10]
        return np.sum(y * y[::-1])
    y = A[10:-10]
    ok(f, f(A), float((y * y[::-1]).sum()))


def test_int_array_fusion():
    @jit
    def f(v):
        return np.sum(v * v + 1) + np.count_nonzero(v % 3 == 0)
    ref = int((IX * IX + 1).sum() + np.count_nonzero(IX % 3 == 0))
    ok(f, f(IX), ref)


def test_np_pi_constant():
    @jit
    def f(a):
        return np.sum(np.sin(a * np.pi) ** 2) / np.e
    ok(f, f(A), float((np.sin(A * np.pi) ** 2).sum() / np.e))


def test_dot_of_expressions():
    @jit
    def f(a, b):
        return np.dot(a * 2.0, b + 1.0)
    ok(f, f(A, B), float(np.dot(A * 2.0, B + 1.0)))
