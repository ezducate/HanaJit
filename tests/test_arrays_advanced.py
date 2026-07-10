"""Differential tests for slicing, 2-D indexing, transpose, reshape —
every kernel compared against numpy on the same inputs, and required to
have actually compiled (silent fallback = failure)."""
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def compiled(jf):
    assert len(jf.cache) >= 1, "kernel fell back instead of compiling"


# ------------------------------ slicing -------------------------------------
@jit
def k_slice3(x, a, b, c):
    y = x[a:b:c]
    s = 0.0
    for i in range(len(y)):
        s += y[i]
    return s + len(y) * 1000.0


def test_slice_full_semantics_differential():
    x = np.random.default_rng(0).uniform(-1, 1, 37)
    cases = [(a, b, c) for a in (-45, -10, -1, 0, 3, 12, 36, 50)
             for b in (-45, -7, 0, 5, 20, 37, 99)
             for c in (-4, -1, 1, 2, 5)]
    for a, b, c in cases:
        y = x[a:b:c]
        ref = float(y.sum()) + len(y) * 1000.0
        assert k_slice3(x, a, b, c) == pytest.approx(ref, abs=1e-9), (a, b, c)
    compiled(k_slice3)


def test_slice_open_ended_forms():
    x = np.arange(50, dtype=np.float64)

    @jit
    def upper(x, b):
        y = x[:b]
        return np.sum(y)

    @jit
    def lower(x, a):
        y = x[a:]
        return np.sum(y)

    @jit
    def stepped(x, c):
        y = x[::c]
        return np.sum(y) + len(y)

    for b in (-100, -3, 0, 7, 50, 90):
        assert upper(x, b) == pytest.approx(float(x[:b].sum()))
    for a in (-100, -3, 0, 7, 50, 90):
        assert lower(x, a) == pytest.approx(float(x[a:].sum()))
    for c in (-7, -1, 1, 3):
        assert stepped(x, c) == pytest.approx(float(x[::c].sum()) + len(x[::c]))
    compiled(upper); compiled(lower); compiled(stepped)


def test_reversal_idiom():
    @jit
    def rev_dot(x):
        y = x[::-1]
        return np.dot(x, y)
    a = np.random.default_rng(1).uniform(-1, 1, 501)
    assert rev_dot(a) == pytest.approx(float(np.dot(a, a[::-1])), rel=1e-9)
    compiled(rev_dot)


def test_slice_of_slice():
    @jit
    def ss(x, a, b):
        y = x[a:]
        z = y[:b]
        return np.sum(z)
    x = np.arange(60, dtype=np.float64)
    for a, b in [(5, 10), (0, 60), (30, -5), (-20, 15)]:
        assert ss(x, a, b) == pytest.approx(float(x[a:][:b].sum())), (a, b)
    compiled(ss)


def test_negative_indexing():
    @jit
    def ends(x):
        return x[0] + x[-1] + x[-2] * 10.0
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert ends(a) == 1.0 + 4.0 + 30.0
    compiled(ends)


# ------------------------------ 2-D arrays ----------------------------------
@jit
def k_sum2d(m):
    s = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            s += m[i, j]
    return s


def test_2d_indexing_and_shape():
    m = np.random.default_rng(2).uniform(-1, 1, (23, 17))
    assert k_sum2d(m) == pytest.approx(float(m.sum()), rel=1e-9)
    compiled(k_sum2d)


def test_2d_store():
    @jit
    def fill_diag(m, v):
        for i in range(m.shape[0]):
            m[i, i] = v
        return 0
    m = np.zeros((8, 8))
    fill_diag(m, 4.5)
    assert np.allclose(np.diag(m), 4.5) and m.sum() == pytest.approx(36.0)


def test_row_views():
    @jit
    def row_means(m, out):
        for i in range(m.shape[0]):
            r = m[i]
            out[i] = np.mean(r)
        return 0
    m = np.random.default_rng(3).uniform(0, 1, (12, 40))
    out = np.zeros(12)
    row_means(m, out)
    assert np.allclose(out, m.mean(axis=1))
    compiled(row_means)


def test_transpose_differential():
    @jit
    def t_sum_col(m, j):
        t = m.T
        r = t[j]        # column j of m, as a row of m.T
        return np.sum(r)
    m = np.random.default_rng(4).uniform(-1, 1, (9, 14))
    for j in range(14):
        assert t_sum_col(m, j) == pytest.approx(float(m[:, j].sum())), j
    compiled(t_sum_col)


def test_2d_transposed_argument():
    m = np.random.default_rng(5).uniform(-1, 1, (11, 6))
    assert k_sum2d(m.T) == pytest.approx(float(m.sum()), rel=1e-9)
    assert ("f64[2s]",) in k_sum2d.cache  # strided 2-D specialization exists


# ------------------------------ reshape / ravel -----------------------------
def test_reshape_1d_to_2d():
    @jit
    def as_matrix_trace(x, r, c):
        m = x.reshape(r, c)
        s = 0.0
        for i in range(r):
            s += m[i, i]
        return s
    x = np.arange(36, dtype=np.float64)
    assert as_matrix_trace(x, 6, 6) == pytest.approx(
        float(np.trace(x.reshape(6, 6))))


def test_reshape_minus_one():
    @jit
    def rows_of(x, c):
        m = x.reshape(-1, c)
        return m.shape[0] + np.sum(m)
    x = np.arange(24, dtype=np.float64)
    assert rows_of(x, 6) == 4 + float(x.sum())


def test_ravel_2d():
    @jit
    def flat_last(m):
        f = m.ravel()
        return f[-1] + len(f)
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert flat_last(m) == 11.0 + 12


def test_reshape_roundtrip_with_reductions():
    @jit
    def pipeline(x):
        m = x.reshape(4, -1)
        r = m[2]
        return np.max(r) - np.min(m.ravel())
    x = np.random.default_rng(6).uniform(-9, 9, 32)
    m = x.reshape(4, -1)
    assert pipeline(x) == pytest.approx(float(m[2].max() - x.min()))


# ------------------------------ boundaries ----------------------------------
def test_transpose_reshape_rejected():
    """reshape on a non-contiguous view must fall back (numpy would copy)."""
    @jit
    def bad(m):
        t = m.T
        f = t.ravel()
        return f[0]
    m = np.arange(6, dtype=np.float64).reshape(2, 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert bad(m) == float(m.T.ravel()[0])
        assert any("falling back" in str(x.message) for x in w)


def test_permutation_still_falls_back():
    @jit
    def perm(x):
        p = np.random.permutation(x)   # allocates -> interpreter
        return float(np.sum(p))
    a = np.arange(10, dtype=np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert perm(a) == 45.0
        assert any("falling back" in str(x.message) for x in w)
