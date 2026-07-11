"""Tests for the experimental narrow-integer compute mode."""
import warnings
import numpy as np
import pytest
from hanajit import jit
from hanajit.errors import UnsupportedError


def _isum(x):
    acc = 0
    for i in range(len(x)):
        acc += x[i]
    return acc


def test_narrow_requires_confirmation():
    f = jit(_isum)
    with pytest.raises(UnsupportedError):
        f.narrow(np.ones(10, dtype=np.int8))


def test_narrow_int8_exact():
    f = jit(_isum)
    x = np.random.default_rng(0).integers(-5, 5, 1_000_000).astype(np.int8)
    assert f.narrow(x, confirmed=True) == int(x.astype(np.int64).sum())


def test_narrow_int16_exact():
    f = jit(_isum)
    x = np.random.default_rng(1).integers(-300, 300, 1_000_000).astype(np.int16)
    assert f.narrow(x, confirmed=True) == int(x.astype(np.int64).sum())


def test_narrow_int32_exact():
    f = jit(_isum)
    x = np.random.default_rng(2).integers(-1000, 1000, 1_000_000).astype(np.int32)
    assert f.narrow(x, confirmed=True) == int(x.astype(np.int64).sum())


def test_narrow_no_overflow_on_large_sum():
    # 1M values of 100 as int8 sums to 100,000,000, far beyond int8 range.
    # A naive narrow accumulator would wrap; the wide accumulator must not.
    f = jit(_isum)
    x = np.full(1_000_000, 100, dtype=np.int8)
    assert f.narrow(x, confirmed=True) == 100_000_000


def test_narrow_negative_values_exact():
    f = jit(_isum)
    x = np.full(500_000, -128, dtype=np.int8)   # most-negative int8
    assert f.narrow(x, confirmed=True) == -128 * 500_000


def test_narrow_tail_handling_non_multiple_length():
    # length not a multiple of the SIMD vector width exercises the scalar tail
    f = jit(_isum)
    for n in (1, 7, 31, 33, 1000, 4097):
        x = np.random.default_rng(n).integers(-5, 5, n).astype(np.int8)
        assert f.narrow(x, confirmed=True) == int(x.astype(np.int64).sum()), n


def test_narrow_unsupported_dtype_falls_back():
    f = jit(_isum)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = f.narrow(np.ones(10, dtype=np.float64), confirmed=True)
    assert r is None
    assert any("narrow" in str(w.message) for w in rec)


def test_narrow_rejects_2d_array():
    f = jit(_isum)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = f.narrow(np.ones((10, 10), dtype=np.int8), confirmed=True)
    assert r is None
