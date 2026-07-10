"""NumPy-native arguments and scipy LowLevelCallable integration."""
import warnings
import numpy as np
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def test_numpy_arrays_direct_lazy():
    """Arrays pass directly — no signature, no .ctypes.data."""
    @jit
    def dot(x, y, n):
        s = 0.0
        for i in range(n):
            s += x[i] * y[i]
        return s
    a = np.arange(1000, dtype=np.float64)
    b = np.ones(1000, dtype=np.float64) * 0.5
    assert dot(a, b, 1000) == pytest.approx(float(np.dot(a, b)))
    assert len(dot.cache) == 1  # compiled, not fallback


def test_numpy_int64_arrays():
    @jit
    def isum(x, n):
        s = 0
        for i in range(n):
            s += x[i]
        return s
    a = np.arange(500, dtype=np.int64)
    assert isum(a, 500) == 124750


def test_numpy_write_kernel_mutates():
    @jit
    def scale(x, a, n):
        for i in range(n):
            x[i] = x[i] * a
        return 0
    a = np.ones(100, dtype=np.float64)
    scale(a, 3.0, 100)
    assert np.allclose(a, 3.0)


def test_dtype_specializations_do_not_collide():
    @jit
    def first(x):
        return x[0]
    f = np.array([2.5, 1.0])
    i = np.array([7, 9], dtype=np.int64)
    assert first(f) == 2.5
    assert first(i) == 7
    assert isinstance(first(i), int) and isinstance(first(f), float)
    assert len(first.cache) == 2  # separate f64*/i64* kernels


def test_unsupported_dtype_falls_back_correctly():
    @jit
    def total(x, n):
        s = 0.0
        for i in range(n):
            s += x[i]
        return s
    a32 = np.ones(50, dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert total(a32, 50) == pytest.approx(50.0)  # interpreter handles it
        assert any("falling back" in str(x.message) for x in w)


def test_strided_arguments_now_compile():
    @jit
    def total2(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i]
        return s
    a = np.arange(100, dtype=np.float64)[::2]     # strided view argument
    r = np.arange(100, dtype=np.float64)[::-3]    # negative-stride view
    assert total2(a) == pytest.approx(float(a.sum()))
    assert total2(r) == pytest.approx(float(r.sum()))
    assert len(total2.cache) == 1  # one strided specialization, compiled


def test_signature_declared_accepts_arrays():
    @jit(signature="f64*, f64, i64")
    def saxpy1(x, a, n):
        for i in range(n):
            x[i] = a * x[i]
        return 0
    arr = np.ones(64)
    saxpy1(arr, 2.0, 64)          # ndarray auto-converted
    saxpy1(arr.ctypes.data, 2.0, 64)  # raw address still works
    assert np.allclose(arr, 4.0)


def test_signature_dtype_mismatch_fallback():
    @jit(signature="f64*, i64")
    def s2(x, n):
        s = 0.0
        for i in range(n):
            s += x[i]
        return s
    wrong = np.arange(10, dtype=np.int64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert s2(wrong, 10) == 45  # falls back, still right
        assert any("falling back" in str(x.message) for x in w)


def test_scipy_lowlevelcallable_quad():
    scipy = pytest.importorskip("scipy")
    from scipy.integrate import quad

    @jit
    def gauss(x):
        return 2.718281828459045 ** (-x * x)

    llc = gauss.scipy_callable()
    got, _ = quad(llc, -6.0, 6.0)
    ref, _ = quad(lambda x: np.exp(-x * x), -6.0, 6.0)
    assert got == pytest.approx(ref, rel=1e-9)


def test_scipy_callable_rejects_int_kernel():
    pytest.importorskip("scipy")
    from hanajit import UnsupportedError

    @jit
    def intk(x):
        return int(x) * 2

    with pytest.raises(UnsupportedError):
        intk.scipy_callable()


def test_fused_loop_matches_numpy_chain():
    @jit
    def fused(a, b, n):
        s = 0.0
        for i in range(n):
            s += a[i] * b[i] + 2.5 * a[i] - b[i] * b[i]
        return s
    a = np.random.default_rng(0).uniform(-1, 1, 10_000)
    b = np.random.default_rng(1).uniform(-1, 1, 10_000)
    ref = float((a * b + 2.5 * a - b * b).sum())
    assert fused(a, b, 10_000) == pytest.approx(ref, rel=1e-9)


def test_math_functions_differential():
    import math

    def f(x):
        return (math.sqrt(x) + math.exp(-x) + math.sin(x) * math.cos(x)
                + math.log(x + 1.0) + math.floor(x) + math.pow(x, 1.5))
    jf = jit(f)
    for v in (0.1, 1.0, 2.7, 9.9, 42.0):
        assert jf(v) == pytest.approx(f(v), rel=1e-12)
    assert len(jf.cache) == 1  # actually compiled


def test_np_scalar_math_in_kernel():
    @jit
    def gauss_sum(n):
        s = 0.0
        for i in range(n):
            x = i * 0.01 - 5.0
            s += np.exp(-x * x)
        return s
    ref = sum(np.exp(-(i * 0.01 - 5.0) ** 2) for i in range(1000))
    assert gauss_sum(1000) == pytest.approx(float(ref), rel=1e-9)


def test_math_in_metal_kernel():
    @jit(target="metal", signature="f64*, i64")
    def mk(x, n):
        i = thread_id()
        if i < n:
            x[i] = np.sqrt(np.exp(x[i]))
        return 0
    _, src, _ = mk.inspect_gpu()
    assert "sqrt(" in src and "exp(" in src


def test_np_sum_now_compiles_composed():
    @jit
    def uses_npsum(x):
        return float(np.sum(x)) * 2.0
    a = np.arange(10, dtype=np.float64)
    assert uses_npsum(a) == 90.0
    assert len(uses_npsum.cache) == 1


def test_array_arithmetic_now_fuses():
    """`np.sum(x * y)` compiles to one fused loop — no temporaries."""
    @jit
    def whole(x, y):
        return float(np.sum(x * y))
    a = np.arange(5, dtype=np.float64)
    assert whole(a, a) == 30.0
    assert len(whole.cache) == 1


def test_np_zeros_falls_back():
    """Allocation inside kernels needs a memory runtime — falls back."""
    @jit
    def alloc(n):
        z = np.zeros(n)
        return float(z.sum()) + n
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert alloc(5) == 5.0
        assert any("falling back" in str(x.message) for x in w)


def test_len_in_kernel():
    @jit
    def total_len(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i]
        return s
    a = np.arange(1000, dtype=np.float64)
    assert total_len(a) == pytest.approx(float(a.sum()))
    assert len(total_len.cache) == 1


def _red_sum(x):
    return np.sum(x)


def _red_min(x):
    return np.min(x)


def _red_max(x):
    return np.max(x)


def _red_mean(x):
    return np.mean(x)


@pytest.mark.parametrize("fn,ref", [
    (_red_sum, lambda a: float(np.sum(a))),
    (_red_min, lambda a: float(np.min(a))),
    (_red_max, lambda a: float(np.max(a))),
    (_red_mean, lambda a: float(np.mean(a)))])
def test_np_reductions_compile(fn, ref):
    jf = jit(fn)
    a = np.random.default_rng(3).uniform(-5, 5, 4001)
    assert jf(a) == pytest.approx(ref(a), rel=1e-12)
    assert len(jf.cache) == 1  # compiled, not fallback


def test_np_sum_int_array():
    @jit
    def s(x):
        return np.sum(x)
    a = np.arange(-50, 500, dtype=np.int64)
    assert s(a) == int(a.sum()) and isinstance(s(a), int)


def test_np_dot_compiled():
    @jit
    def d(x, y):
        return np.dot(x, y)
    a = np.random.default_rng(1).uniform(-1, 1, 3000)
    b = np.random.default_rng(2).uniform(-1, 1, 3000)
    assert d(a, b) == pytest.approx(float(np.dot(a, b)), rel=1e-9)
    assert len(d.cache) == 1


def test_reductions_mixed_with_loops():
    @jit
    def normalize_score(x):
        m = np.mean(x)
        s = 0.0
        for i in range(len(x)):
            s += (x[i] - m) * (x[i] - m)
        return np.sqrt(s / len(x))
    a = np.random.default_rng(9).uniform(0, 10, 2000)
    assert normalize_score(a) == pytest.approx(float(np.std(a)), rel=1e-9)


def test_gpu_signature_rejects_array_tokens():
    from hanajit import UnsupportedError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        @jit(target="cuda", signature="f64[], i64")
        def g(x, n):
            i = thread_id()
            if i < n:
                x[i] = 1.0
            return 0
        assert g.inspect_gpu() is None  # eager emit refused cleanly


def test_evolve_scalar_kernel_stays_correct():
    @jit
    def mixwork(n):
        acc = 0
        for i in range(n):
            if i % 3 == 0:
                acc += i * 2
            else:
                acc -= i % 7
        return acc
    ref = mixwork(50000)
    report = mixwork.evolve(50000, generations=2, population=5, reps=2)
    assert set(report) >= {"baseline_ms", "best_ms", "speedup", "genome",
                           "installed"}
    assert mixwork(50000) == ref          # evolved install is equivalent
    assert mixwork(12345) == mixwork.__wrapped__(12345)
    assert report["speedup"] >= 1.0       # never installs a regression


def test_evolve_array_kernel():
    @jit
    def sq_sum(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i]
        return s
    a = np.random.default_rng(4).uniform(-1, 1, 200_000)
    ref = sq_sum(a)
    report = sq_sum.evolve(a, generations=2, population=5, reps=2)
    assert sq_sum(a) == ref               # exact: no fastmath by default
    assert report["speedup"] >= 1.0


def test_evolve_fastmath_gated_and_validated():
    @jit
    def fsum(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * 1.0000001
        return s
    a = np.random.default_rng(5).uniform(0, 1, 300_000)
    ref = fsum(a)
    report = fsum.evolve(a, generations=3, population=6, reps=3,
                         allow_fastmath=True)
    got = fsum(a)
    assert got == pytest.approx(ref, rel=1e-9)  # tolerance-gated equivalence
    assert report["speedup"] >= 1.0


def test_slice_in_reduction():
    @jit
    def mid_sum(x):
        return np.sum(x[100:900])
    a = np.random.default_rng(6).uniform(-2, 2, 1000)
    assert mid_sum(a) == pytest.approx(float(a[100:900].sum()), rel=1e-12)
    assert len(mid_sum.cache) == 1


def test_slice_open_ended_and_len():
    @jit
    def tails(x, k):
        return np.sum(x[:k]) + np.sum(x[k:]) + len(x[k:])
    a = np.arange(500, dtype=np.float64)
    assert tails(a, 123) == pytest.approx(float(a.sum()) + (500 - 123))


def test_slice_binding_and_loop():
    @jit
    def window_energy(x, lo, hi):
        w = x[lo:hi]
        s = 0.0
        for i in range(len(w)):
            s += w[i] * w[i]
        return s
    a = np.random.default_rng(8).uniform(-1, 1, 2000)
    assert window_energy(a, 250, 1750) == pytest.approx(
        float((a[250:1750] ** 2).sum()), rel=1e-12)


def test_slice_write_through_view():
    @jit
    def zero_middle(x, lo, hi):
        w = x[lo:hi]
        for i in range(len(w)):
            w[i] = 0.0
        return np.sum(x)
    a = np.ones(100)
    assert zero_middle(a, 20, 80) == pytest.approx(40.0)
    assert a[19] == 1.0 and a[20] == 0.0 and a[79] == 0.0 and a[80] == 1.0


def test_nested_slice():
    @jit
    def nested(x):
        return np.sum(x[10:90][20:60])   # == x[30:70]
    a = np.arange(100, dtype=np.float64)
    assert nested(a) == pytest.approx(float(a[30:70].sum()))


def test_strided_slice_compiles():
    @jit
    def strided(x):
        return np.sum(x[::2])
    a = np.arange(10, dtype=np.float64)
    assert strided(a) == 20.0
    assert len(strided.cache) == 1   # strided views compile ("1s" kind)


def test_evolve_persists_tuned_binary(tmp_path, monkeypatch):
    """With cache=True, the evolved winner overwrites the disk-cache
    object: fresh dispatchers warm-start already tuned and equivalent."""
    monkeypatch.setenv("HANAJIT_CACHE_DIR", str(tmp_path))

    @jit(cache=True)
    def pk(n):
        acc = 0.0
        for i in range(n):
            acc += (i % 9) * 0.125
        return acc

    ref = pk(200_000)
    rep = pk.evolve(200_000, generations=2, population=5, reps=2)
    assert rep["evaluations"] >= 3
    if rep["installed"]:
        assert rep["persisted"]
    from hanajit.decorator import Dispatcher
    d2 = Dispatcher(pk.__wrapped__, cache=True)   # fresh: loads from disk
    assert d2(200_000) == ref
    assert not d2.modules                          # cache hit path
