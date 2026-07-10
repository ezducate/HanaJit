import math
import warnings
import pytest
from hanajit import jit


def test_int_arith():
    @jit
    def f(a, b):
        return (a + b) * 3 - a // b + a % b
    assert f(17, 5) == (17 + 5) * 3 - 17 // 5 + 17 % 5


def test_float_and_promotion():
    @jit
    def g(x, n):
        acc = 0.0
        for i in range(n):
            acc += x / (i + 1)
        return acc
    expected = sum(2.5 / (i + 1) for i in range(100))
    assert abs(g(2.5, 100) - expected) < 1e-9


def test_control_flow():
    @jit
    def collatz_len(n):
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        return steps
    assert collatz_len(27) == 111


def test_recursion():
    @jit
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    assert fib(20) == 6765


def test_multi_signature():
    @jit
    def add(a, b):
        return a + b
    assert add(2, 3) == 5
    assert abs(add(2.5, 3.25) - 5.75) < 1e-12
    assert len(add.cache) == 2  # two native specializations


def test_break_continue_ternary_abs():
    @jit
    def f(n):
        s = 0
        for i in range(n):
            if i == 7:
                continue
            if i > 12:
                break
            s += abs(-i) if i % 2 == 0 else i
        return s
    def ref(n):
        s = 0
        for i in range(n):
            if i == 7:
                continue
            if i > 12:
                break
            s += abs(-i) if i % 2 == 0 else i
        return s
    assert f(100) == ref(100)


def test_fallback_full_ecosystem():
    @jit
    def uses_objects(n):
        return sum([x * x for x in range(n)])  # lists: not compilable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert uses_objects(10) == 285
        assert any("falling back" in str(x.message) for x in w)


def test_no_fallback_raises():
    from hanajit import UnsupportedError
    @jit(fallback=False)
    def bad(n):
        return [n]  # noqa
    with pytest.raises(UnsupportedError):
        bad(3)


@pytest.mark.parametrize("vendor,marker", [
    ("cuda", "NVPTX"), ("amd", "amdgcn"), ("intel", "OpCapability")])
def test_gpu_emission(vendor, marker):
    @jit(target=vendor)
    def axpy(a, x, y):
        return a * x + y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert axpy(2.0, 3.0, 4.0) == 10.0  # runs via fallback
    v, text, native = axpy.inspect_gpu()
    assert v == vendor and native and marker in text


def test_specialize_raw_builtin():
    @jit
    def mul3(a, b):
        return a * b * 3
    fast = mul3.specialize(int, int)
    assert fast(4, 5) == 60
    assert mul3(4, 5) == 60
    ffast = mul3.specialize(float, float)
    assert abs(ffast(0.5, 2.0) - 3.0) < 1e-12


def test_dispatch_mode_matches_platform():
    import sys, sysconfig, struct
    @jit
    def h(a, b):
        return a + b
    native_ok = (sys.implementation.name == "cpython"
                 and sys.version_info >= (3, 12)
                 and struct.calcsize("P") == 8
                 and not sysconfig.get_config_var("Py_GIL_DISABLED"))
    name = type(h).__name__
    if native_ok:
        assert name == "HanaFunction", name   # native vectorcall dispatch
    else:
        assert name == "Dispatcher", name      # graceful Python fallback
    assert h(2, 3) == 5


def test_fastmath_opt_in():
    @jit(fastmath=True)
    def dot3(a, b, c):
        return a * a + b * b + c * c
    assert abs(dot3(1.0, 2.0, 3.0) - 14.0) < 1e-9


def test_nogil_releases_gil():
    """A nogil kernel running in a background thread must not block the
    main thread (works even on 1 core: a GIL-holding native call would
    freeze Python entirely until it returns)."""
    import threading, time

    @jit(nogil=True)
    def spin(n):
        acc = 0.0
        for i in range(n):
            acc += (i % 7) * 0.5
        return acc

    @jit  # GIL-holding control
    def spin_gil(n):
        acc = 0.0
        for i in range(n):
            acc += (i % 7) * 0.5
        return acc

    spin(10); spin_gil(10)  # compile

    def probe(kernel):
        t = threading.Thread(target=kernel, args=(600_000_000,))
        t.start()
        ticks = 0
        while t.is_alive():
            ticks += 1       # only possible while the GIL is free
        t.join()
        return ticks

    nogil_ticks = probe(spin)
    gil_ticks = probe(spin_gil)     # frozen while native code holds the GIL
    assert nogil_ticks > 10 * max(gil_ticks, 1), (nogil_ticks, gil_ticks)


def test_pointer_kernel_saxpy():
    import numpy as np

    @jit(signature="f64*, f64*, f64, i64", nogil=True)
    def saxpy(x, y, a, n):
        for i in range(n):
            y[i] = a * x[i] + y[i]
        return 0

    x = np.arange(1000, dtype=np.float64)
    y = np.ones(1000, dtype=np.float64)
    expect = 2.5 * x + y
    saxpy(x.ctypes.data, y.ctypes.data, 2.5, len(x))
    assert np.allclose(y, expect)


def test_pmap_threads():
    from hanajit import pmap

    @jit(nogil=True)
    def work(n, seed):
        acc = float(seed)
        for i in range(n):
            acc = 3.999 * (acc / 10.0) * (1.0 - acc / 10.0) * 10.0
        return acc

    work(10, 1)
    args = [(200000, s) for s in range(8)]
    par = pmap(work, args, workers=4)
    seq = [work(*a) for a in args]
    assert par == seq


def test_gpu_thread_indexing_cuda():
    @jit(target="cuda", signature="f64*, f64*, f64, i64")
    def gpu_saxpy(x, y, a, n):
        i = block_id() * block_dim() + thread_id()
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0
    v, ptx, native = gpu_saxpy.inspect_gpu()
    assert native and "tid.x" in ptx and "ctaid.x" in ptx
    assert "ld.global" in ptx or "ld.f64" in ptx  # real memory traffic


def test_gpu_thread_indexing_amd():
    # AMD exposes workitem/workgroup ids; workgroup size is passed in
    # (reading it from the dispatch packet is a roadmap item)
    @jit(target="amd", signature="f64*, f64*, f64, i64, i64")
    def gpu_saxpy(x, y, a, n, bdim):
        i = block_id() * bdim + thread_id()
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0
    v, isa, native = gpu_saxpy.inspect_gpu()
    assert native and v == "amd"
    assert "workitem" in isa.lower() or "v0" in isa.lower(), isa[:400]


def test_disk_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HANAJIT_CACHE_DIR", str(tmp_path))

    @jit(cache=True)
    def cfun(a, b):
        return a * b + 7

    assert cfun(6, 7) == 49
    assert any(f.suffix == ".o" for f in tmp_path.iterdir())  # object saved

    # a fresh dispatcher over the same source must warm-start from disk:
    from hanajit.decorator import Dispatcher
    d = Dispatcher(cfun.__wrapped__, cache=True)
    assert d(6, 7) == 49
    assert not d.modules   # codegen skipped -> loaded from cache


def test_disk_cache_key_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HANAJIT_CACHE_DIR", str(tmp_path))

    @jit(cache=True)
    def kfun(x):
        return x * 2.0

    @jit(cache=True, fastmath=True)   # different flags -> different key
    def kfun2(x):
        return x * 2.0

    assert kfun(1.5) == kfun2(1.5) == 3.0
    objs = [f for f in tmp_path.iterdir() if f.suffix == ".o"]
    assert len(objs) == 2


def test_lru_cache_composes():
    import functools

    @functools.lru_cache(maxsize=None)
    @jit
    def slowfn(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    assert slowfn(1000) == 499500
    assert slowfn(1000) == 499500          # memoized hit
    assert slowfn.cache_info().hits >= 1   # result caching layers on top


def test_cross_compile_apple_silicon():
    """CPU kernels must codegen cleanly for arm64 macOS (Apple Silicon)."""
    @jit
    def kern(a, x):
        s = 0.0
        for i in range(a):
            s += x * i - s * 0.5
        return s
    kern(10, 1.5)
    from hanajit.backends.cpu import emit_cross
    mod = next(iter(kern.modules.values()))
    asm = emit_cross(mod, "arm64-apple-darwin", cpu="apple-m1")
    assert "ret" in asm
    assert "fmadd" in asm or "fmul" in asm  # real arm64 FP instructions


def test_metal_msl_generation():
    @jit(target="metal", signature="f64*, f64*, f64, i64")
    def msaxpy(x, y, a, n):
        i = block_id() * block_dim() + thread_id()
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0
    vendor, src, native = msaxpy.inspect_gpu()
    assert vendor == "metal" and not native
    assert "kernel void msaxpy" in src
    assert "device float* x [[buffer(0)]]" in src
    assert "thread_position_in_threadgroup" in src
    assert "threadgroup_position_in_grid" in src
    assert "constant float& a" in src


def test_metal_control_flow_kernel():
    @jit(target="metal", signature="i64*, i64")
    def collatz_kernel(out, n):
        i = thread_id()
        if i < n:
            m = i + 2
            steps = 0
            while m != 1:
                if m % 2 == 0:
                    m = m // 2
                else:
                    m = 3 * m + 1
                steps += 1
            out[i] = steps
        return 0
    _, src, _ = collatz_kernel.inspect_gpu()
    assert "while (" in src and "device long* out" in src


def test_detect_always_has_cpu():
    from hanajit import detect
    targets = [t for t, _ in detect()]
    assert targets[-1] == "cpu"


def test_auto_scalar_resolves_to_cpu():
    @jit(target="auto")
    def afun(a, b):
        return a * b + 3
    assert afun(6, 7) == 45
    assert getattr(afun, "dispatcher", afun).target == "cpu"


def test_auto_gpu_kernel_no_gpu_raises_clearly():
    import hanajit.backends.detect as det
    if det.best_gpu() is not None:
        pytest.skip("machine has a GPU runtime")
    with pytest.raises(ValueError, match="no GPU runtime"):
        @jit(target="auto", signature="f64*, i64")
        def gkern(x, n):
            i = thread_id()
            if i < n:
                x[i] = x[i] * 2.0
            return 0


def test_auto_gpu_kernel_with_detected_gpu(monkeypatch):
    import hanajit.backends.detect as det
    monkeypatch.setattr(det, "best_gpu", lambda: "cuda")
    @jit(target="auto", signature="f64*, i64")
    def gkern2(x, n):
        i = block_id() * block_dim() + thread_id()
        if i < n:
            x[i] = x[i] * 2.0
        return 0
    v, text, native = gkern2.inspect_gpu()
    assert v == "cuda" and native and "tid.x" in text


def test_env_override(monkeypatch):
    from hanajit.backends import detect as det
    monkeypatch.setenv("HANAJIT_TARGET", "intel")
    det.detect.cache_clear()
    assert det.detect() == [("intel", "forced via HANAJIT_TARGET")]
    monkeypatch.delenv("HANAJIT_TARGET")
    det.detect.cache_clear()
