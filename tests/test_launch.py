"""GPU kernel execution via f.launch(). Hardware-dependent tests skip
cleanly on machines without the corresponding runtime; the error-path
and SPIR-V-generator tests run everywhere."""
import warnings

import pytest

np = pytest.importorskip("numpy")
warnings.filterwarnings("ignore")


def _runtime(vendor):
    from hanajit.backends import rt
    return rt.get_runtime(vendor)


def _saxpy(target):
    from hanajit import jit

    @jit(target=target, signature="f64*, f64*, f64, i64")
    def saxpy(y, x, a, n):
        i = block_id() * block_dim() + thread_id()
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0

    return saxpy


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan", "amd",
                                    "metal"])
def test_saxpy_executes_on_device(vendor):
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    saxpy = _saxpy(vendor)
    n = 65_537                      # deliberately not a block multiple
    x = np.random.rand(n)
    y0 = np.random.rand(n)
    y = y0.copy()
    saxpy.launch(y, x, 2.0, n)
    if vendor == "metal":           # f64 lowers to float32 on Metal
        assert np.allclose(y, 2.0 * x + y0, rtol=1e-5, atol=1e-5)
    else:
        assert np.allclose(y, 2.0 * x + y0)


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan"])
def test_loop_math_and_int_semantics(vendor):
    """Per-thread loop, sqrt/floor, and Python floor-division/modulo
    semantics (negative operands) must match the CPython result exactly.
    Only natively-lowerable math here: transcendentals need libdevice on
    NVPTX and are covered separately."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    import math
    from hanajit import jit

    def kernel(out, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            acc = 0.0
            for j in range(4):
                acc += math.floor(0.25 * (i + j)) * math.sqrt(1.0 + j)
            k = i - 7
            acc += (k // 3) + (k % 3)      # floor semantics for k < 0
            out[i] = acc
        return 0

    jk = jit(target=vendor, signature="f64*, i64")(kernel)
    n = 4096
    out = np.zeros(n)
    jk.launch(out, n)
    expected = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(4):
            acc += math.floor(0.25 * (i + j)) * math.sqrt(1.0 + j)
        k = i - 7
        acc += (k // 3) + (k % 3)
        expected[i] = acc
    assert np.allclose(out, expected)


@pytest.mark.parametrize("vendor,rtol", [("intel", 1e-12),
                                         ("vulkan", 1e-5),
                                         ("cuda", 1e-12)])
def test_transcendental_math(vendor, rtol):
    """sin/exp across targets. Vulkan computes these at f32
    (GLSL.std.450 defines them for 16/32-bit floats only), so it gets a
    float32-level tolerance; Level Zero evaluates at f64; CUDA links
    NVIDIA's libdevice (skipped when none is installed)."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    if vendor == "cuda":
        from hanajit.backends.gpu import find_libdevice
        if find_libdevice() is None:
            pytest.skip("cuda: no libdevice (pip install "
                        "nvidia-cuda-nvcc-cu12, or install a CUDA toolkit)")
    import math
    from hanajit import jit

    def kernel(out, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            out[i] = math.sin(0.001 * i) + math.exp(-0.001 * i)
        return 0

    jk = jit(target=vendor, signature="f64*, i64")(kernel)
    n = 2048
    out = np.zeros(n)
    jk.launch(out, n)
    i = np.arange(n)
    expected = np.sin(0.001 * i) + np.exp(-0.001 * i)
    assert np.allclose(out, expected, rtol=rtol, atol=1e-6)


def test_launch_requires_gpu_target():
    from hanajit import jit, UnsupportedError

    @jit(signature="f64*, i64")
    def k(x, n):
        for i in range(n):
            x[i] = 1.0
        return 0

    with pytest.raises(UnsupportedError):
        k.launch(np.zeros(4), 4)


def test_launch_argument_validation():
    vendor = next((v for v in ("cuda", "intel", "vulkan")
                   if _runtime(v) is not None), None)
    if vendor is None:
        pytest.skip("no GPU runtime on this machine")
    from hanajit import UnsupportedError
    saxpy = _saxpy(vendor)
    y = np.zeros(8)
    with pytest.raises(UnsupportedError):   # wrong dtype
        saxpy.launch(np.zeros(8, dtype=np.float32), y, 1.0, 8)
    with pytest.raises(UnsupportedError):   # arity
        saxpy.launch(y, y, 1.0)


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan", "amd",
                                    "metal"])
def test_device_array_resident_launches(vendor):
    """DeviceArray keeps data on the GPU: repeated launches accumulate
    without host round-trips, and to_host() sees the final state."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    saxpy = _saxpy(vendor)
    n = 10_000
    x = np.random.rand(n)
    y0 = np.random.rand(n)
    xd = saxpy.to_device(x)
    yd = saxpy.to_device(y0)
    for _ in range(4):
        saxpy.launch(yd, xd, 0.5, n)
    y = yd.to_host()
    expected = y0 + 0.5 * x * 4
    tol = dict(rtol=1e-4, atol=1e-4) if vendor == "metal" else {}
    assert np.allclose(y, expected, **tol)
    # host array untouched until to_host with out=
    assert np.allclose(x, xd.to_host())
    xd.free()
    yd.free()
    xd.free()  # double free is a no-op
    from hanajit import UnsupportedError
    with pytest.raises(UnsupportedError):
        saxpy.launch(yd, xd, 0.5, n)   # freed arrays refuse to launch


def test_device_array_validation():
    vendor = next((v for v in ("cuda", "intel", "vulkan")
                   if _runtime(v) is not None), None)
    if vendor is None:
        pytest.skip("no GPU runtime on this machine")
    from hanajit import UnsupportedError
    saxpy = _saxpy(vendor)
    with pytest.raises(UnsupportedError):   # wrong dtype
        saxpy.to_device(np.zeros(8, dtype=np.float32))
    d = saxpy.to_device(np.zeros(8, dtype=np.int64))
    with pytest.raises(UnsupportedError):   # f64* slot given i64 buffer
        saxpy.launch(d, np.zeros(8), 1.0, 8)
    d.free()


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan", "amd",
                                    "metal"])
def test_shared_memory_block_reduction(vendor):
    """Two-stage dot product: shared_f64 tree reduction per workgroup +
    barrier(), partial sums summed on the host. The portable reduction
    pattern — needs no atomics."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    from hanajit import jit

    @jit(target=vendor, signature="f64*, f64*, f64*, i64")
    def dot_partials(partials, a, b, n):
        tid = thread_id()
        i = block_id() * block_dim() + tid
        s = shared_f64(256)
        acc = 0.0
        if i < n:
            acc = a[i] * b[i]
        s[tid] = acc
        barrier()
        step = 128
        while step > 0:
            if tid < step:
                s[tid] = s[tid] + s[tid + step]
            barrier()
            step = step // 2
        if tid == 0:
            partials[block_id()] = s[0]
        return 0

    n = 100_000
    a = np.random.rand(n)
    b = np.random.rand(n)
    blocks = -(-n // 256)
    partials = np.zeros(blocks)
    dot_partials.launch(partials, a, b, n, grid=blocks, block=256)
    rtol = 1e-3 if vendor == "metal" else 1e-12   # metal computes at f32
    assert np.isclose(partials.sum(), np.dot(a, b), rtol=rtol)


@pytest.mark.parametrize("vendor", ["cuda", "amd", "intel", "vulkan"])
def test_atomic_add(vendor):
    """i64 atomics everywhere; f64 via atomicrmw (cuda/amd), a CAS loop
    on the bit pattern (intel/opencl), or VK_EXT_shader_atomic_float
    (vulkan). Devices lacking the capability raise UnsupportedError with
    the missing feature named — treated as a skip."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    from hanajit import jit, UnsupportedError

    @jit(target=vendor, signature="i64*, i64*, i64")
    def hist(out, x, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            atomic_add(out, x[i] % 8, 1)
        return 0

    x = np.random.randint(0, 1000, 50_000).astype(np.int64)
    out = np.zeros(8, dtype=np.int64)
    try:
        hist.launch(out, x, len(x), grid=-(-len(x) // 256))
    except UnsupportedError as e:
        pytest.skip(f"{vendor}: {e}")
    assert (out == np.bincount(x % 8, minlength=8)).all()

    @jit(target=vendor, signature="f64*, f64*, i64")
    def fsum(out, x, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            atomic_add(out, 0, x[i])
        return 0

    xf = np.random.rand(10_000)
    outf = np.zeros(1)
    try:
        fsum.launch(outf, xf, len(xf), grid=-(-len(xf) // 256))
    except UnsupportedError as e:
        pytest.skip(f"{vendor}: f64 atomics: {e}")
    assert np.isclose(outf[0], xf.sum())


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan", "amd",
                                    "metal"])
def test_2d_kernel(vendor):
    """y-axis thread intrinsics: transpose-add over a 2-D grid, flat
    row-major buffers."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    from hanajit import jit

    @jit(target=vendor, signature="f64*, f64*, i64, i64")
    def transpose_add(out, a, w, h):
        cx = block_id() * block_dim() + thread_id()
        cy = block_id_y() * block_dim_y() + thread_id_y()
        if cx < w and cy < h:
            out[cx * h + cy] = out[cx * h + cy] + a[cy * w + cx]
        return 0

    w, h = 96, 64
    a = np.random.rand(h * w)
    out = np.zeros(w * h)
    transpose_add.launch(out, a, w, h,
                         grid=(-(-w // 16), -(-h // 16)), block=(16, 16))
    expected = a.reshape(h, w).T.ravel()
    tol = dict(rtol=1e-5, atol=1e-5) if vendor == "metal" else {}
    assert np.allclose(out, expected, **tol)


def test_device_index_env(monkeypatch):
    from hanajit.backends import rt
    monkeypatch.delenv("HANAJIT_CUDA_DEVICE", raising=False)
    assert rt.device_index("cuda") == 0
    monkeypatch.setenv("HANAJIT_CUDA_DEVICE", "1")
    assert rt.device_index("cuda") == 1
    monkeypatch.setenv("HANAJIT_CUDA_DEVICE", "junk")
    assert rt.device_index("cuda") == 0
    rt.reset()   # cache clearing is safe to call at any time
    assert rt.get_runtime("nonexistent") is None


@pytest.mark.parametrize("vendor", ["cuda", "intel", "vulkan", "amd",
                                    "metal"])
def test_async_launches(vendor):
    """sync=False queues; synchronize()/to_host() sees the final state;
    transient numpy arrays are refused for async launches."""
    if _runtime(vendor) is None:
        from hanajit.backends import rt
        pytest.skip(f"{vendor}: {rt.why_unavailable(vendor)}")
    from hanajit import UnsupportedError
    saxpy = _saxpy(vendor)
    n = 50_000
    x = np.random.rand(n)
    y0 = np.random.rand(n)
    xd = saxpy.to_device(x)
    yd = saxpy.to_device(y0)
    for _ in range(8):
        saxpy.launch(yd, xd, 0.25, n, sync=False)
    saxpy.synchronize()
    y = yd.to_host()
    tol = dict(rtol=1e-4, atol=1e-4) if vendor == "metal" else {}
    assert np.allclose(y, y0 + 0.25 * x * 8, **tol)
    with pytest.raises(UnsupportedError):
        saxpy.launch(y0.copy(), xd, 1.0, n, sync=False)
    xd.free()
    yd.free()


def test_spirv_barrier_and_workgroup_array_encoded():
    """Structural: the generated SPIR-V contains OpControlBarrier and a
    Workgroup-storage OpVariable."""
    import struct
    from hanajit.backends import spirv
    fn, at, vt, rt_ = _typed(("""
def red(out, n):
    tid = thread_id()
    s = shared_f64(64)
    s[tid] = 1.0
    barrier()
    if tid == 0:
        out[0] = s[0]
    return 0
""", {"out": "f64*", "n": "i64"}))
    for flavor in ("opencl", "vulkan"):
        blob = spirv.generate(fn, at, vt, rt_, flavor)
        assert struct.pack("<I", (4 << 16) | 224) in blob  # OpControlBarrier
        # OpVariable (4 words) with storage class Workgroup(4) trailing
        assert any(
            struct.unpack_from("<I", blob, o)[0] == (4 << 16) | 59
            and struct.unpack_from("<I", blob, o + 12)[0] == 4
            for o in range(0, len(blob) - 12, 4))


def test_unavailable_vendor_has_reason():
    from hanajit.backends import rt
    for vendor in ("cuda", "amd", "intel", "vulkan", "metal"):
        r = rt.get_runtime(vendor)
        if r is None:
            assert rt.why_unavailable(vendor)   # human-readable reason
        else:
            assert rt.why_unavailable(vendor) is None


# ---- SPIR-V generator: structural checks, no hardware needed ----
def _typed(src_args):
    import ast as _ast
    import textwrap
    from hanajit.typeinfer import TypeInferencer
    src, arg_types = src_args
    fn = _ast.parse(textwrap.dedent(src)).body[0]
    var_types, ret = TypeInferencer(fn, arg_types).run()
    return fn, arg_types, var_types, ret


SAXPY_SRC = ("""
def saxpy(y, x, a, n):
    i = block_id() * block_dim() + thread_id()
    if i < n:
        y[i] = a * x[i] + y[i]
    return 0
""", {"y": "f64*", "x": "f64*", "a": "f64", "n": "i64"})


@pytest.mark.parametrize("flavor", ["opencl", "vulkan"])
def test_spirv_module_well_formed(flavor):
    import struct
    from hanajit.backends import spirv
    fn, at, vt, rt_ = _typed(SAXPY_SRC)
    blob = spirv.generate(fn, at, vt, rt_, flavor)
    assert len(blob) % 4 == 0
    magic, version = struct.unpack_from("<II", blob, 0)
    assert magic == 0x07230203
    assert version == (0x00010300 if flavor == "vulkan" else 0x00010000)
    # entry-point name is embedded as a nul-terminated literal
    assert b"saxpy" in blob


def test_spirv_vulkan_local_size_is_baked():
    from hanajit.backends import spirv
    fn, at, vt, rt_ = _typed(SAXPY_SRC)
    a = spirv.generate(fn, at, vt, rt_, "vulkan", local_size=(64, 1, 1))
    b = spirv.generate(fn, at, vt, rt_, "vulkan", local_size=(128, 1, 1))
    assert a != b


def test_spirv_rejects_array_kind_arguments():
    """GPU kernels take raw pointers; numpy array kinds must be refused
    with a clear message, not silently miscompiled."""
    from hanajit.backends import spirv
    from hanajit.errors import UnsupportedError
    fn, at, vt, rt_ = _typed(("""
def k(x, n):
    x[0] = 1.0
    return 0
""", {"x": "f64[1c]", "n": "i64"}))
    with pytest.raises(UnsupportedError):
        spirv.generate(fn, at, vt, rt_, "opencl")
