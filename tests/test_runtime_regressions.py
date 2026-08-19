import ctypes
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hanajit import UnsupportedError
from hanajit.backends import detect, gpu, isolated, rt, wasm
from hanajit.backends.rt import metal as metal_runtime
from hanajit.typeinfer import I64


class FakeRuntime:
    def __init__(self, fail_write=False, fail_free=False):
        self.fail_write = fail_write
        self.fail_free = fail_free
        self.calls = []
        self.storage = {}
        self.next_impl = 100

    def sync(self):
        self.calls.append(("sync",))

    def buf_alloc(self, arr):
        impl = self.next_impl
        self.next_impl += 1
        self.calls.append(("alloc", impl, arr.nbytes))
        self.storage[impl] = np.empty_like(arr)
        return impl

    def buf_write(self, impl, arr):
        self.calls.append(("write", impl))
        if self.fail_write:
            raise RuntimeError("upload failed")
        self.storage[impl][...] = arr

    def buf_read(self, impl, out):
        self.calls.append(("read", impl))
        out[...] = self.storage[impl]

    def buf_free(self, impl):
        self.calls.append(("free", impl))
        if self.fail_free:
            raise RuntimeError("driver already gone")
        self.storage.pop(impl, None)


class FakeCFunction:
    def __init__(self, result=1):
        self.result = result
        self.restype = "unset"
        self.argtypes = "unset"

    def __call__(self, *args):
        return self.result


def test_metal_autorelease_pool_uses_pointer_sized_ctypes_signatures(
        monkeypatch):
    objc = SimpleNamespace(
        sel_registerName=FakeCFunction(),
        objc_msgSend=FakeCFunction(),
        objc_autoreleasePoolPush=FakeCFunction(),
        objc_autoreleasePoolPop=FakeCFunction(),
    )
    metal = SimpleNamespace(MTLCreateSystemDefaultDevice=FakeCFunction())
    core_foundation = SimpleNamespace(
        CFStringCreateWithCString=FakeCFunction())

    def load_library(path):
        if path.endswith("libobjc.A.dylib"):
            return objc
        if path.endswith("/Metal"):
            return metal
        return core_foundation

    def message(_self, _receiver, selector, _restype, _argtypes, *args):
        if selector == b"UTF8String":
            return b"Test Apple GPU"
        return 1

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(metal_runtime.ctypes, "CDLL", load_library)
    monkeypatch.setattr(metal_runtime.Runtime, "_msg", message)

    runtime = metal_runtime.Runtime()

    assert runtime.device_name == "Test Apple GPU"
    assert objc.objc_autoreleasePoolPush.restype is ctypes.c_void_p
    assert objc.objc_autoreleasePoolPush.argtypes == []
    assert objc.objc_autoreleasePoolPop.restype is None
    assert objc.objc_autoreleasePoolPop.argtypes == [ctypes.c_void_p]


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (None, 64, (64, 1, 1)),
        (8, 64, (8, 1, 1)),
        ((8,), 64, (8, 1, 1)),
        ((8, 4), 64, (8, 4, 1)),
        ((8, 4, 2), 64, (8, 4, 2)),
        ([3, 2, 1], 64, (3, 2, 1)),
    ],
)
def test_launch_dimension_normalization(value, default, expected):
    assert rt.normalize_dims(value, default) == expected


@pytest.mark.parametrize(
    "value", [0, -1, (1, 0), (1, 2, -3), (1, 2, 3, 4)]
)
def test_invalid_launch_dimensions_are_rejected(value):
    with pytest.raises(UnsupportedError, match="bad launch dimensions"):
        rt.normalize_dims(value, 1)


@pytest.mark.parametrize(
    "text, expected", [("0", 0), ("3", 3), ("-2", 0), (" 4 ", 4),
                       ("1.5", 0), ("", 0), ("junk", 0)]
)
def test_device_index_environment_parsing(text, expected, monkeypatch):
    monkeypatch.setenv("HANAJIT_CUDA_DEVICE", text)
    assert rt.device_index("cuda") == expected


def test_to_device_upload_readback_and_explicit_free():
    runtime = FakeRuntime()
    source = np.arange(8, dtype=np.float64)
    device = rt.to_device(runtime, source)

    assert len(device) == 8
    assert device.shape == (8,)
    assert device.dtype == "float64"
    assert np.array_equal(device.to_host(), source)
    assert runtime.calls[:4] == [
        ("alloc", 100, source.nbytes), ("write", 100),
        ("sync",), ("read", 100),
    ]

    replacement = source * 3
    assert device.copy_from_host(replacement) is device
    out = np.empty_like(source)
    assert device.to_host(out) is out
    assert np.array_equal(out, replacement)

    device.free()
    device.free()  # idempotent
    assert runtime.calls.count(("free", 100)) == 1
    with pytest.raises(UnsupportedError, match="already freed"):
        device.to_host()


def test_device_array_rejects_dtype_size_and_shape_mismatches():
    runtime = FakeRuntime()
    source = np.arange(8, dtype=np.float64)
    device = rt.to_device(runtime, source)
    try:
        with pytest.raises(UnsupportedError, match="shape/dtype mismatch"):
            device.copy_from_host(source.astype(np.int64))
        with pytest.raises(UnsupportedError, match="shape/dtype mismatch"):
            device.copy_from_host(np.arange(4, dtype=np.float64))
        with pytest.raises(UnsupportedError, match="shape/dtype mismatch"):
            device.copy_from_host(np.zeros((2, 4), dtype=np.float64))
        with pytest.raises(UnsupportedError, match="shape/dtype mismatch"):
            device.to_host(np.zeros((2, 4), dtype=np.float64))
    finally:
        device.free()


def test_to_device_frees_allocation_when_upload_fails():
    runtime = FakeRuntime(fail_write=True)
    source = np.arange(4, dtype=np.int64)
    with pytest.raises(RuntimeError, match="upload failed"):
        rt.to_device(runtime, source)
    assert ("free", 100) in runtime.calls


def test_device_array_free_swallows_driver_teardown_error():
    runtime = FakeRuntime(fail_free=True)
    device = rt.to_device(runtime, np.arange(3, dtype=np.int64))
    device.free()  # interpreter/driver teardown must remain safe
    assert device._impl is None


def test_unknown_runtime_is_cached_with_human_readable_reason():
    rt.reset()
    assert rt.get_runtime("unknown") is None
    assert rt.get_runtime("unknown") is None
    assert rt.why_unavailable("unknown") == \
        "no runtime bridge for target 'unknown'"
    rt.reset()
    assert "unknown" not in rt._cache


def test_runtime_constructor_success_is_cached(monkeypatch):
    instance = object()
    calls = []

    class RuntimeFactory:
        def __new__(cls):
            calls.append("constructed")
            return instance

    monkeypatch.setattr(importlib, "import_module",
                        lambda *a, **k: SimpleNamespace(Runtime=RuntimeFactory))
    rt.reset()
    try:
        assert rt.get_runtime("cuda") is instance
        assert rt.get_runtime("cuda") is instance
        assert calls == ["constructed"]
        assert rt.why_unavailable("cuda") is None
    finally:
        rt.reset()


@pytest.mark.parametrize(
    "error, expected",
    [(rt.RuntimeUnavailable("no device"), "no device"),
     (RuntimeError("broken driver"), "RuntimeError: broken driver")],
)
def test_runtime_constructor_failures_are_contained(
        error, expected, monkeypatch):
    class RuntimeFactory:
        def __new__(cls):
            raise error

    monkeypatch.setattr(importlib, "import_module",
                        lambda *a, **k: SimpleNamespace(Runtime=RuntimeFactory))
    rt.reset()
    try:
        assert rt.get_runtime("cuda") is None
        assert rt.why_unavailable("cuda") == expected
    finally:
        rt.reset()


@pytest.mark.parametrize(
    "sys_platform, expected",
    [("win32", "win32"), ("cygwin", "linux"), ("linux", "linux"),
     ("darwin", "darwin")],
)
def test_detection_platform_mapping(sys_platform, expected, monkeypatch):
    monkeypatch.setattr(detect.sys, "platform", sys_platform)
    assert detect._platform() == expected


def test_driver_probe_tries_candidates_until_one_loads(monkeypatch):
    seen = []

    def cdll(name):
        seen.append(name)
        if name != "good-driver":
            raise OSError("missing")
        return object()

    monkeypatch.setattr(detect.ctypes, "CDLL", cdll)
    assert detect._loadable(["bad-one", "bad-two", "good-driver"]) == \
        "good-driver"
    assert seen == ["bad-one", "bad-two", "good-driver"]


def test_detection_preserves_platform_preference_order(monkeypatch):
    available = {
        "libcuda.so.1": True,
        "libamdhip64.so": True,
        "libze_loader.so.1": True,
    }
    monkeypatch.delenv("HANAJIT_TARGET", raising=False)
    monkeypatch.setattr(detect, "_platform", lambda: "linux")
    monkeypatch.setattr(
        detect, "_loadable",
        lambda names: next((n for n in names if available.get(n)), None),
    )
    detect.detect.cache_clear()
    try:
        found = detect.detect()
        assert [target for target, _ in found] == [
            "cuda", "amd", "intel", "cpu",
        ]
        assert detect.best_gpu() == "cuda"
    finally:
        detect.detect.cache_clear()


def test_macos_detection_always_reports_metal_then_cpu(monkeypatch):
    monkeypatch.delenv("HANAJIT_TARGET", raising=False)
    monkeypatch.setattr(detect, "_platform", lambda: "darwin")
    detect.detect.cache_clear()
    try:
        assert detect.detect() == [
            ("metal", "macOS (Metal is always present)"),
            ("cpu", "always available"),
        ]
        assert detect.best_gpu() == "metal"
    finally:
        detect.detect.cache_clear()


def test_forced_detection_target_skips_driver_probes(monkeypatch):
    monkeypatch.setenv("HANAJIT_TARGET", "vulkan")
    monkeypatch.setattr(
        detect, "_loadable",
        lambda names: pytest.fail("driver probes must be bypassed"),
    )
    detect.detect.cache_clear()
    try:
        assert detect.detect() == [
            ("vulkan", "forced via HANAJIT_TARGET"),
        ]
    finally:
        detect.detect.cache_clear()


def test_isolated_emitter_retries_then_returns_output(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"bad")
        return SimpleNamespace(returncode=0, stdout=b"assembly\n", stderr=b"")

    monkeypatch.setattr(isolated.subprocess, "run", run)
    got = isolated.emit_assembly(
        "define i64 @k() { ret i64 0 }", "fake-triple", "fake-arch",
        link_bitcode="lib.bc", kernel="k")
    assert got == "assembly\n"
    assert len(calls) == 2
    argv, kwargs = calls[0]
    assert argv[-4:] == ["fake-triple", "fake-arch", "lib.bc", "k"]
    assert kwargs["input"].startswith(b"define i64")
    assert kwargs["timeout"] == 120


@pytest.mark.parametrize("behavior", ["empty", "exception"])
def test_isolated_emitter_contains_persistent_failures(behavior,
                                                       monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        if behavior == "exception":
            raise OSError("cannot spawn")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(isolated.subprocess, "run", run)
    assert isolated.emit_assembly("ir", "triple") is None
    assert len(calls) == 2


def test_libdevice_environment_override_has_priority(tmp_path, monkeypatch):
    libdevice = tmp_path / "libdevice.10.bc"
    libdevice.write_bytes(b"bitcode")
    monkeypatch.setenv("HANAJIT_LIBDEVICE", str(libdevice))
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "other"))
    assert gpu.find_libdevice() == str(libdevice)


@pytest.mark.parametrize(
    "vendor, env_name, default, override",
    [("cuda", "HANAJIT_CUDA_ARCH", "sm_75", "sm_90"),
     ("amd", "HANAJIT_AMD_ARCH", "gfx90a", "gfx1100"),
     ("intel", "HANAJIT_INTEL_ARCH", "", "pvc")],
)
def test_gpu_architecture_resolution_precedence(
        vendor, env_name, default, override, monkeypatch):
    monkeypatch.delenv(env_name, raising=False)
    assert gpu.resolve_arch(vendor) == default
    monkeypatch.setenv(env_name, override)
    assert gpu.resolve_arch(vendor) == override
    assert gpu.resolve_arch(vendor, "explicit") == "explicit"


@pytest.mark.parametrize(
    "bits, triple", [(32, "wasm32-unknown-unknown"),
                     (64, "wasm64-unknown-unknown")],
)
def test_wasm_link_flags_export_kernel_and_memory(bits, triple):
    flags = wasm._link_flags("kernel", bits)
    assert f"--target={triple}" in flags
    assert "-Wl,--no-entry" in flags
    assert "-Wl,--export=kernel" in flags
    assert "-Wl,--export-memory" in flags


def test_wasm_loader_escapes_contract_into_javascript():
    loader = wasm._loader(
        "calculate", "calculate.wasm", 64,
        sig=("i64", "f64"), ret="f64")
    assert "signature: calculate(i64, f64) -> f64" in loader
    assert 'new URL("./calculate.wasm"' in loader
    assert "Math.sin" in loader and "Math.pow" in loader
    assert "instance.exports" in loader


def test_wasm_link_failure_keeps_reproducible_artifacts(tmp_path,
                                                        monkeypatch):
    from llvmlite import ir

    module = ir.Module(name="wasm-test")
    monkeypatch.setattr(wasm, "emit", lambda *a, **k: ("fallback", False))
    monkeypatch.setattr(wasm.shutil, "which", lambda _: "fake-clang")
    monkeypatch.setattr(
        wasm.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("link failed")),
    )
    prefix = str(tmp_path / "nested" / "program")
    (tmp_path / "nested").mkdir()
    out = wasm.export_wasm(module, "kernel", prefix, sig=(I64,), ret=I64)
    assert out.wasm is None
    assert Path(out.ll).is_file()
    assert Path(out.mjs).is_file()
    assert Path(out.build).is_file()
    assert out.s is None


def test_doctor_check_wrapper_records_pass_skip_and_failure(monkeypatch):
    from hanajit import doctor

    monkeypatch.setattr(doctor, "CHECKS", [])
    monkeypatch.setattr(doctor, "RESULTS", [])
    monkeypatch.setattr(doctor, "FAILURES", [])

    @doctor.check("unit", "passes")
    def passes():
        return "details"

    @doctor.check("unit", "skips")
    def skips():
        raise doctor.SkipCheck("not available")

    @doctor.check("unit", "fails")
    def fails():
        raise RuntimeError("boom")

    for check in doctor.CHECKS:
        check()
    assert [row[2] for row in doctor.RESULTS] == ["PASS", "SKIP", "FAIL"]
    assert doctor.RESULTS[0][3] == "details"
    assert doctor.RESULTS[1][3] == "not available"
    assert doctor.FAILURES[0][0] == "fails"
    assert "RuntimeError: boom" in doctor.FAILURES[0][1]


def test_doctor_subprocess_reports_success_and_failure():
    from hanajit import doctor

    ok = doctor.subproc("print('hello')", timeout=20)
    assert ok == (0, "hello", "")
    failed = doctor.subproc("raise RuntimeError('child boom')", timeout=20)
    assert failed[0] != 0
    assert "child boom" in failed[2]
