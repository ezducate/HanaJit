"""Vulkan target: SPIR-V shader-flavor emission. Native emission runs in
an isolated subprocess (the SPIR-V backend hard-aborts on IR it cannot
select), so every kernel must yield either real GLCompute SPIR-V or the
annotated-IR fallback — never a dead interpreter."""
import warnings

import pytest

warnings.filterwarnings("ignore")


def test_vulkan_target_config():
    from hanajit.backends import gpu
    cfg = gpu.TARGETS["vulkan"]
    assert cfg["triple"] == "spirv-unknown-vulkan1.3-compute"
    assert cfg["callconv"] is None  # entry is annotated, not callconv'd


def test_pointer_kernel_emits_or_falls_back_annotated():
    from hanajit import jit

    @jit(target="vulkan", signature="f64*, i64")
    def scale(x, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            x[i] = x[i] * 2.0
        return 0

    vendor, text, native = scale.inspect_gpu()
    assert vendor == "vulkan"
    if native:
        assert "OpEntryPoint GLCompute" in text
    else:
        # annotated IR: mandatory shader attributes must be present for
        # an offline llc/llvm-spirv to accept the entry point
        assert '"hlsl.shader"="compute"' in text
        assert '"hlsl.numthreads"="64,1,1"' in text
        assert "llvm.spv.thread.id.in.group" in text
        assert "llvm.spv.group.id" in text


def test_scalar_kernel_native_spirv_when_backend_allows():
    from hanajit import jit

    @jit(target="vulkan", signature="i64")
    def trivial(n):
        return n + 1

    vendor, text, native = trivial.inspect_gpu()
    assert vendor == "vulkan"
    if native:
        assert "OpEntryPoint GLCompute" in text
        assert "OpExecutionMode" in text and "LocalSize" in text
    else:
        assert '"hlsl.shader"="compute"' in text


def test_block_dim_is_compile_time_local_size(monkeypatch):
    monkeypatch.setenv("HANAJIT_VULKAN_LOCAL_SIZE", "128,1,1")
    from hanajit import jit

    @jit(target="vulkan", signature="f64*, i64")
    def k(x, n):
        i = thread_id() + block_id() * block_dim()
        if i < n:
            x[i] = 1.0
        return 0

    _, text, native = k.inspect_gpu()
    if native:
        assert "LocalSize 128 1 1" in text
    else:
        assert '"hlsl.numthreads"="128,1,1"' in text
        # block_dim() lowered to the constant 128, not an intrinsic read
        assert "mul i64" in text and "128" in text


def test_local_size_env_validation(monkeypatch):
    from hanajit.backends import gpu
    monkeypatch.delenv("HANAJIT_VULKAN_LOCAL_SIZE", raising=False)
    assert gpu.vulkan_local_size() == "64,1,1"
    monkeypatch.setenv("HANAJIT_VULKAN_LOCAL_SIZE", "128, 2, 1")
    assert gpu.vulkan_local_size() == "128,2,1"
    monkeypatch.setenv("HANAJIT_VULKAN_LOCAL_SIZE", "0,1,1")
    with pytest.raises(ValueError):
        gpu.vulkan_local_size()
    monkeypatch.setenv("HANAJIT_VULKAN_LOCAL_SIZE", "64,1")
    with pytest.raises(ValueError):
        gpu.vulkan_local_size()


def test_vulkan_and_intel_spirv_flavors_differ():
    """OpenCL-flavor (intel) and shader-flavor (vulkan) SPIR-V are not
    interchangeable; the configs must stay distinct."""
    from hanajit.backends import gpu
    assert (gpu.TARGETS["vulkan"]["triple"]
            != gpu.TARGETS["intel"]["triple"])
    assert gpu.TARGETS["intel"]["callconv"] == "spir_kernel"
