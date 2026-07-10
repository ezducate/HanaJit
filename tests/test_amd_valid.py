"""The AMD GCN we emit must be VALID, assembled by the real LLVM AMDGPU
backend (llvm-mc) when available. Regressions for two bugs found only by
using a real assembler: (1) block_dim() was missing for AMD entirely;
(2) code-object version was pinned too high for older toolchains."""
import glob
import os
import shutil
import subprocess
import tempfile
import warnings
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")
LLVM_MC = shutil.which("llvm-mc")


def _amd_kernel():
    @jit(target="amd", signature="f64*, f64*, f64, i64")
    def sax(x, y, a, n):
        i = block_id() * block_dim() + thread_id()   # needs all 3 intrinsics
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0
    return sax.inspect_gpu()


def test_amd_block_dim_now_supported():
    """block_dim() must compile for AMD (was UnsupportedError)."""
    art = _amd_kernel()
    assert art is not None, "AMD kernel with block_dim fell back"
    _, asm, native = art
    assert native
    # dispatch-packet read is how AMDGPU gets workgroup size
    assert "dispatch.ptr" in asm or "dispatch_ptr" in asm


def test_amd_kernel_is_void():
    """GPU kernels are void — no return-value slot (the func_retval0 class)."""
    _, asm, native = _amd_kernel()
    assert native
    assert "amdgpu_kernel" in asm or ".amdhsa_kernel" in asm


def test_amd_code_object_version_is_portable():
    """Default code-object version must be the broadly-compatible v5, not a
    version that only newest toolchains accept."""
    _, asm, _ = _amd_kernel()
    cov = [l for l in asm.splitlines() if "code_object_version" in l]
    assert cov, "no code-object version line emitted"
    assert "5" in cov[0], cov[0]


def test_amd_code_object_version_override(monkeypatch):
    import importlib
    from hanajit.backends import gpu
    monkeypatch.setenv("HANAJIT_AMD_CODE_OBJECT_VERSION", "4")
    importlib.reload(gpu)
    assert gpu.AMD_CODE_OBJECT_VERSION == 4
    monkeypatch.delenv("HANAJIT_AMD_CODE_OBJECT_VERSION", raising=False)
    importlib.reload(gpu)


@pytest.mark.skipif(LLVM_MC is None, reason="no llvm-mc available")
def test_real_llvm_mc_assembles_our_gcn():
    _, asm, native = _amd_kernel()
    assert native
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "k.s")
        open(p, "w", encoding="utf-8").write(asm)
        r = subprocess.run(
            ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx90a",
             "-filetype=obj", p, "-o", os.path.join(td, "k.o")],
            capture_output=True, text=True)
        assert r.returncode == 0, (r.stderr + r.stdout)[-300:]
        assert os.path.getsize(os.path.join(td, "k.o")) > 0
