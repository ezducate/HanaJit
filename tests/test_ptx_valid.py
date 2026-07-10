"""The PTX we emit must be VALID, not just present. If a real NVIDIA ptxas
is available (CUDA toolkit or the nvidia-cuda-nvcc pip wheel), assemble
our kernels with it. Regression for the func_retval0 bug: GPU kernels are
void — a kernel that returns a value produces PTX ptxas rejects."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import pytest
from hanajit import jit

warnings.filterwarnings("ignore")


def _find_ptxas():
    p = shutil.which("ptxas")
    if p:
        return p
    for pat in (os.path.join(os.path.dirname(os.__file__), "..", "..", "**",
                             "cuda_nvcc", "bin", "ptxas"),
                "/usr/local/lib/python*/dist-packages/nvidia/cuda_nvcc/bin/ptxas",
                os.path.expanduser("~/.local/lib/python*/site-packages/"
                                   "nvidia/cuda_nvcc/bin/ptxas")):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


PTXAS = _find_ptxas()


def _kernel(arch):
    @jit(target="cuda", signature="f64*, f64*, f64, i64", gpu_arch=arch)
    def sax(x, y, a, n):
        i = block_id() * block_dim() + thread_id()
        if i < n:
            y[i] = a * x[i] + y[i]
        return 0
    return sax.inspect_gpu()


def test_gpu_kernel_is_void_in_ptx():
    """Structural regression (runs everywhere, no toolkit needed): the PTX
    entry must not reference a return value slot."""
    _, ptx, native = _kernel(None)
    assert native
    assert ".entry" in ptx or ".visible .entry" in ptx
    assert "func_retval0" not in ptx      # the exact bug ptxas rejected


@pytest.mark.skipif(PTXAS is None, reason="no ptxas available")
@pytest.mark.parametrize("arch", ["sm_75", "sm_90"])
def test_real_ptxas_assembles_our_ptx(arch):
    _, ptx, native = _kernel(arch)
    assert native and f".target {arch}" in ptx
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "k.ptx")
        open(p, "w").write(ptx)
        r = subprocess.run([PTXAS, "-arch=" + arch, p, "-o",
                            os.path.join(td, "k.cubin")],
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.stderr + r.stdout)[-300:]
        assert os.path.getsize(os.path.join(td, "k.cubin")) > 0
