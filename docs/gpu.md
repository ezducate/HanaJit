# GPU targets

All GPU targets are **device-code emission** today: they produce inspectable,
offline-compilable kernels. Host-side launch bridges (device memory, copies,
grid launch) are the top roadmap item; until they land, calling a GPU-target
function on the host falls back to CPython.

## Writing a GPU kernel

```python
from hanajit import jit

@jit(target="cuda", signature="f64*, f64*, f64, i64")
def saxpy(x, y, a, n):
    i = block_id() * block_dim() + thread_id()
    if i < n:
        y[i] = a * x[i] + y[i]
    return 0

vendor, code, native = saxpy.inspect_gpu()
```

`signature=` is required (GPU kernels are never type-inferred from a host
call). Intrinsics: `thread_id()`, `block_id()`, `block_dim()`.

## Per-vendor specifics

| target | Output | Notes |
|---|---|---|
| `cuda` | PTX (LLVM NVPTX) | All three intrinsics (`%tid.x`, `%ctaid.x`, `%ntid.x`). Assemble with `ptxas` to sanity-check on a CUDA machine. |
| `amd` | GCN ISA / HSA code object v6 (LLVM AMDGPU, default `gfx90a`) | `thread_id`/`block_id` only — pass the workgroup size as a kernel argument (reading the dispatch packet is roadmap). IR is optimized before emission (AMDGPU rejects generic-addrspace allocas). |
| `intel` | SPIR-V (LLVM SPIR-V backend) | `spir_kernel` calling convention; runtime would be Level Zero / OpenCL. |
| `metal` | Metal Shading Language **source** | LLVM has no Metal target, so this is an exact source transpiler over the compile subset. **f64 lowers to float32** (Metal has no double); GPU integer `/` and `%` keep C semantics. Same three intrinsics, mapped to threadgroup attributes. Validate on a Mac: `python examples/metal_check.py` (compiles with real `xcrun metal` to `.metallib`). |

## `target="auto"`

Kernels using thread intrinsics resolve to the best detected GPU
(cuda > amd > intel on Linux/Windows; metal on macOS); detection probes
driver libraries (`libcuda`, `libamdhip64`, `ze_loader`) without initializing
devices. With no GPU present, decoration raises immediately — a thread-
indexed kernel can't run on the CPU, and a late `NameError` would be worse.

## FPGA

`f.export_fpga(prefix)` writes optimized LLVM IR plus a Vitis HLS TCL stub.
FPGA flows must go through HLS (Vitis HLS's LLVM front end) or CIRCT; there
is no direct LLVM→bitstream path.

## Toolchain & GPU compatibility

Emitted PTX defaults to **`sm_75`** (Turing). This is the portable choice:
every CUDA toolkit from 11.0 through 13.x assembles it, and PTX is
**forward-compatible** — the driver re-JITs device code for a newer GPU at
load time, so `sm_75` PTX also runs on Ampere, Ada, Hopper and beyond. It
does *not* run on pre-Turing cards (Pascal/Volta); for those, or to tune
for a specific newer architecture, override it:

```python
@jit(target="cuda", gpu_arch="sm_90", signature="f64*, i64")   # Hopper
def k(x, n): ...
```

or set an environment variable (also `HANAJIT_AMD_ARCH`,
`HANAJIT_INTEL_ARCH`):

```
HANAJIT_CUDA_ARCH=sm_60      # Pascal
HANAJIT_CUDA_ARCH=sm_90      # Hopper
```

Resolution order is explicit `gpu_arch=` > env var > portable default. The
`doctor` reads the arch from the emitted PTX and assembles at that arch if
your installed `ptxas` supports it, otherwise the closest supported arch
(newest toolkits drop the oldest targets) — and reports exactly which arch
it used.


Note: the doctor empirically probes ptxas (re-emitting per arch) rather than parsing help text, so it is robust to toolkits like CUDA 13 that support only sm_90+.


## AMD GCN specifics

AMD kernels support `thread_id`, `block_id`, and `block_dim`. Unlike NVIDIA
(where all three are single special-register reads), AMDGPU has no direct
workgroup-size register — `block_dim` is lowered to a read from the HSA
dispatch packet (`llvm.amdgcn.dispatch.ptr`, offset 4, i16), verified
against the real LLVM AMDGPU assembler (`llvm-mc`, gfx90a).

Emitted GCN uses **HSA code-object version 5** by default, which ROCm 5.x
through current all accept. Toolchains older than LLVM 19 reject v6;
newer setups can opt into it. Override with:

```
HANAJIT_AMD_CODE_OBJECT_VERSION=6   # newest ROCm/LLVM
HANAJIT_AMD_CODE_OBJECT_VERSION=4   # older ROCm
```

The `doctor` assembles our GCN with `llvm-mc` when it is on PATH (it ships
with LLVM/clang and ROCm), producing a real object file — the AMD analogue
of the `ptxas` check for NVIDIA.

**Intel note:** SPIR-V *emission* is verified, but Intel thread-index
intrinsics (`thread_id` etc.) are not yet mapped to SPIR-V builtins, so
data-parallel Intel kernels currently fall back. This is the least
complete of the four GPU targets and is on the roadmap.
