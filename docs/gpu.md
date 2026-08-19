# GPU targets

GPU-target kernels both **emit** inspectable device code and **execute**
on the device via `f.launch()`. Calling the function directly (`f(...)`)
still falls back to CPython — device execution is always explicit.

## Executing kernels: `f.launch()`

```python
import numpy as np
from hanajit import jit

@jit(target="cuda", signature="f64*, f64*, f64, i64")   # or intel/vulkan/amd/metal
def saxpy(y, x, a, n):
    i = block_id() * block_dim() + thread_id()
    if i < n:
        y[i] = a * x[i] + y[i]
    return 0

y = np.random.rand(1_000_000); x = np.random.rand(1_000_000)
saxpy.launch(y, x, 2.0, len(y))          # grid derived from len(y)
saxpy.launch(y, x, 2.0, len(y), grid=4096, block=128)
```

Arrays (1-D contiguous f64/i64) are copied to the device, the kernel runs
over `grid x block` threads, and arrays are copied back — in-place writes
are visible on the host afterwards. `block` defaults to 256; `grid`
defaults to `ceil(len(first_array)/block)`.

Each vendor bridge is pure ctypes over the driver's own library — no SDK
or build step:

| target | runtime library | device code | validated on |
|---|---|---|---|
| `cuda` | `nvcuda` (driver API) | emitted PTX, driver-JITed | RTX 2080 Max-Q |
| `intel` | `ze_loader` (Level Zero) | OpenCL-flavor SPIR-V from hanajit's own generator | UHD Graphics 630 |
| `vulkan` | `vulkan-1` | shader-flavor SPIR-V from the same generator | RTX 2080 Max-Q |
| `amd` | `amdhip64` (HIP) | GCN text assembled by clang into an HSA code object | code-complete, awaiting AMD hardware |
| `metal` | Metal.framework (macOS) | transpiled MSL compiled at runtime | Apple Silicon GitHub runner |

### Async launches

`f.launch(..., sync=False)` queues the kernel and returns immediately;
`f.synchronize()` (or `DeviceArray.to_host()`, which synchronizes first)
waits for completion. Async launches require DeviceArray pointer
arguments — transient numpy arrays imply a synchronous copy-back and are
refused with a clear error.

```python
for step in range(1000):
    integrate.launch(state_d, forces_d, dt, n, sync=False)
integrate.synchronize()
result = state_d.to_host()
```

### Resident device buffers

Every plain launch copies all arrays both ways. To keep data on the GPU
across launches:

```python
xd = saxpy.to_device(x)          # upload once
yd = saxpy.to_device(y)
for _ in range(100):
    saxpy.launch(yd, xd, 2.0, n) # no copies: ~0.2 ms vs ~17 ms (CUDA, 2M)
result = yd.to_host()            # explicit readback
xd.free(); yd.free()             # or let GC handle it
```

Execution caveats (honest list):

- **cuda**: sin/cos/exp/log/pow are linked from NVIDIA's libdevice when
  one is found (CUDA toolkit, `CUDA_PATH`, `HANAJIT_LIBDEVICE`, or
  `pip install hanajit[cuda-math]`) and then run at full f64 precision;
  without libdevice, kernels using them refuse to launch with a clear
  error. sqrt, fabs, floor, ceil always lower to native PTX.
- **vulkan**: trig/exp/pow evaluate at float32 (GLSL.std.450 defines them
  for 16/32-bit floats only); everything else is f64. Requires a Vulkan
  1.1 device exposing `shaderFloat64` and `shaderInt64`. Integer
  remainders avoid `OpSRem` entirely — NVIDIA's driver evaluates 64-bit
  `OpSRem` as an *unsigned* remainder — so Python floor-division/modulo
  semantics hold on every driver.
- **metal**: all f64 computes at float32 (Metal has no double); f64
  arrays round-trip through a float32 copy.
- **amd**: needs any clang on PATH (`HANAJIT_HIP_CLANG` overrides) to
  assemble the code object.
- Plain NumPy-array launches copy arrays both ways. `DeviceArray` arguments
  keep their storage resident and avoid those transfers until `.to_host()`.

`hanajit.backends.rt.get_runtime(vendor)` returns the bridge or None;
`why_unavailable(vendor)` explains a None (no driver, no device, missing
tool).

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
call). Intrinsics: `thread_id()`, `block_id()`, `block_dim()`, plus:

- `s = shared_f64(N)` / `shared_i64(N)` — workgroup-shared array
  (N is a compile-time literal). Index it like any pointer: `s[tid]`.
- `barrier()` — synchronize the workgroup (call it from uniform control
  flow, i.e. outside thread-dependent branches).
- `atomic_add(buf, i, v)` — atomically add `v` to `buf[i]`, returning
  the old value. f64/i64 on every target: `atomicrmw` (cuda/amd), a
  compare-exchange loop (intel), native atomics with the required
  device features auto-enabled (vulkan — a device without Vulkan 1.2
  int64 atomics / `VK_EXT_shader_atomic_float` gets an error naming
  the missing feature). Not available on metal (MSL lacks 64-bit
  atomics).
- 2-D/3-D indexing: `thread_id_y()`/`block_id_y()`/`block_dim_y()` and
  `_z` variants; pass tuples to `grid=`/`block=`.

Device selection on multi-GPU machines: `HANAJIT_CUDA_DEVICE=1` (also
`HANAJIT_INTEL_DEVICE` / `HANAJIT_VULKAN_DEVICE` / `HANAJIT_AMD_DEVICE`)
picks the device ordinal; Vulkan prefers the discrete GPU when unset.
`python -m hanajit.doctor` includes a **launch** section that runs a
real kernel on every available bridge and names the device it used.

The portable reduction pattern (no atomics needed) — each workgroup
tree-reduces in shared memory, partial sums are combined on the host:

```python
@jit(target="cuda", signature="f64*, f64*, f64*, i64")
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
```

## Per-vendor specifics

| target | Output | Notes |
|---|---|---|
| `cuda` | PTX (LLVM NVPTX) | 1-D/2-D/3-D thread, block, and block-size intrinsics; shared memory, barriers, and atomics. The driver JIT loads embedded PTX at launch. |
| `amd` | GCN ISA / HSA code object v5 (LLVM AMDGPU, default `gfx90a`) | Thread/block/block-size intrinsics, shared memory, barriers, and atomics; HIP launches a clang-assembled HSA object. IR is optimized before emission. |
| `intel` | SPIR-V, OpenCL flavor (HanaJit's binary generator) | Thread indexing, shared memory, barriers, and atomics dispatched by the Level Zero bridge; validated on UHD 630. |
| `vulkan` | SPIR-V, shader flavor (`GLCompute`) | Entry point annotated `hlsl.shader="compute"` + `hlsl.numthreads`. `thread_id`/`block_id` map to `llvm.spv.thread.id.in.group` / `llvm.spv.group.id`; `block_dim()` folds to the compile-time workgroup size (`HANAJIT_VULKAN_LOCAL_SIZE`, default `64,1,1`). Emission runs in an isolated subprocess (LLVM's shader backend hard-aborts on IR it cannot select, e.g. buffer indexing under logical addressing); kernels it rejects fall back to annotated IR with `native=False`. Not interchangeable with the `intel` target's OpenCL-flavor SPIR-V. |
| `metal` | Metal Shading Language **source** | LLVM has no Metal target, so this is an exact source transpiler over the compile subset. **f64 lowers to float32** (Metal has no double); GPU integer `/` and `%` keep C semantics. Thread indexing, shared memory, and barriers are supported; atomics are not. Runtime compilation, launch, and copy-back are validated on Apple Silicon. |

## `target="auto"`

Kernels using thread intrinsics resolve to the best detected GPU
(cuda > amd > intel on Linux/Windows; metal on macOS); detection probes
driver libraries (`libcuda`, `libamdhip64`, `ze_loader`) without initializing
devices. With no GPU present, decoration raises immediately — a thread-
indexed kernel can't run on the CPU, and a late `NameError` would be worse.

## FPGA

`f.export_fpga(prefix, part=None, clock_ns=None)` writes a Vitis HLS
project kit: LLVM IR, a **synthesizable C++ top function** transpiled from
the typed Python AST (interface + pipeline pragmas included), a C-simulation
testbench, and a runnable TCL script (`csim → csynth → export_design`).
Kernels outside the transpilable subset still get IR + a TCL stub. FPGA
flows must go through HLS (Vitis HLS synthesizes the generated C++
directly; its LLVM front end ingests the IR) or CIRCT; there is no direct
LLVM→bitstream path.

## WebAssembly

`f.export_wasm(prefix, sig=None, bits=32)` retargets the kernel IR to
`wasm32`/`wasm64` and writes: the retargeted `.ll`, WebAssembly assembly
`.s`, an ES-module loader `.mjs` (libm → JS `Math` imports; `i64` ↔
`BigInt` at the boundary), and the clang build script. When clang is on
PATH (any standard build — no Emscripten), the final `.wasm` is linked
automatically; `HANAJIT_WASM_CLANG` overrides which clang is used.
`f.inspect_wasm()` returns the assembly text without writing files.

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

**Intel note:** HanaJit's OpenCL-flavor SPIR-V generator maps thread indices
to work-item builtins directly, and the Level Zero bridge has been validated
with data-parallel kernels on UHD 630. The `vulkan` target uses shader-flavor
SPIR-V and a different runtime contract; the two formats are not
interchangeable.
