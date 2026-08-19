# Changelog

## Unreleased

- Added `f.export_executable(output, sig=None, cuda="off", cuda_arch=None)` for standalone
  x86-64 command-line programs on Windows, Linux, and Intel macOS. Scalar
  `i64`/`f64`/`bool` arguments are parsed from the command line and the result
  is printed to stdout; no Python or HanaJit runtime is present in the output.
- `cuda="optional"` embeds PTX and dynamically uses the NVIDIA driver when
  available, with a compiled CPU fallback in the same executable;
  `cuda="required"` fails clearly instead. The destination does not need the
  CUDA toolkit or sidecar PTX.
- Standalone builds auto-detect MSVC (including non-developer Windows shells),
  Clang, or GCC and always write reproducible C source and build scripts.
- Supported scalar `math.*`/`numpy.*` calls lower to native operations without
  bundling those packages. Arbitrary PyPI package calls, imports inside the
  kernel, `eval`, and other dynamic behavior fail export explicitly.
- Fixed Metal's Objective-C autorelease-pool ABI declarations on 64-bit macOS;
  Metal SAXPY launch/copy-back now passes on Apple Silicon GitHub runners.
- Added architecture-aware executable tests: x86-64 integrations skip on
  Apple Silicon while architecture-independent exporter unit tests simulate
  their intended x86-64 precondition. Also fixed the clean-environment launch
  test on Windows Python 3.10.

## 0.23.0

**GPU kernels now execute.** `f.launch(*args, grid=, block=)` runs a
GPU-target kernel on the device: numpy arrays are copied over, the kernel
dispatches, and arrays are copied back. Five runtime bridges, all pure
ctypes over the vendor's driver library — no SDKs, no build steps:

- **CUDA** (`nvcuda`): driver API, PTX loaded via `cuModuleLoadDataEx`
  (the driver JIT keeps the portable `sm_75` default running on newer
  GPUs). *Validated on an RTX 2080 Max-Q.* Transcendental math
  (sin/exp/log/pow) needs libdevice and is not yet linked — such kernels
  refuse to launch; sqrt/floor/ceil/fabs lower natively.
- **Intel** (`ze_loader`, Level Zero): kernels are compiled to OpenCL-
  flavor SPIR-V by a new from-scratch SPIR-V binary generator
  (`backends/spirv.py`) working directly from the typed AST — LLVM's
  SPIR-V backend has no binary writer. Thread intrinsics map to the
  workitem builtin variables. *Validated on UHD Graphics 630.*
- **Vulkan** (`vulkan-1`): same SPIR-V generator, shader flavor —
  storage buffers per pointer argument, scalars in one push-constant
  block, workgroup size baked per (kernel, block). Vendor-neutral: any
  Vulkan 1.1 device with shaderFloat64+shaderInt64. Works around an
  NVIDIA driver bug (64-bit OpSRem evaluates unsigned) by deriving
  remainders from OpSDiv; trig/exp/pow compute at f32 (GLSL.std.450
  defines them for 16/32-bit floats only). *Validated on an RTX 2080.*
- **AMD** (`amdhip64`, HIP): assembles the emitted GCN text into an HSA
  code object with clang (any standard build), then
  `hipModuleLaunchKernel`. Code-complete; awaiting validation on AMD
  hardware.
- **Metal** (macOS): compiles the transpiled MSL at runtime via
  `newLibraryWithSource:` and dispatches through `objc_msgSend`. f64
  arrays round-trip through float32 (Metal has no double). Written to
  the documented ABI; awaiting validation on Apple hardware.

**Kernel-side intrinsics** — enough to write real GPU algorithms:

- `s = shared_f64(N)` / `shared_i64(N)` — workgroup-shared arrays
  (addrspace(3) on cuda/amd, Workgroup storage in generated SPIR-V,
  `threadgroup` in MSL). Size is a compile-time literal.
- `barrier()` — workgroup synchronization (`nvvm.barrier0`,
  `amdgcn.s.barrier`, `OpControlBarrier`, `threadgroup_barrier`).
- `atomic_add(buf, i, v)` — atomic add on f64/i64 buffers, on every
  target: LLVM `atomicrmw` (cuda/amd), `OpAtomicIAdd` + a
  compare-exchange loop over the bit pattern (intel), and
  `OpAtomicIAdd` / `OpAtomicFAddEXT` with the Vulkan 1.2 int64-atomics
  and `VK_EXT_shader_atomic_float` device features detected and enabled
  automatically (a device lacking them gets an error naming the exact
  missing feature). Returns the old value.
- 2-D/3-D kernels: `thread_id_y()`, `block_id_y()`, `block_dim_y()` and
  the `_z` variants, on every target; `grid`/`block` already took up to
  three dimensions.
- The classic two-stage shared-memory dot-product reduction is
  bit-exact against numpy on CUDA, Level Zero, and Vulkan; an atomic
  histogram matches `np.bincount` on CUDA.
- Bare-call statements are now type-checked (previously silently
  dropped): unsupported calls fall back to CPython honestly, and
  `barrier()`/`atomic_add()` in statement position actually emit.

**Async launches** — `f.launch(..., sync=False)` queues the kernel and
returns immediately (CUDA: 20 kernels queued in <1 ms); `f.synchronize()`
or `DeviceArray.to_host()` waits. Requires DeviceArray pointer arguments
(transient numpy arrays need synchronous copy-back and are refused).
Implemented per vendor: CUDA stream, Level Zero async immediate list,
Vulkan deferred-release submissions, HIP, Metal retained command
buffers.

Also in this release:

- **Resident device buffers.** `d = f.to_device(arr)` uploads once;
  `launch()` accepts the DeviceArray in place of the numpy array and
  skips the per-launch copies (measured on saxpy/2M: 16.8 -> 0.20 ms
  CUDA, 21.7 -> 3.0 ms Level Zero, 117 -> 7.0 ms Vulkan). Read back with
  `d.to_host()`; refresh with `d.copy_from_host(arr)`; freed on `free()`
  or garbage collection.
- **CUDA transcendental math via libdevice.** When NVIDIA's
  libdevice.10.bc is present (CUDA toolkit, `CUDA_PATH`,
  `HANAJIT_LIBDEVICE`, or `pip install hanajit[cuda-math]`),
  llvm.sin/cos/exp/log/pow intrinsics are rewritten to `__nv_*`, the
  bitcode is linked and internalized, and kernels run at full f64
  precision (validated: max err 1.5e-11 over a mixed expression on the
  RTX 2080). Without libdevice such kernels still refuse to launch with
  a clear error.
- Device-code emission for ALL GPU vendors now runs in an isolated
  subprocess: llvmlite's backends hard-crash the host process on
  unsupported IR (NVPTX with libdevice-only intrinsics, SPIR-V shader
  GEPs, flaky wasm32 emissions) — a crash now degrades to the
  annotated-IR fallback instead of killing the interpreter.
- `launch()` derives `grid` from the first array argument when omitted
  (`ceil(len/block)`); `block` defaults to 256 threads.
- Runtime availability is introspectable:
  `hanajit.backends.rt.get_runtime(vendor)` /
  `why_unavailable(vendor)`.
- Multi-GPU machines can pick the device per vendor:
  `HANAJIT_CUDA_DEVICE` / `HANAJIT_INTEL_DEVICE` /
  `HANAJIT_VULKAN_DEVICE` / `HANAJIT_AMD_DEVICE` (ordinal; Vulkan
  defaults to the discrete GPU when unset).
- `python -m hanajit.doctor` gained a **launch** section: it runs saxpy
  on every available runtime bridge and reports the device name, or the
  precise reason a vendor cannot launch here.

270 tests passing across Python 3.10-3.14 on Linux / Windows 11 / macOS
Apple Silicon; GPU execution hardware-validated on NVIDIA RTX 2080
Max-Q (CUDA + Vulkan) and Intel UHD Graphics 630 (Level Zero).

## 0.22.0

Adds a **WebAssembly export backend**, a **Vulkan GPU target**, and a
substantially expanded **FPGA export**.

- **WebAssembly** (`f.export_wasm(prefix, sig=None, bits=32)`, plus
  `f.inspect_wasm()`): retargets the kernel IR to `wasm32`/`wasm64` and
  writes the retargeted `.ll`, WebAssembly assembly `.s`, an ES-module
  loader `.mjs` (libm calls map to JS `Math`; `i64` ↔ `BigInt`), and the
  exact clang build command. When any standard clang is on PATH the final
  `.wasm` is linked automatically (`HANAJIT_WASM_CLANG` overrides).
  llvmlite's wasm emitter is crash-prone in-process (access violations on
  alloca-bearing modules and repeated emissions), so assembly emission
  runs in an isolated subprocess and degrades to retargeted IR.
- **Vulkan** (`@jit(target="vulkan")`): emits shader-flavor SPIR-V
  (`spirv-unknown-vulkan1.3-compute`, `OpEntryPoint GLCompute`), with the
  mandatory `hlsl.shader`/`hlsl.numthreads` entry-point attributes.
  `thread_id()`/`block_id()` lower to `llvm.spv.thread.id.in.group` /
  `llvm.spv.group.id`; `block_dim()` folds to the compile-time workgroup
  size (`HANAJIT_VULKAN_LOCAL_SIZE`, default `64,1,1`). LLVM's shader
  backend hard-aborts on IR it cannot select (e.g. buffer indexing under
  logical addressing), so emission is subprocess-isolated: accepted
  kernels yield real SPIR-V (`native=True`), the rest fall back to
  annotated IR. Distinct from the `intel` target's OpenCL-flavor SPIR-V.
- **FPGA** (`f.export_fpga(prefix, sig=None, part=None, clock_ns=None)`):
  now writes a full Vitis HLS project kit — a synthesizable C++ top
  function transpiled from the typed Python AST (`PIPELINE II=1` on
  innermost loops, `m_axi`/`s_axilite` interface pragmas), a C-simulation
  testbench, and a runnable `csim → csynth → export_design` TCL script
  with configurable part and clock, alongside the `.ll`. Kernels outside
  the transpilable subset still export IR + a TCL stub. **Breaking:** the
  return value is now an `FpgaExport` namedtuple `(ll, tcl, cpp, tb)`
  instead of a 2-tuple.
- Exports no longer embed the CPython fastcall wrapper: `export_wasm` /
  `export_fpga` re-derive a pristine kernel-only module, and accept a
  signature string (`sig="f64, i64"`) so a function can be exported
  without being called first.

236 tests passing across Python 3.10-3.14 on Linux / Windows 11 / macOS
Apple Silicon.

## 0.21.0

Adds **narrow-integer compute mode** (`f.narrow(...)`, experimental, opt-in) —
the integer companion to float32 mode. For a memory-bandwidth-bound integer
reduction over a large 1-D `int8` / `int16` / `int32` array, the compiled
kernel loads narrow elements as SIMD vectors and accumulates in a wide `int64`
vector, moving far fewer bytes per element.

- **Exact results, no overflow.** Accumulation is always 64-bit, so the result
  is bit-identical to the `int64` sum. This is the key difference from naive
  narrowing, where an int8 accumulator wraps around.
- **Measured speedups** on a memory-bound sum: `int8` ~2.3-3.2x, `int16`
  ~2.0-2.3x, `int32` ~1.5x over an `int64` baseline. Bandwidth-dependent;
  re-measure on your hardware.
- **Opt-in and scoped.** Gated behind `confirmed=True` (like hyper mode). Unlike
  hyper mode the result is exact — what is "experimental" is the specialized
  codegen path and the narrow-storage requirement. Currently accelerates the
  sum reduction over one narrow array; other patterns fall back to the normal
  compiler with a warning.
- **`int4` / `int2` are intentionally not supported on CPU** — there are no
  sub-byte SIMD load instructions, so they require bit-unpacking whose cost eats
  the bandwidth saving. They belong on the accelerator roadmap.

217 tests passing across Python 3.10-3.14 on Linux / Windows 11 / macOS Apple
Silicon.

## 0.20.2

Documentation, website, and repository release. No functional code changes;
the compiler and its behavior are identical to 0.20.1, and all 208 tests
pass across Python 3.10–3.14 on Linux / Windows 11 / macOS Apple Silicon.

- Windows CI: `examples/demo.py` now writes FPGA output to a temporary
  directory instead of a hardcoded `/tmp` path, so the demo smoke test runs
  on Windows runners.
- Documentation accuracy pass: corrected the test count (208), the
  `reduce_reassoc` reduction figure (~1.5x over the default), the float32
  reduction figure (~2.7x over the float64 baseline), and the default CUDA
  arch reference (sm_75) to match the code. Added an explicit note that GPU
  backends emit and assemble vendor-valid code but do not yet launch
  kernels.
- Landing page and GitHub Pages: added a project landing page under `docs/`
  for Pages, five architecture diagrams to the README, and a project logo.

## 0.20.1

Patch: fixed a stale test that assumed float32 arrays were unsupported.
float32 became a supported dtype in 0.20.0, so the fallback test now uses
float16 (still unsupported) to verify the transparent-interpreter-fallback
path. No functional code change; CI is green.

## 0.20.0 — stable

The CPU-performance and GPU-validation release. 207 tests passing across
Python 3.10–3.14 on Linux / Windows 11 / macOS Apple Silicon.

### Performance (measured)
- **float32 arrays** (new): native 32-bit compute — half the memory
  bandwidth, 2x SIMD lanes. With `reduce_reassoc`, **3.24x** the float64
  baseline on a memory-bound reduction, at exact (bounded) f32 precision.
- **`reduce_reassoc=True`** (new): reassociation only on reduction
  accumulators → numpy-class reduction speed (**2.48x** on f64) without
  global fastmath; integers stay bit-exact.
- **Fusion**: 3.18x vs numpy, 3.85x vs numba on a 5-op fused reduction
  (allocation-free, structural).
- **GA autotune** `evolve()`: 2.13x, equivalence-guaranteed.
- **Dispatch**: 36 ns/call, 3.54x faster than numba.
- **Helper inlining** + **auto-parallel** (`parallel=True`): Taichi-style
  ergonomics, no DSL — cross-call overhead ~1.05x of hand-inlined.

### GPU codegen — validated against real vendor assemblers
- CUDA PTX assembles to cubin on real NVIDIA `ptxas` (sm_75..sm_121,
  arch-adaptive, empirically probed).
- AMD GCN assembles to object on real LLVM AMDGPU `llvm-mc` (gfx90a).
- Fixed a day-one bug: GPU kernels now emit `void` (were `i64` → invalid
  PTX). Fixed AMD `block_dim` (dispatch-packet read) and code-object
  version (v5 default, configurable).
- Metal MSL compiles via `xcrun metal` on M4.
- **Note:** emission/assembly is verified; host-side *launch*
  (`cuLaunchKernel`) is roadmap.

### Experimental (opt-in, CPU-only, clearly warned)
- `@jit(rewrite=True)`: pattern-matched structural rewrites (closed-form
  reductions), each proven equivalent.
- `evolve_hyper(..., confirmed=True)`: hyper-aggressive fp transforms,
  validated on random probes only, never cached, requires confirmation.
  Documented as workload-dependent (often a no-op vs safe GA).

### Fixed
- **Latent F32 arithmetic bug**: all float32 math was computing through
  integer conversion (`0.5*0.5 → 0`). Surfaced by float32 support, fixed in
  type resolution + op selection, locked down with 9 tests.
- Windows temp-file UTF-8 encoding; CUDA arch selection across toolkit
  versions; reduction stack-spilling (SSA phi form).
