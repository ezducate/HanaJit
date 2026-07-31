# Changelog

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
