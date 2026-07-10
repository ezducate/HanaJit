# Changelog

## 0.20.0 — stable

The CPU-performance and GPU-validation release. 207 tests passing across
Python 3.10–3.14 on Linux / Windows 11 / macOS Apple Silicon.

### Performance (all measured, all honest)
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
  Honestly documented as workload-dependent (often a no-op vs safe GA).

### Fixed
- **Latent F32 arithmetic bug**: all float32 math was computing through
  integer conversion (`0.5*0.5 → 0`). Surfaced by float32 support, fixed in
  type resolution + op selection, locked down with 9 tests.
- Windows temp-file UTF-8 encoding; CUDA arch selection across toolkit
  versions; reduction stack-spilling (SSA phi form).
