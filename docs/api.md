# API Reference

## `hanajit.jit(func=None, *, ...)`

Decorator returning a JIT-dispatching callable.

| Parameter | Default | Meaning |
|---|---|---|
| `target` | `"cpu"` | `"cpu"`, `"cuda"`, `"amd"`, `"intel"`, `"metal"`, or `"auto"`. GPU targets emit device code (see gpu.md); `"auto"` resolves per kernel via hardware detection. |
| `fallback` | `True` | On `UnsupportedError`, run the original Python function (with one warning). `False` raises instead. |
| `cache` | `False` | Persist compiled machine code to disk; later processes warm-start (see performance.md). |
| `nogil` | `False` | Release the GIL around the native kernel call. Kernels are pure native code, so this is always safe; it enables true multi-core threading. |
| `fastmath` | `False` | Emit `fast` flags on FP ops (reassociation, FMA). Results may differ in the last bits. |
| `signature` | `None` | Explicit types, e.g. `"f64*, f64*, f64, i64"`. Enables pointer arguments and triggers eager compilation (required for GPU targets, optional for CPU). Tokens: `i64`, `f64`, `bool`, `f64*`, `i64*`. |
| `native_dispatch` | `True` | Use the native vectorcall dispatcher on CPython ≥3.12 (see architecture.md). `False` forces the pure-Python dispatcher. |
| `verbose` | `False` | Print compilation/cache events. |

## Methods on a jitted function

| Method | Returns |
|---|---|
| `f.specialize(*types)` | Raw native entry point for those Python types (e.g. `f.specialize(int, int)`), bypassing dispatch entirely. No type checking beyond CPython unboxing — use where types are stable. |
| `f.inspect_llvm(sig=None)` | LLVM IR text of a compiled specialization. Unavailable for disk-cache hits (codegen was skipped). |
| `f.inspect_asm(sig=None)` | Host-native assembly. |
| `f.inspect_gpu()` | `(vendor, device_code, native)` after a GPU-target compile. `native=True` means real PTX/GCN/SPIR-V; `False` means annotated IR or MSL source. |
| `f.export_fpga(prefix)` | Writes `<prefix>.ll` + a Vitis HLS TCL stub; returns the paths. |
| `f.cache` / `f._fast` / `f.modules` | Introspection dicts: abstract-signature → callable, Python-type-tuple → callable, signature → IR module. |

## NumPy arrays as arguments

1-D C-contiguous `float64`/`int64` numpy arrays pass directly — no
`.ctypes.data`, and no `signature=` needed in lazy mode (the dtype drives
inference). Kernels index with `x[i]`; pass lengths explicitly. Wrong dtype,
strided views, or ndim>1 fall back to the interpreter (where numpy works
anyway), so nothing breaks. Array specializations are dtype-keyed (f64 and
i64 arrays never collide) and bypass the native vectorcall dispatcher —
array calls pay ~1 µs of Python dispatch, negligible for array-sized work.

## `f.scipy_callable(nargs=1)`

Compiles a float64 specialization and wraps its raw kernel address as a
`scipy.LowLevelCallable`, which `scipy.integrate.quad` and friends invoke
as a C function pointer — zero Python per evaluation. Requires a
float-returning kernel of `nargs` float parameters; requires scipy.

## float32 arrays (native, 2x on memory-bound work)

Pass a `float32` numpy array and hanajit compiles the kernel with native
32-bit LLVM float operations — **half the memory bandwidth and twice the
SIMD lane count** of float64. No flag required; the dtype drives it. Combined
with `reduce_reassoc=True`, a memory-bound reduction runs **2.87x** the
float64 baseline (measured, 20M elements).

Semantics are **exact float32**, not an approximation — results match
numpy's own float32 computation. The precision is well-defined float32
(~7 significant digits, ~1e-7 relative), a *bounded* tradeoff, not undefined
behavior. Use it where float32 precision suffices (ML inference, graphics,
large reductions where you don't need 15 digits). Mixed f32/f64 in one
kernel promotes to f64 following numpy's rules.

## `@jit(reduce_reassoc=True)`

Applies the `reassoc` fast-math flag **only to reduction accumulators**
(float `+=`/`-=` loops and `np.sum`/`np.dot`/`np.mean`), letting LLVM
vectorize the reduction with parallel SIMD accumulators — the same
reassociation numpy's pairwise summation performs. Result: **numpy-class
reduction throughput** (measured 1.58x on a 20M sum, reaching numpy's
GB/s; 1.19x on dot, matching BLAS) without enabling global fastmath.

Unlike `fastmath=True`, this enables *only* reassociation — not
no-NaN/no-Inf/no-signed-zero assumptions — so it is safe on normal data.
Integer reductions are untouched (bit-exact). The result is not
bit-identical to the sequential sum (reassociation reorders additions, by
design), but stays within the same tolerance as numpy (~1e-10 relative).
Best for memory-bound reductions; compute-bound fused kernels see no
benefit (they vectorize elsewhere already). Off by default.

## `@jit(parallel=True)`

Auto-parallelizes the outermost `for i in range(n)` / `range(lo, hi)` loop
by promoting it to `prange` and running it chunked across a GIL-released
thread pool. Same eligibility as `prange` (one top-level loop, at most one
`acc +=` reduction, array writes to disjoint indices). Not applicable →
compiles serially. `workers=N` sets the pool size. Float reductions are
reassociated across chunks (≈1e-12 vs serial, like `numba.prange`); integer
results are exact.

## Helper inlining (automatic)

Any `@jit` function whose body is straight-line (assignments + one
`return`) is registered and **inlined** into other `@jit` functions that
call it, at the AST level, before type inference. Cross-function calls
become in-line arithmetic — no dispatch cost, and the fusion engine can
see across the former call boundary. Recursion and helpers containing
loops/branches are never inlined (they dispatch, still correct). No API:
just decorate both functions with `@jit`.

## `f.evolve(*example_args, generations=6, population=10, reps=5, seed=0, allow_fastmath=False, verbose=False)`

Genetic search over semantics-preserving compilation strategies for the
matching specialization. Installs the winner into live dispatch only if
measurably faster; returns `{baseline_ms, best_ms, speedup, genome,
installed}`. `allow_fastmath=True` adds an FP-reassociation gene (results
validated to 1e-9 relative). Mutating kernels are evaluated on array
copies. Not available for disk-cache-hit specializations (no IR retained).

## Module-level

- `hanajit.pmap(fn, argtuples, workers=None)` — thread-pool map; with
  `nogil=True` kernels this is real multi-core parallelism.
- `hanajit.detect()` — ordered `[(target, evidence), ...]` for this machine;
  always ends with `("cpu", "always available")`.
- `hanajit.UnsupportedError` — raised (or swallowed by fallback) when code
  leaves the compilable subset.

## Environment variables

- `HANAJIT_CACHE_DIR` — disk-cache location (default `~/.cache/hanajit`).
- `HANAJIT_TARGET` — force detection results (e.g. `HANAJIT_TARGET=cpu`).

## Compilable subset (v0.x)

Scalars `int`/`float`/`bool` (as i64/f64/i1); `f64*`/`i64*` pointers via
`signature=` with `x[i]` loads/stores; operators `+ - * / // % ** << >> & |
^`, comparisons, `and/or/not` (non-short-circuit), ternary; `if/elif/else`,
`while`, `for i in range(...)`, `break`/`continue`; self-recursion;
`abs`/`int`/`float`; GPU intrinsics `thread_id()`/`block_id()`/`block_dim()`.
Integer `//` and `%` follow Python floor semantics including negative
operands. Everything else falls back — see limitations.md.
