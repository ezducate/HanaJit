# Limitations & safety notes

## Semantic deviations from CPython (compiled code)

- Integers are i64: **wraparound on overflow**, no bigint promotion.
- **Division by zero traps** (SIGFPE) instead of raising
  `ZeroDivisionError`.
- `and`/`or` evaluate both operands (no short-circuit).
- `x ** negative_int` with integer base is not Python-equivalent — use
  floats.
- Metal only: f64→float32; integer `/` `%` use C semantics on the GPU.

Integer `//` and `%` DO follow Python floor semantics (negative operands
included), verified by randomized differential tests.

## Unsupported constructs (→ interpreter fallback)

Containers, strings, classes, closures over mutable state, generators,
exceptions, `import` inside kernels, whole-array numpy calls (`np.sum(x)` etc.), calls
to anything except `abs`/`int`/`float`/scalar `math.*`/`np.*` math
functions/self-recursion/GPU intrinsics, chained comparisons,
multiple-assignment targets, `*args`/`**kwargs`/defaults, lambdas and any
function whose source `inspect` can't retrieve (REPL).

## Safety

Standalone executable export is currently limited to scalar `i64`/`f64`/`bool`
signatures on x86-64 Windows, Linux, and Intel macOS. Pointer/array command-line
interfaces are not yet defined. CUDA standalone mode runs one scalar invocation,
does not support recursion or GPU thread intrinsics, and is unavailable on macOS.
The output needs no Python or CUDA toolkit; optional/required CUDA execution does
require an installed NVIDIA driver.

The exporter does not freeze arbitrary Python dependencies. Supported scalar
`math.*`/`numpy.*` calls are lowered to native operations, but imports inside
the function, PyTorch/pandas/custom package calls, `eval`, dynamic imports, and
other runtime-dependent behavior are rejected with `UnsupportedError`.

- **Distinct array/pointer arguments are assumed non-overlapping**
  (`noalias`, same contract as numba). Passing overlapping views as two
  separate arguments of a kernel that writes through one of them is
  undefined behavior.
- **Pointer signatures are C-level unsafe**: raw addresses, no bounds or
  lifetime checking. Passing a wrong address or freeing the buffer while a
  kernel runs is undefined behavior. Keep the numpy array alive for the
  duration of the call.
- The native dispatcher relies on documented-but-internal CPython details
  (vectorcall ABI, `PyObject` layout, `METH_FASTCALL`). Guards restrict it
  to CPython ≥3.12, 64-bit, non-free-threaded builds; anywhere else it
  silently degrades to the Python dispatcher. Free-threaded (no-GIL) builds
  are dispatcher-fallback only for now.
- JIT compilation happens at runtime: never `@jit` source you don't trust —
  it's still arbitrary code execution, same as running it.

## Platform status

Linux x86_64, Windows x86-64, and Apple Silicon macOS are in the CI matrix;
arm64 codegen is additionally cross-verified from Linux. GPU execution
(`f.launch()`) is hardware-verified for CUDA (RTX 2080 Max-Q), Intel Level Zero
(UHD 630), Vulkan (RTX 2080), and Metal (Apple Silicon GitHub runner). The AMD
HIP bridge is code-complete but still awaiting AMD hardware. GPU execution is
explicit: calling a GPU-target function directly still falls back to CPython.
Alpha software: pin your version.
