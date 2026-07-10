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

Linux x86_64 is the fully verified platform. macOS (Apple Silicon) and
Windows are in the CI matrix; arm64 codegen is additionally cross-verified
from Linux. GPU backends are emission-verified only — no hardware execution
has been performed. Alpha software: pin your version.
