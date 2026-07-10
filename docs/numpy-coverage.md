# NumPy coverage

Every entry below is verified by the test suite or the probe script; nothing
is aspirational. Unsupported constructs never break — they fall back to the
interpreter with a one-time warning.

## Compiles (natively)

**Arrays as arguments** — `float64`/`int64`, **1-D and 2-D**, C-order,
Fortran-order, and strided (views like `a[::2]` pass straight in); dtype/
layout-keyed specializations that never collide.

**Indexing & views** — `x[i]` with **negative indices**; 2-D `m[i, j]`;
contiguous slices `x[a:b]` (open-ended forms too); **strided slices**
`x[::k]`, `x[a::k]`; nested slices; slice views bound to locals and written
through; `.shape[k]`, `len(x)`.

**Reshape family** — `x.reshape(r, c)` and `.ravel()` as views on
contiguous data (no copy).

**Reductions** — `np.sum` (1-D and 2-D), `np.dot`, `np.min`, `np.max`,
`np.mean`, compiled to fused loops.

**Scalar math** — `np.sqrt/exp/log/sin/cos/floor/ceil/pow/fabs` and their
`math.*` twins, lowered to LLVM intrinsics (CPU + Metal).

## Falls back (interpreter — correct, not accelerated)

Whole-array arithmetic (`a * b`, `a + 1`); matmul `@`; 2-D block slices
`m[a:b, c:d]`; mixed int+slice 2-D indexing `m[1, :]`; **any allocation**
(`np.zeros/empty/arange/copy` inside kernels — needs a refcounted memory
runtime, numba's NRT equivalent); boolean/fancy indexing; broadcasting;
dtypes beyond f64/i64; ndim ≥ 3; everything else in numpy's API.

## Coverage summary

Roughly **20 numpy API surface points** compile (arrays, ~8 indexing/view
forms, 5 reductions, 9 math functions, shape/len/reshape/ravel) out of a
public API numba covers in the **hundreds of functions**, built over ~a
decade with a dedicated memory runtime. hanajit's coverage is chosen to
maximize the loop-kernel sweet spot per line of compiler code; the two
walls before the next tier are elementwise whole-array expressions and
the allocator.
