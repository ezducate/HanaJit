# Performance guide

## Measured results

See `benchmarks/RESULTS.md` (5-section suite) and
`benchmarks/RESULTS-MICRO.md` (52 verified workloads). Headline geomeans on
the reference machine: **55x over CPython, 1.12x over numba** across 52
workloads; dispatch is ~3.5–4x faster than numba's and cheaper than a plain
Python function call; warm process start is ~8.5x faster than numba's.
Regenerate on your machine: `python benchmarks/bench.py` and
`python benchmarks/microbench.py`.

## Disk cache (`cache=True`)

Object code is stored under `$HANAJIT_CACHE_DIR` (default
`~/.cache/hanajit`), keyed by source, signature, host CPU, flags, and
llvmlite/hanajit/Python versions — edits, upgrades, or a different CPU
recompile automatically. Writes are atomic (temp+rename), so many workers
share one cache safely: with gunicorn/uvicorn, the first worker on a machine
compiles and every later worker warm-starts. Measured first-call cost:
43 ms → 16 ms (the rest is LLVM engine setup). Caching is best-effort — a
read-only filesystem or corrupt entry silently recompiles. Trade-off:
cache hits skip codegen, so `inspect_llvm()` is unavailable for them.

## Threading (`nogil=True`, `pmap`)

Kernels are pure native code, so `nogil=True` is always safe and lets
Python threads execute kernels concurrently across cores.
`hanajit.pmap(fn, argtuples, workers=N)` is the one-liner. In web servers
this multiplies per-worker throughput for compute-bound endpoints (FastAPI
sync handlers already run in a thread pool). Overhead is ~two C calls per
invocation — skip it for sub-microsecond kernels.

## Call-overhead ladder (fastest first)

1. `f.specialize(int, int)` → raw fastcall builtin (~60 ns/call measured)
2. `f(...)` with native dispatch (~70 ns)
3. `f(...)` on the Python dispatcher fallback (~450 ns)
4. numba's dispatcher for comparison (~245 ns)

## Profiling

Under cProfile, native dispatch shows **no Python frames per call** — if you
see `decorator.py` in a profile, that function is on the fallback path
(look for the one-time warning, or check `type(f).__name__`). For kernel
internals use `f.inspect_asm()` and `perf`; results memoization composes
with `functools.lru_cache()(jit(f))`.

## Known performance gaps vs numba (by design, for now)

- Cross-function jit calls aren't compiled (each crossing pays dispatch);
  numba inlines them. Visible as the deliberately-included 0.14x row in the
  microbenchmarks.
- No value-range analysis: numba proves loop indices non-negative and emits
  unsigned division; we emit signed + Python-semantics adjustment
  (mitigated by the `% == 0` peephole; worst observed case 0.80x).
