# Testing & benchmarking hanajit

## Test suite (46 tests)

```bash
pip install -e .[test]
pytest -q                      # full suite
pytest tests/test_numerics.py  # differential correctness only
pytest -q -k "gpu"             # GPU emission tests only
pytest -q -k "cache or nogil"  # caching + threading
```

Layout:
- `tests/test_basic.py` — compilation, dispatch modes, multi-signature
  specialization, fallback behavior, `.specialize()`, disk cache round-trip
  and key isolation, GIL release, pointer kernels, GPU thread indexing,
  `lru_cache` composability.
- `tests/test_numerics.py` — **differential suite**: every kernel runs both
  JIT-compiled and as pure Python on randomized inputs (seeded), and results
  must match exactly. Each test also asserts compilation actually happened,
  so a silent fallback fails rather than comparing Python with Python.
  This suite caught real bugs: Python floor-division/modulo semantics for
  negative operands, and boolean arithmetic wrapping at 1 bit.

## Version matrix (what CI runs)

```bash
pip install uv
for V in 3.10 3.11 3.12 3.13; do
  uv venv --python $V /tmp/fj$V
  uv pip install --python /tmp/fj$V/bin/python -e .[test]
  /tmp/fj$V/bin/python -m pytest -q
done
```

3.12+ exercises the native vectorcall dispatcher; 3.10/3.11 exercise the
pure-Python dispatcher fallback (verified by
`test_dispatch_mode_matches_platform`).

## Benchmarks

```bash
pip install numba          # optional comparison baseline
python benchmarks/bench.py           # 5-section suite (~2 min), RESULTS.md
python benchmarks/bench.py --quick   # ~15 s smoke run
python benchmarks/microbench.py      # 52 workloads (~3 min), RESULTS-MICRO.md
python benchmarks/microbench.py --quick
```

`microbench.py` covers 7 categories (integer/float arithmetic, control flow,
recursion, bit manipulation, pointer/array kernels, dispatch patterns), each
workload verified against CPython and numba before timing, with per-category
and overall geometric means. One row is an intentionally-included known gap
(cross-function jit calls, where numba wins ~7x).

Covers: compute workloads vs CPython and numba, dispatch overhead per call
path, time-to-first-result across fresh processes (cold/warm disk cache,
both frameworks), GIL-release measurement, and a cProfile breakdown of the
dispatch layers. Results land in `benchmarks/RESULTS.md`; the committed copy
is from the machine described in its header.

## Profiling your own kernels

```bash
python -m cProfile -s cumulative your_script.py   # Python-side time
```

A `@jit` function with native dispatch shows **zero** Python frames per
call under cProfile — if you see `decorator.py:__call__` in your profile,
you're on the fallback path (check for a one-time fallback warning, or
inspect `type(fn).__name__`: `HanaFunction` = native, `Dispatcher` =
Python). For kernel internals, use `fn.inspect_llvm()` / `fn.inspect_asm()`
and `perf record`/`perf annotate` — JITed frames appear as anonymous
addresses unless you enable a perf map.

## `hanajit doctor` — the platform report

On each machine, run:

```bash
pip install hanajit[test] numba scipy   # extras improve coverage
python -m hanajit.doctor
```

It probes every tier (dispatch, numpy, threads, prange scaling, GA, cache,
scipy, all GPU emissions, real toolchains like ptxas/xcrun when present)
with crash isolation, and writes `hanajit_report_<os>_<arch>.md` including
environment details and full failure tracebacks — send that file back for
debugging.

## Restricted-network / corporate-proxy installs

`pip install -e .` uses *build isolation*: pip tries to download setuptools
from your configured index before building. Behind a corporate mirror with
broken/expired credentials (symptom: `401 Error` warnings, then
`No matching distribution found for setuptools`), use any of these —
hanajit is pure Python, nothing is compiled at install time:

```bash
# 1. reuse the setuptools you already have (check: python -c "import setuptools")
pip install -e . --no-build-isolation --no-deps

# 2. zero-install: run straight from the repo root (pytest adds CWD to sys.path)
python -m pytest tests/ -q
python -m hanajit.doctor

# 3. if outbound PyPI is permitted on that machine
pip install -e ".[test]" --index-url https://pypi.org/simple
```

If `import setuptools` fails on Python 3.12+ (no longer bundled), grab it
once via your working index or `python -m ensurepip`, then use option 1.
Dependencies (llvmlite/numpy) you already have via numba are sufficient —
hanajit accepts llvmlite >= 0.42 and handles both pass-manager APIs.

## Per-platform testing

- **Linux (CPU only)**: everything above runs as-is; this is the fully
  verified platform.
- **Windows + NVIDIA**: `pytest -q` runs the same suite (CI covers it, but
  the ctypes data-symbol lookups against python3xx.dll are the likeliest
  first-run surprise — please report). `target="cuda"` PTX can be assembled
  with `ptxas` from the CUDA toolkit as an extra check.
- **macOS Apple Silicon**: `pytest -q` (CI's macos runners are M-series;
  `test_cross_compile_apple_silicon` also verifies arm64 codegen from any
  host). For the GPU path: `python examples/metal_check.py` compiles a
  generated kernel with Apple's real Metal compiler (needs Xcode CLT).
- **AMD GPUs (no hardware)**: GCN emission is covered by
  `pytest -k gpu_thread_indexing_amd`. Actual execution requires ROCm on
  AMD hardware — the realistic options are a cloud MI300X/MI250 instance
  (AMD Developer Cloud, or DigitalOcean's AMD GPU droplets) for a one-off
  validation, or leaving the backend marked emission-only until a
  contributor with hardware confirms it.

## Accuracy contract

Correctness comparisons are the core of the suite, at three strictness
levels: (1) **bit-exact `==`** against a pure-Python oracle wherever we
execute the same IEEE ops in the same order — all integer kernels, all
non-fastmath float loops, and `np.sum` vs a sequential Python loop;
(2) **1e-9..1e-12 relative** only where the oracle itself computes in a
different order (numpy's pairwise summation) or fastmath was explicitly
opted into; (3) **type identity**: return types must satisfy `type(x) is`
the CPython type (int stays int, `/` promotes to float, comparisons and
`np.any/all` return bool, `np.argmax/count_nonzero` return int). Input
domains covered: bools, numpy scalars, ±inf, NaN (poisons min/max exactly
like numpy), -0.0 sign preservation, and out-of-i64 Python ints — which
raise OverflowError loudly rather than ever computing a wrong value.
Auto-parallel (`parallel=True`) and helper inlining are both tested for
equivalence with the serial / non-inlined oracle before any speed claim:
inlining is verified bit-identical and confirmed to remove the helper call
from the IR; auto-parallel matches serial exactly for integer reductions
and to ~1e-10 for reassociated float reductions. Deliberate deviations are
pinned by their own tests below so they can
never drift silently.

## Known semantic deviations from CPython

- Integers are `i64`: they wrap on overflow instead of promoting to bigint.
- Division by zero traps (SIGFPE) instead of raising `ZeroDivisionError`.
- `x ** negative_int` on integers is not Python-equivalent (use floats).
- `and`/`or` do not short-circuit (both operands are evaluated).

Integer `//` and `%` follow Python floor semantics, including negative
operands (tested differentially).
