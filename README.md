<div align="center">

# HanaJit

**An LLVM-backed JIT compiler for Python — compile ordinary functions and NumPy code to native machine code, with a transparent interpreter fallback and no DSL to learn.**

[![CI](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml/badge.svg)](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#project-status)

*ها أنا — "here I am" (Darija) + JIT*

</div>

---

> **Project status: alpha.** HanaJit is under active development. The CPU
> compiler is stable and tested (207 tests across Python 3.10–3.14 on Linux,
> Windows, and macOS). GPU support is **code generation only** — hanajit
> emits GPU assembly that real vendor toolchains accept, but does not yet
> *launch* kernels on a GPU (see [Scope & honest limitations](#scope--honest-limitations)).
> APIs may change before 1.0.

## What is HanaJit?

HanaJit takes a normal Python function, compiles it to optimized native
machine code through [LLVM](https://llvm.org/) (via
[llvmlite](https://llvmlite.readthedocs.io/)), and runs that instead of the
interpreter — often **10–100× faster than CPython**, and competitive with or
faster than [Numba](https://numba.pydata.org/) on the workloads it targets.

You do not learn a new language, annotate types, or restructure your data.
You add a decorator:

```python
from hanajit import jit

@jit
def sum_squares(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i] * x[i]
    return acc
```

The first call with a given argument type compiles a specialization; later
calls reuse it. Anything HanaJit can't compile falls back to the normal
Python interpreter **transparently**, with a single warning — so adopting it
never breaks working code.

### Why it exists

HanaJit was built in the R&D pipeline of [EZducate](https://ezducate.ai), an
AI-powered special-education platform, to accelerate the numeric and
array-heavy code (on-device inference, simulation, data processing) that sits
between "too slow in pure Python" and "not worth rewriting in C." It is
designed around three principles:

1. **No DSL.** It compiles the Python you already wrote, parsed by CPython's
   own `ast` module — not a restricted dialect, not a new syntax.
2. **Correctness is never negotiable.** Every optimization is either provably
   equivalent, or a *bounded, opt-in* trade-off (like float32 precision)
   documented with its exact cost. Anything HanaJit can't compile runs in the
   interpreter rather than compiling something wrong.
3. **Honesty about performance.** Every number in this README is measured and
   reproducible from the scripts in [`benchmarks/`](benchmarks/). Where a
   feature ties or loses, we say so.

---

## Highlights

- **Drop-in `@jit`** on ordinary functions — loops, recursion, math, NumPy.
- **Lazy fusion engine** — whole-array NumPy expressions compile to a single
  allocation-free loop (no temporaries), beating NumPy *and* Numba
  structurally.
- **`reduce_reassoc=True`** — numpy-class reduction throughput by vectorizing
  accumulators, without global fast-math; integers stay bit-exact.
- **Native float32** — pass a `float32` array and get 32-bit compute: half
  the memory bandwidth, 2× the SIMD lanes, exact (bounded) f32 precision.
- **Genetic optimizer** (`f.evolve()`) — an equivalence-preserving search over
  compilation strategies that tunes each kernel to your machine.
- **Near-zero dispatch** — ~20–50 ns/call via an LLVM-compiled vectorcall path
  on CPython 3.12+.
- **Multithreading** — `prange`, `pmap`, `nogil`, and `parallel=True`
  auto-parallelization.
- **GPU code generation** — emits CUDA PTX, AMD GCN, Intel SPIR-V, and Apple
  Metal, each validated against the real vendor assembler (emission only; see
  scope below).
- **Transparent fallback** — unsupported code runs in CPython automatically.

---

## Performance

Measured on a single core (shared CI container — **ratios are reliable,
absolute milliseconds are noisy**; rerun on your hardware with the scripts in
[`benchmarks/`](benchmarks/)). Compared against NumPy 2.x and Numba 0.66.

| Benchmark | Result |
|---|---|
| 5-operation fused NumPy reduction | **3.2× vs NumPy, 3.9× vs Numba** |
| 20M-element reduction (`reduce_reassoc`) | **2.5×** over the float64 baseline |
| 20M-element reduction (float32 + `reduce_reassoc`) | **3.2×** over the float64 baseline |
| Genetic optimizer (`evolve`) on an fp reduction | **2.1×**, equivalence-guaranteed |
| Dispatch / call overhead | **~36 ns** (3.5× faster than Numba) |
| `fib(30)` recursion | **1.85× vs Numba** |

**How to read this honestly:** on bare scalar loops, HanaJit and Numba are at
parity — they share the LLVM backend, so loop codegen is a wash. HanaJit's
wins come from (a) the fusion engine, which is a structural advantage on
array expressions, (b) targeted reduction vectorization, (c) native float32,
(d) dispatch latency and cold-start. Where the underlying operation is already
at the hardware roofline, there is no magic to extract, and we don't pretend
otherwise.

---

## Installation

HanaJit requires **Python 3.10+** and depends only on `llvmlite` (which ships
prebuilt LLVM wheels for all major platforms — you do **not** need to install
LLVM yourself).

### From PyPI

> Not yet published. Once released, this will be:
>
> ```bash
> pip install hanajit
> ```
>
> See [`docs/publishing.md`](docs/publishing.md) for the release process.

### From GitHub (available now)

Install the latest version directly from this repository:

```bash
pip install "git+https://github.com/ezducate/HanaJit.git"
```

Pin to a specific tag or commit for reproducibility:

```bash
# a released tag
pip install "git+https://github.com/ezducate/HanaJit.git@v0.20.0"

# a specific commit
pip install "git+https://github.com/ezducate/HanaJit.git@<commit-sha>"
```

Add it to `requirements.txt`:

```
hanajit @ git+https://github.com/ezducate/HanaJit.git@v0.20.0
```

Or to `pyproject.toml` dependencies:

```toml
dependencies = [
    "hanajit @ git+https://github.com/ezducate/HanaJit.git@v0.20.0",
]
```

### Optional extras

```bash
# run the test suite
pip install "hanajit[test] @ git+https://github.com/ezducate/HanaJit.git"

# run the benchmarks (adds numba + scipy for comparison)
pip install "hanajit[bench] @ git+https://github.com/ezducate/HanaJit.git"
```

### From a local clone (for development)

```bash
git clone https://github.com/ezducate/HanaJit.git
cd HanaJit
pip install -e ".[test]"     # editable install with test deps
python -m pytest tests/ -q   # run the suite
python -m hanajit.doctor     # environment + capability diagnostic
```

---

## Quick start

### Accelerate a numeric function

```python
from hanajit import jit
import numpy as np

@jit
def euclidean_norm(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i] * x[i]
    return acc ** 0.5

a = np.random.rand(1_000_000)
euclidean_norm(a)   # compiles on first call, runs native thereafter
```

### Fuse a NumPy expression (no temporaries)

```python
@jit
def score(a, b):
    # compiles to ONE loop — no intermediate arrays are ever allocated
    return np.sum(np.exp(-a * a) * b + np.where(a > 0, a, 2 * a))
```

### Numpy-class reductions

```python
@jit(reduce_reassoc=True)     # vectorizes the accumulator; integers stay exact
def total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]
    return acc
```

### Native float32 (2× on memory-bound work)

```python
total(a.astype(np.float32))   # 32-bit compute, exact f32 precision
```

### Tune a kernel to your machine

```python
f = jit(heavy_kernel)
f(example_args)                       # compile
report = f.evolve(example_args)       # genetic search, keeps the fastest
                                      # equivalence-verified variant
```

### Parallelize

```python
from hanajit import jit, prange

@jit(parallel=True)          # outermost range loop auto-parallelizes
def process(x, out):
    for i in range(len(x)):
        out[i] = expensive(x[i])
    return 0
```

See [`docs/`](docs/) for the full API, and [`examples/`](examples/) for
runnable programs.

---

## How it works

HanaJit is ~3,000 lines of readable Python. One IR, many machines:

1. **Frontend** — `inspect.getsource` + `ast.parse` give the exact tree
   CPython would execute. No custom parser.
2. **Type inference** — a fixpoint over a small lattice (`i64`, `f64`, `f32`,
   `bool`, pointers, array kinds). Anything outside it raises
   `UnsupportedError` → transparent interpreter fallback.
3. **Code generation** — the typed AST lowers to LLVM IR (via `llvmlite.ir`),
   including the fusion engine that turns array expressions into element
   generators fused into one loop.
4. **Backends** — the same IR module is optimized (`-O3`) and either JITed for
   your exact host CPU, or re-targeted for GPUs (PTX / GCN / SPIR-V / MSL) or
   exported for FPGA HLS.

For a deeper tour, see [`docs/architecture.md`](docs/architecture.md).

---

## Scope & honest limitations

HanaJit is deliberately clear about what it does and does not do.

**What works and is tested:**
- CPU compilation of a useful subset of Python + NumPy, with transparent
  fallback for the rest.
- The fusion engine, reductions, float32, the genetic optimizer, inlining,
  auto-parallelization, and multithreading.
- **207 tests** passing across Python 3.10–3.14 on Linux, Windows 11, and
  macOS (Apple Silicon).

**GPU: code generation is verified; kernel launch is not implemented.**
HanaJit emits GPU assembly and this output is validated by the *real* vendor
toolchains — NVIDIA `ptxas` assembles our PTX to a cubin, LLVM's AMDGPU
`llvm-mc` assembles our GCN to an object, and `xcrun metal` compiles our Metal
source on Apple Silicon. **However, HanaJit does not yet launch kernels on a
GPU** (the host-side `cuLaunchKernel`/HIP/Metal dispatch bridge is on the
roadmap). Today the GPU backends are a verified *compiler target*, not a
runtime. Claims in this repo say "emits and assembles," never "runs on GPU."

**FPGA** support is IR + Vitis HLS TCL *export* only — an FPGA is synthesized,
not JITed, so it can never be a runtime target.

**Not supported** (falls back to CPython): allocating new arrays inside a
kernel, most of the object model (classes, dicts, arbitrary Python objects),
generators, exceptions as control flow, and the long tail of the NumPy API.
HanaJit targets numeric, loop- and array-heavy code, not general Python.

**Numerical notes:** `reduce_reassoc` reorders float additions (like NumPy's
pairwise sum) so results are not bit-identical to a sequential sum, but stay
within the same ~1e-10 tolerance; integers are unaffected. `float32` gives
exact float32 precision (~7 significant digits), a bounded trade-off you opt
into by passing float32 arrays. Experimental modes
(`evolve_hyper`, `rewrite=True`) are opt-in, CPU-only, and documented with
their risks in [`docs/experimental.md`](docs/experimental.md).

---

## Diagnostics

HanaJit ships a self-diagnostic that probes your environment and writes a
report:

```bash
python -m hanajit.doctor
```

It checks compilation, dispatch, threading, caching, the GPU code-generation
backends (and runs the real assemblers if `ptxas` / `llvm-mc` are on your
PATH), and hardware detection — then writes `hanajit_report_<platform>.md`.
Committed example reports live in [`reports/`](reports/).

---

## Project layout

```
hanajit/            the package (frontend, typeinfer, codegen, backends, ...)
  backends/         cpu, gpu (cuda/amd/intel), metal, fpga
docs/               API, architecture, GPU, performance, limitations, ...
benchmarks/         reproducible benchmark scripts
examples/           runnable example programs
tests/              the test suite (207 tests)
reports/            committed doctor reports (Linux / Windows / macOS)
site/               the project landing page
```

---

## Contributing

Issues and pull requests are welcome. Please run the suite before submitting:

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

New optimizations must include tests that verify **correctness against a
reference** before any performance claim — that is the core rule of this
project. Contributions are accepted under the repository's license (below).

---

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).

Apache-2.0 was chosen over MIT for its explicit patent grant and
patent-retaliation clause, which matter for a compiler project in
patent-adjacent fields. If you prefer MIT, the change is a one-file swap plus
a line in `pyproject.toml` — but note that relicensing after external
contributions requires contributors' agreement.

---

## Acknowledgements

Built on [LLVM](https://llvm.org/) and
[llvmlite](https://llvmlite.readthedocs.io/). Benchmarked against
[NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/). Some
ergonomic ideas (helper inlining, auto-parallel `for`) were inspired by
[Taichi](https://github.com/taichi-dev/taichi) — the *ideas*, implemented
without a DSL.

Developed in the R&D pipeline of [EZducate](https://ezducate.ai).
