<div align="center">

# Hana Jit

**A little compiler that turns ordinary Python into fast native code — no rewrite, no new language, and it gets out of your way when it can't help.**

[![CI](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml/badge.svg)](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#a-note-on-status)

</div>

## About the name

**Hana Jit** is two words, and it's a bit of a bilingual pun that I couldn't resist.

- **Hana** — from the Moroccan Darija **ها أنا** *(ha ana)*, which means **"here I am."**
- **Jit** — on the surface it's just **JIT**, as in *Just-In-Time compiler*. But in Darija, **جيت** *(jit)* means **"I arrived."**

So depending on how you read it, "Hana Jit" is either *"here I am, a JIT compiler"* or *"here I am, I've arrived."* Both are true, and both are the feeling I wanted: a compiler that shows up, does the work, and is just… here when you need it. (It's written as two words on purpose — **Hana Jit**, not HanaJit — though the package you install is `hanajit`, because PyPI doesn't love spaces.)

---

## What is it, really?

You've got a Python function. Maybe a numerical loop, maybe some NumPy, maybe a bit of recursion. It works, but it's slow — the kind of slow where pure Python is a bottleneck but rewriting it in C or CUDA feels like a bigger project than it's worth.

Hana Jit sits in that gap. You add one decorator:

```python
from hanajit import jit

@jit
def sum_squares(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * x[i]
    return total
```

…and the first time you call it, Hana Jit reads your function, compiles it down to native machine code through [LLVM](https://llvm.org/), and runs *that* instead of the interpreter. Often **10–100× faster than plain CPython**, and neck-and-neck with (or ahead of) [Numba](https://numba.pydata.org/) on the things it's built for.

Three ideas shaped the whole project:

1. **No new language to learn.** It compiles the Python you already wrote — parsed by CPython's own `ast` module, not some restricted lookalike dialect. No type annotations required, no special data containers.
2. **It never lies to you about correctness.** Every optimization is either provably identical to your original code, or a *clearly-labeled, opt-in trade-off* (like using float32 precision) with the exact cost written down. If Hana Jit can't compile something safely, it quietly runs your code in normal Python instead of compiling something wrong.
3. **Every number here is real.** All the benchmarks below are measured, and reproducible from the scripts in [`benchmarks/`](benchmarks/). Where a feature ties, or loses, or does nothing — I say so. There's enough hype in this space already.

It was built inside the R&D work at [EZducate](https://ezducate.ai) (an AI special-education platform) to speed up the numeric and array-heavy code that keeps showing up — on-device inference, simulation, data crunching — and it grew from there.

---

## A note on status

**Hana Jit is alpha.** The CPU compiler is solid and well-tested — **207 tests passing across Python 3.10–3.14 on Linux, Windows 11, and macOS (Apple Silicon).** The GPU support is **code generation only** right now: it emits real GPU assembly that real vendor tools accept, but it doesn't yet *launch* kernels on a GPU (more on that honestly, below). APIs might shift before a 1.0. If you build on it, pin a version.

---

## Installing it

You need **Python 3.10 or newer**. The only dependency is `llvmlite`, which ships its own prebuilt LLVM — so **you do not need to install LLVM yourself.** It just works.

**From PyPI:**

```bash
pip install hanajit
```

**From GitHub** (handy for the very latest, or a specific commit):

```bash
pip install "git+https://github.com/ezducate/HanaJit.git"

# pin to a released tag
pip install "git+https://github.com/ezducate/HanaJit.git@v0.20.1"
```

**For development** (clone and work on it):

```bash
git clone https://github.com/ezducate/HanaJit.git
cd HanaJit
pip install -e ".[test]"      # editable, with test dependencies
python -m pytest tests/ -q    # run the suite
python -m hanajit.doctor      # check what your machine supports
```

Optional extras: `pip install "hanajit[bench]"` pulls in `numba` and `scipy` so you can run the comparison benchmarks; `hanajit[test]` pulls in the test dependencies.

---

## The features, with examples

Here's the whole toolbox. Everything is opt-in beyond the basic `@jit`, and everything degrades gracefully.

### The basics: just decorate it

```python
from hanajit import jit
import numpy as np

@jit
def norm(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * x[i]
    return total ** 0.5

norm(np.random.rand(1_000_000))
```

The first call with a given argument *type* compiles a specialized version; later calls with the same types reuse it instantly. Call it with a different type (say, an integer array instead of a float array) and it compiles a second specialization. You never manage any of this.

### The fusion engine — NumPy without the temporaries

This is the piece I'm proudest of. When you write a NumPy expression, NumPy allocates a temporary array for *every* operation — `a * b` makes one array, `+ c` makes another, and so on. Numba does the same. Hana Jit instead compiles the whole expression into a **single loop with no intermediate arrays at all**:

```python
@jit
def score(a, b):
    # this whole thing becomes ONE pass over the data.
    # no temporary arrays are ever created.
    return np.sum(np.exp(-a * a) * b + np.where(a > 0, a, 2 * a) - np.clip(b, 0.2, 1.5))
```

Because that's a *structural* difference — not a flag you flip — no amount of Numba tuning closes the gap. On a 5-operation expression like this one, Hana Jit runs about **3× faster than NumPy and 3.7× faster than Numba.**

The engine understands ufuncs (`exp`, `sqrt`, `sin`, …), comparisons, `np.where`, `np.clip`, `np.minimum`/`maximum`, and even "virtual" arrays like `np.arange` and `np.linspace` that never actually get built in memory. Anything the fusion engine can't handle falls back cleanly.

### `reduce_reassoc` — reductions at NumPy speed

A plain summation loop (`total += x[i]`) can't be vectorized by the compiler, because each step depends on the one before it. NumPy gets around this by summing in a reordered, parallel way (pairwise summation). `reduce_reassoc=True` gives Hana Jit that same permission — but *only* on reduction accumulators, nothing else:

```python
@jit(reduce_reassoc=True)
def total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]          # now vectorizes into parallel SIMD accumulators
    return acc
```

This reaches **NumPy-class reduction speed (about 2.1× the default)** without turning on global fast-math. Integer reductions stay bit-for-bit exact. Float reductions get reordered the same way NumPy already reorders them, so the result matches NumPy to about 1 part in 10¹⁰ — not identical to a strict left-to-right sum, but no less accurate than what you'd get from `np.sum`. It also speeds up `np.sum`, `np.dot`, and `np.mean`.

### float32 — half the bytes, twice the lanes

Pass a `float32` array and Hana Jit compiles the kernel with real 32-bit math: half the memory traffic, and twice as many values per SIMD instruction. No flag — the dtype drives it:

```python
@jit(reduce_reassoc=True)
def total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]
    return acc

total(x.astype(np.float32))   # 32-bit compute path
```

On a memory-bound reduction, float32 with `reduce_reassoc` runs about **4.3× the float64 baseline.** The trade-off is honest and bounded: you get exact *float32* precision (roughly 7 significant digits), not undefined behavior — the same precision you'd get computing in float32 anywhere else. Use it where 7 digits is plenty (a lot of ML inference, graphics, and large reductions), and don't where you need 15.

### The genetic optimizer — `evolve()` tunes to your machine

Different CPUs like different compilation choices — unroll factors, vectorization widths, and so on. Rather than guess, `evolve()` runs a small **genetic search over compilation strategies**, times each candidate on *your* hardware with *your* data, and keeps the fastest one:

```python
f = jit(heavy_kernel)
f(example_args)                    # compile once
report = f.evolve(example_args)    # search; installs the winner
print(report["speedup"])
```

The important part: **every candidate is guaranteed to compute the same answer.** The genes are all semantics-preserving transforms, and each candidate is checked against the baseline before its time even counts. So this is free speed — it can only make things faster, never wrong. In the benchmarks below it's consistently the biggest safe win (up to ~5× on some kernels).

### Parallelism — threads without the ceremony

```python
from hanajit import jit, prange

# auto-parallelize the outermost loop
@jit(parallel=True)
def process(x, out):
    for i in range(len(x)):
        out[i] = expensive(x[i])
    return 0

# or be explicit with prange
@jit
def process2(x, out):
    for i in prange(len(x)):
        out[i] = expensive(x[i])
    return 0
```

There's also `@jit(nogil=True)` to release the GIL around a kernel (so it runs alongside other Python threads), and `pmap` to parallelize a function across a batch of argument tuples. Realistic speedups on multi-core machines are in the 1.8–3.6× range — memory bandwidth is usually the real ceiling, and the docs don't pretend otherwise.

### Near-instant calls

On CPython 3.12+, each jitted function becomes a native "vectorcall" object whose dispatch is itself compiled machine code. Calls land in the **~20–50 nanosecond** range — roughly **3.6× less overhead than Numba** — which matters a lot for small functions called in tight loops.

### Helper inlining — small functions disappear into their callers

If you call one `@jit` function from another, Hana Jit inlines the small one at the source level before compiling, so there's no call overhead and the fusion engine can see straight through it:

```python
@jit
def sq(x):
    return x * x

@jit
def energy(a):
    total = 0.0
    for i in range(len(a)):
        total += sq(a[i]) + sq(a[i] + 1)   # sq() is inlined away
    return total
```

### The experimental corner (opt-in, clearly labeled)

Two features live behind explicit opt-ins because they take on more risk. They're documented in full in [`docs/experimental.md`](docs/experimental.md).

**`@jit(rewrite=True)`** applies pattern-matched algebraic rewrites — for example, a loop that sums an arithmetic series collapses to its closed-form formula. Each rewrite is individually proven correct; it only fires when the exact pattern matches.

**`evolve_hyper(..., confirmed=True)`** is like `evolve()` but allows genuinely *unsafe* floating-point transforms (aggressive reassociation, reciprocals, approximate functions). It keeps whatever is fastest that matches the original within a tolerance **on a large batch of random test inputs** — but it does **not** guarantee correctness on inputs it didn't test, it requires `confirmed=True`, and it's never cached. In practice (see the benchmark table) it's often a no-op, because the *safe* `evolve()` has usually already reached the hardware limit. It exists for the rare kernel where the aggressive flags unlock something, and it's honest about being a "try it and measure" tool rather than a guaranteed win. **Don't use it for anything where a wrong answer causes harm.**

---

## Benchmarks

All measured on a single core in a shared CI container, so treat the **ratios** as the signal and the absolute milliseconds as noisy — rerun on your own hardware with the scripts in [`benchmarks/`](benchmarks/). Compared against NumPy 2.x and Numba 0.66.

### The headline numbers

| What | Result |
|---|---|
| 5-operation fused NumPy expression | **3.0× vs NumPy, 3.7× vs Numba** |
| Reduction, `reduce_reassoc` (float64) | **2.1×** over the default |
| Reduction, `reduce_reassoc` + float32 | **4.3×** over the float64 baseline |
| `evolve()` genetic optimizer | up to **~5×**, always correctness-verified |
| Call / dispatch overhead | **~46 ns** (3.6× less than Numba) |
| `fib(30)` recursion | **1.7× vs Numba** |

### The one you asked for: with GA, without GA, hyper-aggressive, and Numba

This is the honest four-way comparison across a few representative kernels. Each column is the same kernel, compiled differently:

| Workload | Hana Jit (plain) | + `evolve()` (safe GA) | + hyper-aggressive | Numba |
|---|---|---|---|---|
| fp reduction | 0.78 ms | **0.23 ms** | 0.79 ms | 0.74 ms |
| poly5 eval | 1.04 ms | **0.22 ms** | 1.01 ms | 0.96 ms |
| transcendental | 3.46 ms | 3.50 ms | 3.46 ms | 3.42 ms |
| dot product | 0.80 ms | **0.31 ms** | 0.37 ms | 0.74 ms |

How to read this honestly:

- **Plain Hana Jit ≈ Numba.** On bare loops they're basically tied — same LLVM backend underneath, so the generated loop code is a wash. Hana Jit's real edges are elsewhere (fusion, dispatch, float32, cold-start).
- **The safe GA (`evolve()`) is the star.** It's the biggest mover by far — up to ~4-5× — and it beats Numba on every row where there's headroom. And it's *guaranteed correct*. If you take one thing from this table: **run `evolve()`**.
- **Hyper-aggressive is often a no-op**, and I want to be blunt about that. Look at the column — it mostly matches plain Hana Jit and *loses* to the safe GA. That's because the safe GA already reaches the hardware roofline on these kernels, leaving nothing for the "unsafe" transforms to gain. It took on correctness risk and bought basically nothing. On this workload set, the honest recommendation is: **use the safe GA, skip hyper.** It's in the toolbox for the rare case it helps, labeled for what it is.
- **transcendental barely moves** in any column, because it's bottlenecked on the hardware's `exp`/`sqrt` units — no compiler flag changes that.

Reproduce this table yourself:

```bash
pip install "hanajit[bench]"
python benchmarks/bench_experimental.py    # rewrite + hyper-aggressive
python benchmarks/bench_reductions.py      # reduce_reassoc + float32
python benchmarks/fourway.py               # the four-way comparison
```

---

## How it works (the short version)

Hana Jit is about 3,000 lines of readable Python. One intermediate representation, many targets:

1. **Frontend** — `inspect.getsource` + `ast.parse` hand back the exact tree CPython would run. No custom parser exists.
2. **Type inference** — a fixpoint over a small set of types (`int64`, `float64`, `float32`, `bool`, pointers, array shapes). Anything outside that set raises an internal `UnsupportedError`, which becomes a transparent fallback to the interpreter.
3. **Code generation** — the typed tree lowers to LLVM IR, including the fusion engine that turns array expressions into element generators fused into one loop.
4. **Backends** — that one IR module is optimized (`-O3`) and either JIT-compiled for your exact CPU, re-targeted for a GPU, or exported for FPGA synthesis.

There's a proper tour in [`docs/architecture.md`](docs/architecture.md).

---

## FPGAs — what Hana Jit does and doesn't do here

This one needs a careful, honest explanation, because FPGAs are fundamentally different from CPUs and GPUs.

**An FPGA isn't a processor you send instructions to — it's reconfigurable hardware.** There's no instruction stream to emit. Instead, your algorithm has to be *synthesized into a physical circuit*: loops become pipelined datapaths, multiplies map onto DSP blocks, arrays onto on-chip memory. That synthesis takes hours and a large licensed toolchain, and it produces a "bitstream" that reconfigures the chip's wiring. **This can never be a Just-In-Time process** — it's inherently ahead-of-time.

So Hana Jit's role is deliberately narrow and clearly scoped: it does the part a Python compiler is actually qualified to do — **produce clean, HLS-friendly LLVM IR** — and hands off to the real hardware tools for the rest. The `export_fpga` method writes two files:

```python
from hanajit import jit

@jit
def saxpy(y, x, a, n):
    for i in range(n):
        y[i] = a * x[i] + y[i]
    return 0

saxpy(y, x, 2.0, len(y))                 # compile it first
ll_path, tcl_path = saxpy.export_fpga("out/saxpy")
print(ll_path)    # out/saxpy.ll        — self-contained LLVM IR
print(tcl_path)   # out/saxpy_hls.tcl   — a Vitis HLS project script
```

- **`<prefix>.ll`** is the same typed, verified LLVM IR the CPU and GPU backends use. This matters because the FPGA high-level-synthesis tools speak exactly this language: **AMD/Xilinx Vitis HLS is itself built on LLVM** and can ingest IR through its front-end flow, and LLVM's [CIRCT](https://circt.llvm.org/) project lowers LLVM IR to hardware dialects (FIRRTL/Calyx) and emits Verilog. Either flow can pick up this file.
- **`<prefix>_hls.tcl`** is a ready-to-run Vitis HLS project script: it sets the top function, targets a board (an Alveo U250 by default), sets a clock constraint, runs synthesis, and exports an IP block. It's a starting scaffold you'd tune with HLS pragmas for your specific design.

### How to test the FPGA export

You don't need any FPGA hardware or licensed tools to test the *export itself* — that's the whole point of keeping this step lightweight:

```python
import numpy as np
from hanajit import jit

@jit
def dot(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

a = np.ones(64); b = np.ones(64)
dot(a, b, 64)                                    # compile
ll, tcl = dot.export_fpga("dot_export")          # writes dot_export.ll + .tcl

# verify the IR is well-formed and self-contained
print(open(ll).read()[:400])                     # should be valid LLVM IR
print(open(tcl).read())                          # the Vitis HLS script
```

If you *do* have the Vitis toolchain and a board, the next step is `vitis_hls -f dot_export_hls.tcl`, then place-and-route to a bitstream — but that's outside Hana Jit, in AMD's tools. What Hana Jit guarantees and tests is that the exported IR is valid and pipelining-friendly. (The exported IR is plain scalar compute, which is exactly what HLS tools pipeline well.)

**Honest status:** the *export path* is tested — the files are written and the IR is self-contained. There's no bitstream in CI, because that needs Vitis and hardware. If you work with FPGAs, running one exported kernel through your real Vitis flow is the natural way to take this from "export tested" to "hardware verified."

---

## Limitations — read this before you rely on it

I'd rather you know the edges up front than discover them at a bad time.

**What compiles well:** numeric code, loops, recursion, scalar math, and a useful chunk of NumPy (elementwise ops, the fusion-engine operations, reductions, slicing, 1-D and 2-D indexing, `float32`/`float64`/`int64` arrays). This is the sweet spot: array- and loop-heavy numerical work.

**What falls back to normal Python** (correctly, with one warning): allocating brand-new arrays *inside* a kernel, most of the object model (classes, dictionaries, arbitrary objects), generators, using exceptions as control flow, string manipulation, `float16`/`complex` dtypes, and the long tail of the NumPy API. Hana Jit targets numeric kernels, not general-purpose Python — and when it meets code outside its lane, it runs it in the interpreter rather than refusing or, worse, miscompiling.

**GPU is code-generation-only, and this is the big one.** Hana Jit *emits* GPU assembly for four targets — NVIDIA (PTX), AMD (GCN), Intel (SPIR-V), and Apple (Metal) — and this output is validated by the **real vendor toolchains**: NVIDIA's own `ptxas` assembles the PTX into a cubin, LLVM's AMDGPU `llvm-mc` assembles the GCN into an object file, and `xcrun metal` compiles the Metal source on Apple Silicon. That's a strong, verifiable claim — the generated code is real and vendor-valid. **But Hana Jit does not yet *launch* kernels on a GPU.** The host-side machinery to allocate device memory, copy data over, and dispatch the kernel (`cuLaunchKernel` and friends) is on the roadmap, not done. So today the GPU backends are a verified *compiler target*, not a runtime. Everywhere in this project, GPU claims say "emits and assembles," never "runs on GPU." I'd rather under-claim than oversell.

**FPGA is export-only** — see the section above. It writes IR + an HLS script; synthesis happens in external tools.

**Numerical fine print:** `reduce_reassoc` reorders float additions (like NumPy does), so results aren't bit-identical to a strict sequential sum but stay within NumPy-level tolerance; integers are unaffected. `float32` gives exact float32 precision, a bounded trade-off you opt into. The experimental `evolve_hyper` mode explicitly does not guarantee correctness on untested inputs.

There's a fuller list in [`docs/limitations.md`](docs/limitations.md).

---

## Checking your setup

Hana Jit ships a diagnostic that probes your machine and writes a report:

```bash
python -m hanajit.doctor
```

It checks compilation, dispatch, threading, caching, and the GPU code-generation backends — and if `ptxas` or `llvm-mc` are on your PATH, it runs the real assemblers to confirm the generated GPU code is valid. Then it writes `hanajit_report_<platform>.md`. There are example reports from Linux, Windows, and macOS in [`reports/`](reports/).

---

## The rest of the docs

- [`docs/quickstart.md`](docs/quickstart.md) — a gentler walkthrough
- [`docs/api.md`](docs/api.md) — every option, in detail
- [`docs/architecture.md`](docs/architecture.md) — how the compiler is built
- [`docs/gpu.md`](docs/gpu.md) — the GPU backends and how they're validated
- [`docs/performance.md`](docs/performance.md) — deeper benchmark notes
- [`docs/numpy-coverage.md`](docs/numpy-coverage.md) — exactly which NumPy is supported
- [`docs/experimental.md`](docs/experimental.md) — the risky opt-in features
- [`docs/limitations.md`](docs/limitations.md) — the full list of edges
- [`docs/publishing.md`](docs/publishing.md) — how releases go to PyPI
- [`examples/`](examples/) — runnable programs

---

## Contributing

Issues and pull requests are genuinely welcome. Please run the suite first:

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

One firm rule: **any new optimization must come with tests that check it against a correct reference, before any performance claim.** That's the core discipline of this project — fast is only worth having if it's also right. Contributions are accepted under the repository's license.

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE). I chose Apache over MIT for its explicit patent grant and patent-retaliation clause, which matter for a compiler in patent-adjacent territory.

---

## Thanks

Built on [LLVM](https://llvm.org/) and [llvmlite](https://llvmlite.readthedocs.io/). Benchmarked honestly against [NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/). A couple of the ergonomic ideas (inlining small helpers, auto-parallelizing the outer loop) were inspired by [Taichi](https://github.com/taichi-dev/taichi) — the *ideas*, built here without any DSL.

Made with care in the R&D pipeline at [EZducate](https://ezducate.ai). ها أنا — here I am. جيت — I arrived.
