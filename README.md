<div align="center">

# Hana Jit

**An LLVM-backed JIT compiler for Python. It compiles ordinary functions and NumPy code to native machine code, and falls back to the interpreter for code it cannot compile.**

By [EZducate](https://www.ezducate.ai) — [www.ezducate.ai](https://www.ezducate.ai)

Author: **Iqbal Addou** · [iqbal.addou@gmail.com](mailto:iqbal.addou@gmail.com) · [cto@ezducate.ai](mailto:cto@ezducate.ai)

[![CI](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml/badge.svg)](https://github.com/ezducate/HanaJit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#status)

</div>

## The name

**Hana Jit** — **هانا جيت** — is written as two words, and is a bilingual play on words.

- **Hana** — from Moroccan Arabic (Darija) **ها أنا** *(ha ana)*, meaning "here I am."
- **Jit** — **JIT**, as in a Just-In-Time compiler. In Moroccan Arabic, **جيت** *(jit)* means "I arrived."

Read either way, "Hana Jit" resolves to "here I am, a JIT compiler" or "here I am, I've arrived." The installable package is named `hanajit` (PyPI names cannot contain spaces).

---

## Overview

Hana Jit compiles a Python function to native machine code through [LLVM](https://llvm.org/) (via [llvmlite](https://llvmlite.readthedocs.io/)) and runs that in place of the interpreter. Typical results are 10–100× faster than CPython, and comparable to [Numba](https://numba.pydata.org/) on the workloads it targets.

It requires no type annotations, no restructured data, and no new language. Add a decorator:

```python
from hanajit import jit

@jit
def sum_squares(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * x[i]
    return total
```

The first call with a given argument type compiles a specialization; subsequent calls with the same types reuse it. Code that Hana Jit cannot compile falls back to the CPython interpreter with a single warning, so existing programs continue to run.

```mermaid
flowchart TD
    Call[Call jitted function] --> Seen{Seen these<br/>argument types?}
    Seen -->|yes| Cached[Run cached<br/>native code]
    Seen -->|no| Infer{Types in the<br/>supported set?}
    Infer -->|yes| Compile[Compile specialization<br/>cache it, run native]
    Infer -->|no| Fallback[Warn once<br/>run in CPython]
    style Cached fill:#EAEBF6,stroke:#2B3FC4
    style Compile fill:#EAEBF6,stroke:#2B3FC4
    style Fallback fill:#FDECEC,stroke:#C44B3F
```


Design goals:

1. **No DSL.** It compiles the Python you wrote, parsed by CPython's own `ast` module — not a restricted dialect or a new syntax.
2. **Correctness.** Every optimization is either provably equivalent to the original code, or an opt-in trade-off (such as float32 precision) documented with its exact cost. Code that cannot be compiled safely runs in the interpreter rather than being miscompiled.
3. **Reproducible measurement.** The benchmark figures below are measured and reproducible from the scripts in [`benchmarks/`](benchmarks/).

Hana Jit was developed in the R&D pipeline at [EZducate](https://ezducate.ai) to accelerate numeric and array-heavy code — on-device inference, simulation, and data processing.

---

## Status

Hana Jit is alpha software. The CPU compiler is stable and tested: **270 tests pass across Python 3.10–3.14 on Linux, Windows 11, and macOS (Apple Silicon).** GPU kernels execute on-device via `f.launch()` through ctypes driver bridges — CUDA, Level Zero, and Vulkan validated on real hardware; HIP and Metal code-complete (see [Limitations](#limitations)). WebAssembly and FPGA are export paths for external toolchains. APIs may change before 1.0; pin a version if you depend on it.

---

## Installation

Requires Python 3.10 or newer. The only dependency is `llvmlite`, which ships prebuilt LLVM wheels for all major platforms. A separate LLVM installation is not required.

**From PyPI:**

```bash
pip install hanajit
```

**From GitHub:**

```bash
pip install "git+https://github.com/ezducate/HanaJit.git"

# pin to a released tag
pip install "git+https://github.com/ezducate/HanaJit.git@v0.23.0"
```

**For development:**

```bash
git clone https://github.com/ezducate/HanaJit.git
cd HanaJit
pip install -e ".[test]"      # editable install with test dependencies
python -m pytest tests/ -q    # run the suite
python -m hanajit.doctor      # environment and capability diagnostic
```

Optional extras: `hanajit[bench]` adds `numba` and `scipy` for the comparison benchmarks; `hanajit[test]` adds the test dependencies.

---

## Features

All features beyond the base `@jit` decorator are opt-in.

### Base decorator

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

The first call with a given argument type compiles a specialization; later calls with the same types reuse it. A call with a different type compiles a separate specialization.

### Fusion engine

A NumPy expression normally allocates a temporary array for every operation (`a * b` produces one array, `+ c` another). Numba does the same. Hana Jit compiles the entire expression into a single loop with no intermediate arrays:

```python
@jit
def score(a, b):
    # compiled to one pass over the data; no temporary arrays are allocated
    return np.sum(np.exp(-a * a) * b + np.where(a > 0, a, 2 * a) - np.clip(b, 0.2, 1.5))
```

This is a structural difference rather than a flag, so it is not affected by Numba tuning options. On a 5-operation expression, Hana Jit runs about 3× faster than NumPy and 3.7× faster than Numba.

```mermaid
flowchart LR
    subgraph NumPy["NumPy / Numba — temporaries"]
        direction LR
        n1["a*a"] --> t1[(temp 1)]
        t1 --> n2["exp(...)"] --> t2[(temp 2)]
        t2 --> n3["* b"] --> t3[(temp 3)]
        t3 --> n4["sum"]
    end
    subgraph Hana["Hana Jit — fused"]
        direction LR
        f1["one loop:<br/>acc += exp(-a[i]*a[i]) * b[i] ..."] --> f2["sum"]
    end
    style t1 fill:#FDECEC,stroke:#C44B3F
    style t2 fill:#FDECEC,stroke:#C44B3F
    style t3 fill:#FDECEC,stroke:#C44B3F
    style f1 fill:#EAEBF6,stroke:#2B3FC4
```


The engine supports ufuncs (`exp`, `sqrt`, `sin`, …), comparisons, `np.where`, `np.clip`, `np.minimum`/`maximum`, and virtual arrays such as `np.arange` and `np.linspace` that are never materialized. Operations outside its scope fall back.

### `reduce_reassoc`

A summation loop (`total += x[i]`) cannot be vectorized by default because each iteration depends on the previous one. NumPy reorders its summation (pairwise) to work around this. `reduce_reassoc=True` grants Hana Jit the same reordering permission, applied only to reduction accumulators:

```python
@jit(reduce_reassoc=True)
def total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]          # vectorizes into parallel SIMD accumulators
    return acc
```

This reaches NumPy-class reduction throughput (about 1.5× the default) without enabling global fast-math. Integer reductions remain bit-exact. Float reductions are reordered the same way NumPy reorders them, matching NumPy to approximately 1 part in 10¹⁰ — not identical to a strict left-to-right sum, but no less accurate than `np.sum`. It also applies to `np.sum`, `np.dot`, and `np.mean`.

### float32

A `float32` array compiles with 32-bit arithmetic: half the memory traffic and twice the SIMD lane count of float64. The dtype selects the path; no flag is required:

```python
@jit(reduce_reassoc=True)
def total(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]
    return acc

total(x.astype(np.float32))   # 32-bit compute path
```

On a memory-bound reduction, float32 with `reduce_reassoc` runs about 2.7× the float64 baseline. The result carries float32 precision (approximately 7 significant digits) — a bounded trade-off, equivalent to computing in float32 elsewhere. Use it where float32 precision is sufficient.

### narrow (experimental)

The integer companion to float32 mode. For a memory-bandwidth-bound integer reduction over a large 1-D `int8` / `int16` / `int32` array, narrow mode loads the narrow elements as SIMD vectors and accumulates in a wide `int64` vector — moving far fewer bytes per element while keeping the result exact:

```python
import numpy as np
from hanajit import jit

@jit
def total(x):
    acc = 0
    for i in range(len(x)):
        acc += x[i]
    return acc

data = np.random.default_rng(0).integers(-100, 100, 50_000_000).astype(np.int8)
result = total.narrow(data, confirmed=True)   # exact int64 sum, ~3× faster
```

The result is bit-identical to the `int64` sum because accumulation is always 64-bit — there is no accumulator overflow (the failure mode of naive narrowing, where an `int8` accumulator wraps around). On a memory-bound sum, measured speedups are roughly `int8` 2.3–3.2×, `int16` 2.0–2.3×, and `int32` 1.5× over an `int64` baseline; these are bandwidth-dependent and vary by hardware.

This mode is experimental and opt-in: it requires `confirmed=True`, exactly like the hyper-aggressive optimizer. Unlike hyper mode, the result is exact — what is experimental is the specialized codegen path and the requirement that the input already be a narrow-dtype array. It currently accelerates the sum reduction over one narrow array; other patterns fall back to the normal compiler with a warning. `int4` and `int2` are not supported on CPU, because there are no sub-byte SIMD load instructions and the bit-unpacking they require eats the bandwidth saving.

For a worked scientific example, [`examples/rdf_narrow.py`](examples/rdf_narrow.py) computes protein-water coordination numbers (a radial distribution function analysis) on a real solvated protein, using `narrow` to reduce millions of per-pair `int8` indicators — the memory-bound integer sum that narrow targets.

### Genetic optimizer

Different CPUs favor different compilation choices (unroll factors, vectorization widths). `evolve()` runs a genetic search over compilation strategies, times each candidate on the current hardware with the supplied data, and installs the fastest:

```python
f = jit(heavy_kernel)
f(example_args)                    # compile
report = f.evolve(example_args)    # search; installs the winner
print(report["speedup"])
```

Every candidate is guaranteed to compute the same result: the genes are semantics-preserving transforms, and each candidate is checked against the baseline before it is timed. In the benchmarks below it is consistently the largest correctness-preserving gain, up to approximately 5× on some kernels.

### Parallelism

```python
from hanajit import jit, prange

# auto-parallelize the outermost loop
@jit(parallel=True)
def process(x, out):
    for i in range(len(x)):
        out[i] = expensive(x[i])
    return 0

# or use prange explicitly
@jit
def process2(x, out):
    for i in prange(len(x)):
        out[i] = expensive(x[i])
    return 0
```

`@jit(nogil=True)` releases the GIL around a kernel so it can run alongside other Python threads. `pmap` parallelizes a function across a batch of argument tuples. Measured speedups on multi-core machines are in the 1.8–3.6× range; memory bandwidth is typically the limiting factor.

### Dispatch overhead

On CPython 3.12+, each jitted function becomes a native vectorcall object whose dispatch is itself compiled machine code. Call overhead is approximately 20–50 nanoseconds, about 3.6× less than Numba.

```mermaid
flowchart TD
    C[Function call] --> T1{Native vectorcall<br/>available? CPython 3.12+}
    T1 -->|yes| V["HanaFunction proxy<br/>~20-50 ns"]
    T1 -->|no| T2{Fastcall<br/>path?}
    T2 -->|yes| FC["fastcall wrapper"]
    T2 -->|no| DP["Python Dispatcher<br/>fallback"]
    V --> N[Native machine code]
    FC --> N
    DP --> N
    style V fill:#EAEBF6,stroke:#2B3FC4
    style N fill:#FBF0DD,stroke:#E8A020
```


### Helper inlining

A small `@jit` function called from another `@jit` function is inlined at the source level before compilation, removing call overhead and allowing the fusion engine to see through it:

```python
@jit
def sq(x):
    return x * x

@jit
def energy(a):
    total = 0.0
    for i in range(len(a)):
        total += sq(a[i]) + sq(a[i] + 1)   # sq() is inlined
    return total
```

### Experimental features

Three features are gated behind explicit opt-ins because they carry additional risk. All are documented in [`docs/experimental.md`](docs/experimental.md).

**`@jit(rewrite=True)`** applies pattern-matched algebraic rewrites — for example, a loop summing an arithmetic series is replaced by its closed-form expression. Each rewrite is individually proven correct and fires only on an exact pattern match.

**`evolve_hyper(..., confirmed=True)`** extends `evolve()` with unsafe floating-point transforms (aggressive reassociation, reciprocals, approximate functions). It keeps the fastest candidate that matches the original within a tolerance across a large batch of random inputs. It does not guarantee correctness on untested inputs, requires `confirmed=True`, and is never cached. In the benchmark table below it is frequently a no-op, because the safe `evolve()` has usually already reached the hardware limit. It is intended for kernels where the aggressive transforms unlock additional gains, and should not be used where an incorrect result is unacceptable.

**`narrow(..., confirmed=True)`** (see the [narrow section](#narrow-experimental) above) accelerates a memory-bound integer sum over an `int8` / `int16` / `int32` array using narrow SIMD loads with wide accumulation. Unlike the two features above, its result is always exact; the opt-in reflects the specialized codegen path and the narrow-storage requirement, not a correctness trade-off.

---

## Benchmarks

Measured on a single core in a shared CI container. Treat the ratios as the signal; absolute milliseconds are noisy — rerun on target hardware with the scripts in [`benchmarks/`](benchmarks/). Compared against NumPy 2.x and Numba 0.66.

### Summary

| Benchmark | Result |
|---|---|
| 5-operation fused NumPy expression | 3.0× vs NumPy, 3.7× vs Numba |
| Reduction, `reduce_reassoc` (float64) | ~1.5× over the default |
| Reduction, `reduce_reassoc` + float32 | ~2.7× over the float64 baseline |
| Reduction, `narrow` int8 (memory-bound sum) | ~2.3–3.2× over the int64 baseline |
| Reduction, `narrow` int16 (memory-bound sum) | ~2.0–2.3× over the int64 baseline |
| `evolve()` genetic optimizer | up to ~5×, correctness-verified |
| Call / dispatch overhead | ~46 ns (3.6× less than Numba) |
| `fib(30)` recursion | 1.7× vs Numba |

### With GA, without GA, hyper-aggressive, and Numba

The same kernel compiled four ways:

| Workload | Hana Jit (plain) | + `evolve()` (safe GA) | + hyper-aggressive | Numba |
|---|---|---|---|---|
| fp reduction | 0.78 ms | **0.23 ms** | 0.79 ms | 0.74 ms |
| poly5 eval | 1.04 ms | **0.22 ms** | 1.01 ms | 0.96 ms |
| transcendental | 3.46 ms | 3.50 ms | 3.46 ms | 3.42 ms |
| dot product | 0.80 ms | **0.31 ms** | 0.37 ms | 0.74 ms |

Notes:

- On scalar loops, plain Hana Jit and Numba are approximately equal, as they share the LLVM backend. Hana Jit's advantages are in fusion, dispatch, float32, and cold start.
- The safe GA (`evolve()`) is the largest gain — up to ~4-5× — and exceeds Numba on every row with available headroom, while guaranteeing an identical result.
- The hyper-aggressive column is frequently a no-op and in some rows slower than the safe GA, because the safe GA already reaches the hardware limit on these kernels. Recommendation: use the safe GA; the hyper-aggressive mode applies only to the narrow set of kernels where the unsafe transforms yield further gains.
- The transcendental row barely changes in any column, as it is bound by the hardware `exp`/`sqrt` units.

Reproduce:

```bash
pip install "hanajit[bench]"
python benchmarks/bench_experimental.py    # rewrite + hyper-aggressive
python benchmarks/bench_reductions.py      # reduce_reassoc + float32
python benchmarks/fourway.py               # the four-way comparison
```

---

## Architecture

Hana Jit is approximately 3,000 lines of Python. One intermediate representation, multiple targets:

1. **Frontend** — `inspect.getsource` + `ast.parse` produce the exact tree CPython would execute. There is no custom parser.
2. **Type inference** — a fixpoint over a small set of types (`int64`, `float64`, `float32`, `bool`, pointers, array shapes). Anything outside the set raises an internal `UnsupportedError`, which becomes a fallback to the interpreter.
3. **Code generation** — the typed tree lowers to LLVM IR, including the fusion engine that compiles array expressions into element generators fused into one loop.
4. **Backends** — the IR module is optimized (`-O3`) and either JIT-compiled for the host CPU, re-targeted for a GPU, exported as WebAssembly, or exported for FPGA synthesis.

```mermaid
flowchart TD
    IR["Typed LLVM IR<br/>(one module)"] --> OPT["LLVM -O3"]
    OPT --> CPU["CPU backend<br/>JIT → runs now ✓"]
    OPT --> NV["NVIDIA → PTX<br/>launch() via nvcuda ✓"]
    OPT --> AMD["AMD → GCN<br/>launch() via HIP + clang"]
    OPT --> INT["Intel → SPIR-V<br/>launch() via Level Zero ✓"]
    OPT --> VLK["Vulkan → SPIR-V GLCompute<br/>launch() via vulkan-1 ✓"]
    OPT --> APL["Apple → Metal<br/>launch() via Metal.framework"]
    OPT --> WASM["WebAssembly → .ll/.s + JS loader<br/>clang links .wasm"]
    OPT --> FPGA["FPGA → HLS C++ + IR + TCL<br/>export-only"]
    style CPU fill:#EAEBF6,stroke:#2B3FC4
    style NV fill:#FBF0DD,stroke:#E8A020
    style AMD fill:#FBF0DD,stroke:#E8A020
    style INT fill:#FBF0DD,stroke:#E8A020
    style VLK fill:#FBF0DD,stroke:#E8A020
    style APL fill:#FBF0DD,stroke:#E8A020
    style WASM fill:#FBF0DD,stroke:#E8A020
    style FPGA fill:#FBF0DD,stroke:#E8A020
```

The CPU backend runs compiled code directly. GPU kernels emit inspectable device code *and* execute on the device through `f.launch()` — pure-ctypes bridges over the vendor driver libraries, no SDK required (see [GPU execution](#gpu-execution) and [Limitations](#limitations)). The WebAssembly and FPGA paths export artifacts for external toolchains (clang / Vitis HLS).



```mermaid
flowchart LR
    A[Python function] -->|inspect.getsource| B[Source text]
    B -->|ast.parse| C[AST]
    C --> D[Type inference<br/>fixpoint lattice]
    D -->|supported| E[LLVM IR<br/>+ fusion engine]
    D -->|unsupported| F[Interpreter fallback<br/>+ one warning]
    E --> G[LLVM -O3]
    G --> H1[Host CPU<br/>JIT machine code]
    G --> H2[GPU target<br/>PTX / GCN / SPIR-V / Vulkan / Metal]
    G --> H3[FPGA<br/>HLS C++ + IR export]
    G --> H4[WebAssembly<br/>wasm32/wasm64 export]
    style F fill:#FDECEC,stroke:#C44B3F
    style H1 fill:#EAEBF6,stroke:#2B3FC4
    style H2 fill:#FBF0DD,stroke:#E8A020
    style H3 fill:#FBF0DD,stroke:#E8A020
    style H4 fill:#FBF0DD,stroke:#E8A020
```

See [`docs/architecture.md`](docs/architecture.md) for detail.

---

## GPU execution

GPU-target kernels run on the device with `f.launch()`. Each vendor bridge is pure ctypes over the library the GPU driver already installs — no CUDA toolkit, no Vulkan SDK, no build step:

```python
import numpy as np
from hanajit import jit

@jit(target="cuda", signature="f64*, f64*, f64, i64")   # or "intel", "vulkan", "amd", "metal"
def saxpy(y, x, a, n):
    i = block_id() * block_dim() + thread_id()
    if i < n:
        y[i] = a * x[i] + y[i]
    return 0

y = np.random.rand(1_000_000); x = np.random.rand(1_000_000)
saxpy.launch(y, x, 2.0, len(y))          # arrays copied over, kernel runs, results copied back
```

Keep data resident and launch asynchronously for tight iteration loops:

```python
yd, xd = saxpy.to_device(y), saxpy.to_device(x)   # upload once
for _ in range(1000):
    saxpy.launch(yd, xd, 0.01, len(y), sync=False)  # ~0.2 ms/launch on CUDA
saxpy.synchronize()
result = yd.to_host()
```

Kernels have workgroup-shared memory, barriers, atomics, and 2-D/3-D thread indexing — enough for the standard reduction patterns:

```python
@jit(target="cuda", signature="f64*, f64*, f64*, i64")
def dot_partials(partials, a, b, n):
    tid = thread_id()
    i = block_id() * block_dim() + tid
    s = shared_f64(256)                   # workgroup-shared array
    acc = 0.0
    if i < n:
        acc = a[i] * b[i]
    s[tid] = acc
    barrier()
    step = 128
    while step > 0:                       # tree reduction in shared memory
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step = step // 2
    if tid == 0:
        partials[block_id()] = s[0]
    return 0
```

| target | driver library | validated on |
|---|---|---|
| `cuda` | `nvcuda` (driver API; PTX driver-JITed) | RTX 2080 Max-Q — bit-exact vs NumPy |
| `intel` | `ze_loader` (Level Zero; hanajit's own SPIR-V generator) | UHD Graphics 630 — bit-exact |
| `vulkan` | `vulkan-1` (vendor-neutral; any 1.1 device with f64/i64 shaders) | RTX 2080 Max-Q — bit-exact |
| `amd` | `amdhip64` (HIP; GCN assembled by clang) | code-complete, awaiting AMD hardware |
| `metal` | Metal.framework (macOS; runtime-compiled MSL) | code-complete, awaiting Apple hardware |

CUDA transcendentals (sin/exp/log/pow) link NVIDIA's libdevice automatically when found — `pip install hanajit[cuda-math]` provides it without a CUDA toolkit. Full details, caveats, and multi-GPU selection: [`docs/gpu.md`](docs/gpu.md).

---

## Standalone x86-64 executables

`export_executable` turns a scalar `@jit` function into a native command-line
program. The output is a single application file and does not import or embed
Python or HanaJit. Positional command-line arguments follow the declared
`i64`/`f64`/`bool` signature; the return value is printed to stdout.

```python
from hanajit import jit

@jit(signature="f64, i64")
def compound(x, years):
    for _ in range(years):
        x *= 1.05
    return x

out = compound.export_executable(
    "compound.exe",       # use "compound" on Linux/macOS
    cuda="optional",      # "off", "optional", or "required"
)
print(out.executable)
```

```bash
compound.exe 1000 10
```

- **`cuda="off"`** creates a CPU-only executable.
- **`cuda="optional"`** embeds a one-thread PTX version, dynamically loads
  the NVIDIA driver when present, and otherwise runs the embedded CPU kernel.
- **`cuda="required"`** embeds PTX but exits with status 70 if CUDA cannot
  run. No CUDA toolkit is needed on the destination machine; only the NVIDIA
  driver is required.
- The exporter targets x86-64 Windows, Linux, and Intel macOS. CUDA modes are
  unavailable on macOS. Apple Silicon and 32-bit x86 are outside this export's
  contract.
- The current CLI contract is scalar-only; pointer and NumPy-array signatures
  are rejected. CUDA mode proves deployment portability but is generally not a
  speedup for one tiny scalar invocation because GPU startup dominates.
- A host C11 toolchain is needed at build time: MSVC is auto-discovered on
  Windows; Clang/GCC are used on Unix. `HANAJIT_CC` overrides discovery. If no
  compiler is available, HanaJit writes the self-contained C source and exact
  `.build.bat`/`.build.sh`; install a compiler and run that script.

The returned `ExecutableExport(executable, object, source, build, ptx)` keeps
all build artifacts available for inspection. At runtime the executable uses
only normal operating-system libraries and, in CUDA mode, the installed NVIDIA
driver.

---

## WebAssembly export

`export_wasm` retargets the same typed LLVM IR to `wasm32` (or `wasm64`) and writes everything needed to run the kernel in a browser or Node:

```python
from hanajit import jit

@jit
def sum_squares(n):
    total = 0.0
    for i in range(n):
        total += i * i
    return total

sum_squares(10)                          # compile first (or pass sig=)
out = sum_squares.export_wasm("out/ss")  # WasmExport(ll, s, mjs, build, wasm)
```

- **`<prefix>.ll`** — the kernel retargeted to `wasm32-unknown-unknown` (clang input).
- **`<prefix>.s`** — WebAssembly assembly emitted by LLVM's wasm backend, for inspection.
- **`<prefix>.mjs`** — an ES-module loader: instantiates the module and maps libm calls (`sin`, `exp`, …) to JavaScript `Math` imports. At the JS boundary `i64` is `BigInt`, `f64`/`bool` are `Number`.
- **`<prefix>.build.sh`** — the exact clang command (`--target=wasm32 … -Wl,--export=<fn>`). Any standard clang has the WebAssembly backend; no Emscripten needed.
- **`<prefix>.wasm`** — the linked module, produced automatically when clang is on PATH (or `HANAJIT_WASM_CLANG` points to one); otherwise run the build script.

`inspect_wasm()` returns `(text, native)` — the assembly text without touching disk. `export_wasm(prefix, sig="f64, f64")` exports a specialization without calling the function first. Pass `bits=64` for `wasm64`.

---

## FPGA export

An FPGA is not a processor that executes an instruction stream; it is reconfigurable hardware. An algorithm targeting an FPGA is synthesized into a circuit — loops become pipelined datapaths, multiplies map to DSP blocks, arrays to on-chip memory. Synthesis requires a licensed toolchain and produces a bitstream that configures the device. This process is ahead-of-time and cannot be performed just-in-time.

Hana Jit exports a complete Vitis HLS project kit. The `export_fpga` method writes up to four files:

```python
from hanajit import jit

@jit(signature="f64*, f64*, f64, i64")
def saxpy(y, x, a, n):
    for i in range(n):
        y[i] = a * x[i] + y[i]
    return 0

out = saxpy.export_fpga("out/saxpy")     # FpgaExport(ll, tcl, cpp, tb)
print(out.cpp)    # out/saxpy_hls.cpp   — synthesizable HLS C++ top function
print(out.tcl)    # out/saxpy_hls.tcl   — runnable Vitis HLS script
```

- **`<prefix>_hls.cpp`** — the kernel transpiled from the typed Python AST to synthesizable C++ (the same way the Metal backend transpiles to MSL), with HLS pragmas already in place: `PIPELINE II=1` on innermost loops, `m_axi` interfaces for pointer arguments, `s_axilite` for scalars and control. Vitis HLS synthesizes this directly — the turnkey route.
- **`<prefix>_tb.cpp`** — a C-simulation testbench for `csim_design`.
- **`<prefix>_hls.tcl`** — a runnable project script: `csim → csynth → export_design`, targeting an Alveo U250 at 3.3 ns by default. Both are parameters: `export_fpga(prefix, part="xcvu9p-…", clock_ns=5.0)`.
- **`<prefix>.ll`** — the typed LLVM IR, for IR-level flows: AMD/Xilinx Vitis HLS ingests IR through its LLVM front-end flow, and LLVM's [CIRCT](https://circt.llvm.org/) project lowers LLVM IR to hardware dialects (FIRRTL/Calyx) and emits Verilog.

Kernels outside the transpilable subset (e.g. NumPy-array signatures — use raw-pointer signatures like `"f64*"` instead) still export `.ll` plus a TCL stub; `out.cpp` and `out.tb` are `None`.

### Testing the FPGA export

The export can be tested without FPGA hardware or a licensed toolchain:

```python
from hanajit import jit

@jit(signature="f64*, f64*, i64")
def dot(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

out = dot.export_fpga("dot_export")
print(open(out.cpp).read())                      # HLS C++ with pragmas
print(open(out.tcl).read())                      # Vitis HLS script
```

With the Vitis toolchain and a board, the next step is `vitis_hls -f dot_export_hls.tcl`, followed by place-and-route to a bitstream — steps that occur in AMD's tools, outside Hana Jit. The export path is tested (the files are written, the C++ is self-contained, and the IR is self-contained); no bitstream is produced in CI, as that requires Vitis and hardware.

---

## Limitations

**Supported:** numeric code, loops, recursion, scalar math, and a subset of NumPy (elementwise operations, the fusion-engine operations, reductions, slicing, 1-D and 2-D indexing, `float32`/`float64`/`int64` arrays).

**Falls back to the interpreter** (with one warning): allocating new arrays inside a kernel, most of the object model (classes, dictionaries, arbitrary objects), generators, exceptions as control flow, string manipulation, `float16`/`complex` dtypes, and the remainder of the NumPy API. Hana Jit targets numeric kernels; code outside that scope runs in the interpreter.

**GPU execution is explicit and experimental.** `f.launch(*args, grid=, block=)` executes a GPU-target kernel on the device through a pure-ctypes bridge over the vendor's driver library (no SDK needed): CUDA (`nvcuda`), Intel (Level Zero), Vulkan (any 1.1 device with `shaderFloat64`/`shaderInt64`), AMD (HIP; needs a clang to assemble the code object), and Metal (macOS). The CUDA, Level Zero, and Vulkan bridges are validated on real hardware (RTX 2080 Max-Q, UHD 630); the HIP and Metal bridges are code-complete but not yet hardware-validated. Calling a GPU-target function *directly* (`f(...)`) still falls back to CPython — device execution never happens implicitly. Per-vendor caveats: CUDA transcendentals (sin/exp/log/pow) run at full f64 precision when NVIDIA's libdevice is found (CUDA toolkit or `pip install hanajit[cuda-math]`), and refuse to launch without it; Vulkan computes those at float32 precision; Metal computes all f64 at float32 (no double in Metal). Plain launches copy arrays both ways; `f.to_device(arr)` returns a resident DeviceArray that skips the copies across launches (~80× lower launch overhead measured on CUDA), and `launch(..., sync=False)` + `f.synchronize()` queues kernels without blocking. Kernels can use workgroup-shared memory (`shared_f64(N)`), `barrier()`, `atomic_add()` (all targets except Metal), and 2-D/3-D thread indexing — enough for the standard shared-memory reduction patterns, verified bit-exact on CUDA, Level Zero, and Vulkan.

**Vulkan SPIR-V emission is best-effort.** LLVM's shader-flavor SPIR-V backend rejects constructs that are routine in the other targets (notably buffer indexing under logical addressing), so emission runs in an isolated subprocess: kernels it accepts yield real `GLCompute` SPIR-V; the rest fall back to annotated LLVM IR (`hlsl.shader`/`hlsl.numthreads` attributes in place) for offline lowering. The workgroup size is fixed at compile time (`HANAJIT_VULKAN_LOCAL_SIZE`, default `64,1,1`), and `block_dim()` folds to that constant.

**WebAssembly is export only.** `export_wasm` writes retargeted IR, wasm assembly, a JS loader, and a build script; the final `.wasm` link needs any standard clang (run automatically when found). Hana Jit does not embed a wasm runtime.

**FPGA is export only** — see the section above. It writes synthesizable HLS C++, a testbench, IR, and an HLS script; synthesis occurs in external tools.

**Numerical behavior:** `reduce_reassoc` reorders float additions (as NumPy does), so results are not bit-identical to a sequential sum but remain within NumPy-level tolerance; integers are unaffected. `float32` carries float32 precision. `evolve_hyper` does not guarantee correctness on untested inputs.

See [`docs/limitations.md`](docs/limitations.md) for the full list.

---

## Diagnostics

```bash
python -m hanajit.doctor
```

The diagnostic checks compilation, dispatch, threading, caching, and the GPU code-generation backends, and its **launch** section runs a real kernel on every available GPU runtime bridge and reports the device it used (or the precise reason a vendor cannot launch on this machine). If `ptxas` or `llvm-mc` are on the PATH, it also runs the vendor assemblers to validate the generated GPU code. It writes `hanajit_report_<platform>.md`. Example reports for Linux, Windows, and macOS are in [`reports/`](reports/).

---

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — walkthrough
- [`docs/api.md`](docs/api.md) — full API reference
- [`docs/architecture.md`](docs/architecture.md) — compiler internals
- [`docs/gpu.md`](docs/gpu.md) — GPU backends and validation
- [`docs/performance.md`](docs/performance.md) — benchmark detail
- [`docs/numpy-coverage.md`](docs/numpy-coverage.md) — supported NumPy operations
- [`docs/experimental.md`](docs/experimental.md) — experimental features
- [`docs/limitations.md`](docs/limitations.md) — full limitations list
- [`docs/publishing.md`](docs/publishing.md) — release process
- [`examples/`](examples/) — example programs

---

## Contributing

Issues and pull requests are welcome. Run the suite before submitting:

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

New optimizations must include tests that verify the result against a reference before any performance claim. Contributions are accepted under the repository's license.

---

## Contact

Hana Jit is developed by **Iqbal Addou** at [EZducate](https://www.ezducate.ai) ([www.ezducate.ai](https://www.ezducate.ai)).

- Email: [iqbal.addou@gmail.com](mailto:iqbal.addou@gmail.com)
- Work: [cto@ezducate.ai](mailto:cto@ezducate.ai)
- Issues: [github.com/ezducate/HanaJit/issues](https://github.com/ezducate/HanaJit/issues)

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

---

## Acknowledgements

Built on [LLVM](https://llvm.org/) and [llvmlite](https://llvmlite.readthedocs.io/). Benchmarked against [NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/). The helper-inlining and auto-parallelization features were informed by [Taichi](https://github.com/taichi-dev/taichi), implemented without a DSL. Developed at [EZducate](https://ezducate.ai).
