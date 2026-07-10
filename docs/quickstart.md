# HanaJit Quickstart

*HanaJit ("ha ana" — Moroccan Darija for "here I am" — × JIT): an LLVM-backed
JIT for Python that uses CPython's own parser and falls back to the
interpreter for anything it can't compile, so the whole Python ecosystem
keeps working.*

## Install

```bash
pip install hanajit          # requires llvmlite (installed automatically)
pip install hanajit[test]    # + pytest, numpy (to run the suite)
pip install hanajit[bench]   # + numba (benchmark comparisons)
```

## First kernel

```python
from hanajit import jit

@jit
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fib(32)          # first call compiles for (int,) — then native speed
fib(32.0)        # separate specialization for (float,)
```

Compilation is lazy and per-type-signature (Numba-style). Anything outside
the compilable subset — lists, dicts, classes, numpy calls, arbitrary
imports — triggers a one-time warning and runs on the CPython interpreter
instead. Your code never breaks; it just doesn't accelerate that function.

## The options you'll actually use

```python
@jit                          # CPU, lazy specialization
@jit(cache=True)              # persist machine code to disk (fast restarts)
@jit(nogil=True)              # release the GIL during kernel execution
@jit(fastmath=True)           # allow FP reassociation/FMA (changes rounding)
@jit(target="auto")           # resolve cpu/GPU per kernel on this machine
@jit(target="cuda",           # GPU kernel with pointer args
     signature="f64*, f64*, f64, i64")
```

## Quick wins checklist

- Hot numeric loop in a request handler / script? `@jit(nogil=True)`.
- Short-lived processes or many workers? add `cache=True`.
- Stable argument types in a tight loop? `f2 = f.specialize(int, int)`
  gives the raw native entry point (lowest possible call overhead).
- Not sure it compiled? `type(f).__name__` is `HanaFunction` when the
  native dispatcher is active; watch for a one-time fallback warning.
