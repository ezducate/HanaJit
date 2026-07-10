# Architecture

## Pipeline

```
Python source
  └─ inspect.getsource + ast.parse        (CPython's own parser)
      └─ TypeInferencer                   (fixpoint abstract interpretation:
         │                                 i64/f64/bool/f64*/i64*, driven by
         │                                 actual argument types on first call
         │                                 or by an explicit signature=)
         └─ CodeGen (llvmlite.ir)         (typed AST → LLVM IR; Python
             │                             floor-div/mod semantics; per-vendor
             │                             GPU intrinsics; optional fastmath)
             └─ MCJIT (-O3, host CPU      (or: retarget triple → PTX/GCN/
                name + features)            SPIR-V; or: MSL transpiler)
```

Anything the inferencer or codegen can't handle raises `UnsupportedError`;
the dispatcher catches it and the original Python function runs. That
fallback *is* the full-ecosystem story: compilation is an optimization,
never a compatibility constraint.

## Dispatch: three tiers

**Tier 1 — native vectorcall (CPython ≥3.12, default).** `@jit` returns a
`HanaFunction`: a heap type created via `PyType_FromSpec` (through ctypes,
no C compilation) with `Py_TPFLAGS_HAVE_VECTORCALL` and a per-instance
vectorcall pointer. That pointer targets LLVM-JITed dispatch code which:
checks `kwnames`/arity, compares `Py_TYPE(arg)` pointers against known
specializations, unboxes via direct CPython C-API calls, calls the kernel,
boxes the result. Unknown signatures tail-call the Python slow path via
`PyObject_Vectorcall`; after it compiles a new specialization it rebuilds
the dispatcher and swaps the instance's vectorcall pointer (atomic 8-byte
store). Measured: calling a jitted function is cheaper than calling an
empty pure-Python function.

**Tier 2 — fastcall builtins.** Each specialization also gets an LLVM-
generated `METH_FASTCALL` wrapper installed as a real builtin via
`PyCFunction_NewEx`. `f.specialize(...)` hands you this directly. Wrapper
and kernel share one module, so unboxing+compute inline together at -O3.

**Tier 3 — Python dispatcher.** On <3.12, 32-bit, free-threaded builds, or
any proxy-construction failure, a plain Python `Dispatcher` object handles
calls through ctypes. Slower, never wrong.

## Object-code lifetimes

MCJIT engines own the executable memory; they're appended to a module-level
keepalive list and live for the process. The vectorcall dispatcher bakes
process-specific addresses (kernel entry points, the fallback bound method,
type-object pointers), which is why it is rebuilt per process and per new
specialization, and why only kernel+wrapper object code — never the
dispatcher — goes to the disk cache.

## GIL interaction

Wrappers hold the GIL through unboxing and boxing. With `nogil=True` they
bracket only the kernel call with `PyEval_SaveThread`/`RestoreThread` —
safe because kernels never touch Python objects.
