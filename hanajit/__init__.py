"""hanajit: an LLVM-backed JIT for Python with full-ecosystem fallback.

Uses CPython's own parser (ast), compiles a numeric subset to native code
via llvmlite/LLVM, and transparently falls back to the interpreter for
everything else — so any Python library keeps working.
"""
from .decorator import jit, pmap
from .backends.detect import detect
prange = range
from .errors import UnsupportedError

__version__ = "0.20.0"
__all__ = ["jit", "pmap", "prange", "detect", "UnsupportedError"]
