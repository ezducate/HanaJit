"""Disk compilation cache.

With @jit(cache=True), compiled machine code (the ELF/Mach-O object emitted
by MCJIT, containing both the kernel and its fastcall wrapper) is written to
disk keyed by a hash of: function source, type signature, host CPU name,
opt/fastmath/nogil flags, and llvmlite/hanajit/Python versions. Subsequent
processes load the object file directly — no parsing, type inference,
LLVM IR generation, or optimization. This eliminates JIT warmup for
short-lived processes and multi-worker servers (each gunicorn/uvicorn
worker warm-starts from the shared cache).

Cache location: $HANAJIT_CACHE_DIR, else ~/.cache/hanajit.
Corrupt or stale entries are ignored and recompiled; saving is best-effort
(a read-only filesystem never breaks compilation).
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def cache_dir():
    d = os.environ.get("HANAJIT_CACHE_DIR")
    return Path(d) if d else Path.home() / ".cache" / "hanajit"


def make_key(source, sig, opts):
    payload = json.dumps({"src": source, "sig": list(sig),
                          "py": list(sys.version_info[:2]), **opts},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def load(key):
    base = cache_dir() / key
    try:
        obj = base.with_suffix(".o")
        meta = base.with_suffix(".json")
        if obj.exists() and meta.exists():
            return obj.read_bytes(), json.loads(meta.read_text())
    except Exception:
        pass
    return None


def save(key, obj_bytes, meta):
    try:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{key}.o.tmp{os.getpid()}"
        tmp.write_bytes(obj_bytes)
        tmp.replace(d / f"{key}.o")            # atomic: safe across workers
        tmp = d / f"{key}.json.tmp{os.getpid()}"
        tmp.write_text(json.dumps(meta))
        tmp.replace(d / f"{key}.json")
    except Exception:
        pass  # caching is best-effort, never fatal


def clear():
    """Remove all cached compilations."""
    import shutil
    shutil.rmtree(cache_dir(), ignore_errors=True)
