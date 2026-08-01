import time
from hanajit import jit

def bench(fn, *args, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return r, best

# --- CPU JIT speedup ---
def mandel_escape_py(cr, ci, maxit):
    zr = 0.0; zi = 0.0; total = 0
    for i in range(maxit):
        zr2 = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = zr2
        if zr * zr + zi * zi > 4.0:
            break
        total += 1
    return total

def heavy_py(n):
    acc = 0
    for i in range(n):
        acc += mandel_escape_py(-0.7 + i * 1e-9, 0.27015, 500)
    return acc

@jit
def mandel_escape(cr, ci, maxit):
    zr = 0.0; zi = 0.0; total = 0
    for i in range(maxit):
        zr2 = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = zr2
        if zr * zr + zi * zi > 4.0:
            break
        total += 1
    return total

@jit
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def fib_py(n):
    if n < 2:
        return n
    return fib_py(n - 1) + fib_py(n - 2)

fib(10)  # warm up / compile
r1, t_jit = bench(fib, 32)
r2, t_py = bench(fib_py, 32)
assert r1 == r2
print(f"fib(32):        CPython {t_py*1e3:8.1f} ms | hanajit {t_jit*1e3:8.2f} ms | {t_py/t_jit:6.0f}x")

mandel_escape(-0.7, 0.27015, 500)  # compile
r1, t_jit = bench(mandel_escape, -0.7, 0.27015, 100000000 and 500)
# tight-loop bench: many calls
t0 = time.perf_counter()
for i in range(20000):
    mandel_escape(-0.7 + i * 1e-9, 0.27015, 500)
t_jit = time.perf_counter() - t0
t0 = time.perf_counter()
for i in range(20000):
    mandel_escape_py(-0.7 + i * 1e-9, 0.27015, 500)
t_py = time.perf_counter() - t0
print(f"mandelbrot 20k: CPython {t_py*1e3:8.1f} ms | hanajit {t_jit*1e3:8.2f} ms | {t_py/t_jit:6.1f}x")

# --- full-ecosystem fallback ---
import warnings
@jit
def uses_numpy(n):
    import numpy as np
    return float(np.arange(n).sum())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print("numpy fallback:", uses_numpy(1000), "| warned:", any("falling back" in str(x.message) for x in w))

# --- inspect artifacts ---
print("\n--- LLVM IR (fib, first 6 lines) ---")
print("\n".join(fib.inspect_llvm().splitlines()[:6]))

# --- GPU: NVPTX IR / PTX emission ---
@jit(target="cuda")
def saxpy_core(a, x, y):
    return a * x + y
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    saxpy_core(2.0, 3.0, 4.0)  # triggers emission, runs on CPython
art = saxpy_core.inspect_gpu()
if art:
    vendor, text, native = art
    print(f"\n--- GPU artifact ({vendor}, {'native' if native else 'IR'}) ---")
    print("\n".join(text.splitlines()[:5]))

# --- FPGA: HLS export ---
import os
import tempfile
try:
    _fpga_dir = tempfile.mkdtemp(prefix="hanajit_fpga_")
    fpga = fib.export_fpga(os.path.join(_fpga_dir, "fib_fpga"))
    print(f"\nFPGA export: {fpga.ll}, {fpga.tcl}")
    if fpga.cpp:
        print(f"  HLS C++ + testbench: {fpga.cpp}, {fpga.tb}")
except Exception as e:
    print(f"\nFPGA export skipped: {e}")

# --- WebAssembly export ---
try:
    _wasm_dir = tempfile.mkdtemp(prefix="hanajit_wasm_")
    w = fib.export_wasm(os.path.join(_wasm_dir, "fib"))
    print(f"\nWASM export: {w.ll}, loader {w.mjs}"
          + (f", linked {w.wasm}" if w.wasm else " (no clang: run "
             + w.build + ")"))
except Exception as e:
    print(f"\nWASM export skipped: {e}")

# --- GPU execution: run saxpy on whatever bridge this machine has ---
try:
    import numpy as _np
    from hanajit.backends import rt as _rt
    _vendor = next((v for v in ("cuda", "intel", "vulkan", "amd", "metal")
                    if _rt.get_runtime(v) is not None), None)
    if _vendor is None:
        print("\nGPU launch skipped: no runtime bridge on this machine")
    else:
        @jit(target=_vendor, signature="f64*, f64*, f64, i64")
        def gpu_saxpy(y, x, a, n):
            i = block_id() * block_dim() + thread_id()
            if i < n:
                y[i] = a * x[i] + y[i]
            return 0
        _n = 100_000
        _x = _np.random.rand(_n)
        _y0 = _np.random.rand(_n)
        _y = _y0.copy()
        gpu_saxpy.launch(_y, _x, 2.0, _n)
        _ok = _np.allclose(_y, 2.0 * _x + _y0, rtol=1e-4, atol=1e-4)
        print(f"\nGPU launch ({_vendor}, "
              f"{_rt.get_runtime(_vendor).device_name}): saxpy over "
              f"{_n:,} elements -> {'CORRECT' if _ok else 'WRONG'}")
except Exception as e:
    print(f"\nGPU launch skipped: {e}")

# --- multithreading: nogil kernels ---
import threading
@jit(nogil=True)
def heavy(n):
    acc = 0.0
    for i in range(n):
        acc += (i % 7) * 0.5
    return acc
heavy(10)
t = threading.Thread(target=heavy, args=(300_000_000,)); t.start()
ticks = 0
while t.is_alive():
    ticks += 1
t.join()
print(f"\nnogil: main thread ran {ticks:,} iterations while kernel computed in background")
