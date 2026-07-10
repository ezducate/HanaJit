"""Four-way benchmark: CPython vs numba vs hanajit vs hanajit+GA.
Writes RESULTS-FOURWAY.md. GA rows use evolve() with the same budget per
kernel (gens=6, pop=10); fastmath is allowed only where noted, and numba
gets a fastmath=True datapoint on those kernels for fairness."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit
from numba import njit

L = []
def emit(s=""):
    print(s); L.append(s)

def best(fn, *a, reps=5):
    b = float("inf"); r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(*a); b = min(b, time.perf_counter() - t0)
    return r, b

GA = dict(generations=6, population=10, reps=4, seed=0)
ROWS = []

def bench(name, pyfn, args, fm=False, nb_variant=None, py_reps=2):
    jf = jit(pyfn)
    nf = nb_variant or njit(pyfn)
    small = tuple((min(x, 64) if isinstance(x, int) else
                   (x[:64] if isinstance(x, np.ndarray) else x)) for x in args)
    try:
        jf(*small); nf(*small)
    except Exception:
        jf(*args); nf(*args)
    r_py, t_py = best(pyfn, *args, reps=py_reps)
    r_nb, t_nb = best(nf, *args)
    r_hj, t_hj = best(jf, *args)
    rep = jf.evolve(*args, allow_fastmath=fm, **GA)
    r_ga, t_ga = best(jf, *args)
    tol = 1e-6 * max(1.0, abs(r_py))
    assert abs(r_py - r_nb) < tol and abs(r_py - r_hj) < tol \
        and abs(r_py - r_ga) < tol, name
    extra = ""
    if fm:
        nfm = njit(fastmath=True)(pyfn); nfm(*small)
        _, t_nfm = best(nfm, *args)
        extra = f" (numba fastmath: {t_nfm*1e3:,.1f} ms)"
    ROWS.append((name, t_py, t_nb, t_hj, t_ga, rep["installed"], extra))

# 1. recursion
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
if True:
    from numba import njit as _nj
    @_nj
    def fib_nb(n):
        if n < 2:
            return n
        return fib_nb(n - 1) + fib_nb(n - 2)
bench("fib(30) recursion", fib, (30,), nb_variant=fib_nb)

# 2. int loop, mod + branch
def iloop(n):
    acc = 0
    for i in range(n):
        acc += i % 7 - i % 3 if i % 2 == 0 else i % 5
    return acc
bench("int loop 20M (mod/branch)", iloop, (20_000_000,))

# 3. float serial recurrence
def logi(n):
    acc = 0.0; x = 0.5
    for i in range(n):
        x = 3.999 * x * (1.0 - x)
        acc += x
    return acc
bench("logistic map 20M (serial fp)", logi, (20_000_000,), fm=True)

# 4. fp array reduction (vectorizable under fastmath)
A2 = np.random.default_rng(0).uniform(-1, 1, 2_000_000)
def fred(x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * x[i] + 0.5 * x[i]
    return s
def fred_nb_src(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i] + 0.5 * x[i]
    return s
bench("fp reduction 2M", fred, (A2,), fm=True, nb_variant=njit(fred_nb_src))

# 5. fused numpy expression
B2 = np.random.default_rng(1).uniform(0.1, 2, 2_000_000)
def fexpr(a, b):
    return np.sum(np.exp(-a * a) * b + np.where(a > 0.0, a, 2.0 * a)
                  - np.clip(b, 0.2, 1.5))
bench("fused numpy expr 2M (5 ops)", fexpr, (A2, B2), fm=True)

# 6. mandelbrot batch
def mand(k):
    total = 0
    for t in range(k):
        cr = -0.74 + t * 1e-6; ci = 0.11
        zr = 0.0; zi = 0.0; it = 0
        while it < 300 and zr * zr + zi * zi <= 4.0:
            zr2 = zr * zr - zi * zi + cr
            zi = 2.0 * zr * zi + ci
            zr = zr2
            it += 1
        total += it
    return total
bench("mandelbrot x4000", mand, (4000,), fm=True)

emit("# Four-way benchmark\n")
emit("| workload | CPython | numba | hanajit | hanajit+GA | GA gain | vs numba (GA) |")
emit("|---|---|---|---|---|---|---|")
for name, tp, tn, th, tg, inst, extra in ROWS:
    emit(f"| {name} | {tp*1e3:,.1f} ms | {tn*1e3:,.1f} ms{extra} | "
         f"{th*1e3:,.1f} ms | {tg*1e3:,.1f} ms | "
         f"{th/tg:.2f}x{'*' if not inst else ''} | {tn/tg:.2f}x |")
emit("\n`*` = GA found no improvement; baseline kept (never a regression).")
emit("fastmath allowed for GA on: logistic, fp reduction, fused expr, "
     "mandelbrot — numba fastmath datapoints shown for fairness.")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
     "RESULTS-FOURWAY.md"), "w").write("\n".join(L) + "\n")
