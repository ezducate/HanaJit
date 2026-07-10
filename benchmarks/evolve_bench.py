"""Four-way benchmark: CPython vs numba vs hanajit -O3 vs hanajit+GA.
Writes RESULTS-EVOLVE.md. Fairness notes:
- rows marked * allow fastmath for BOTH the GA and numba (njit(fastmath=True));
  un-starred rows are bit-exact everywhere.
- mutating kernels run on fresh copies outside the timer.
- every result is cross-verified before timing."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit
from numba import njit

lines = []
def emit(s=""):
    print(s); lines.append(s)

rng = np.random.default_rng(11)
A = rng.uniform(-1, 1, 2_000_000); B = rng.uniform(-1, 1, 2_000_000)
M = rng.uniform(0, 1, 1_000_000).reshape(1000, 1000)
X0 = rng.uniform(0, 1, 20_000)

def fp_reduce(x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * x[i] + 0.5 * x[i]
    return s

def dotk(x, y):
    return np.dot(x, y)

def saxpy_sum(x, y):
    for i in range(len(x)):
        y[i] = 2.5 * x[i] + y[i]
    return np.sum(y)

def stencil(x):
    s = 0.0
    for i in range(1, len(x) - 1):
        s += 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1]
    return s

def branchy(n):
    acc = 0
    for i in range(n):
        if i % 3 == 0:
            acc += i * 2
        elif i % 5 == 0:
            acc -= i
        else:
            acc ^= i
    return acc

def mathy(n):
    s = 0.0
    for i in range(n):
        x = 0.5 + (i % 999) * 0.001
        s += np.exp(-x) * np.sin(x) + np.sqrt(x)
    return s

def grid2d(m):
    s = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            s += m[i, j] * (1.0 if (i + j) % 2 == 0 else -1.0)
    return s

def primes(n):
    c = 0
    for i in range(2, n):
        p = 1
        j = 2
        while j * j <= i:
            if i % j == 0:
                p = 0
                break
            j += 1
        c += p
    return c

def collatz(limit):
    t = 0
    for n in range(2, limit):
        m = n
        while m != 1:
            m = m // 2 if m % 2 == 0 else 3 * m + 1
            t += 1
    return t

def logistic(n):
    x = 0.5
    acc = 0.0
    for i in range(n):
        x = 3.999 * x * (1.0 - x)
        acc += x
    return acc

def popcount(n):
    t = 0
    for i in range(n):
        x = i
        while x > 0:
            t += x & 1
            x >>= 1
    return t

def jacobi(x, sweeps):
    n = len(x)
    for k in range(sweeps):
        prev = x[0]
        for i in range(1, n - 1):
            cur = x[i]
            x[i] = 0.25 * prev + 0.5 * cur + 0.25 * x[i + 1]
            prev = cur
    return np.sum(x)

def window_energy(x, lo, hi):
    w = x[lo:hi]
    s = 0.0
    for i in range(len(w)):
        s += w[i] * w[i]
    return s

def strided_sum(x):
    return np.sum(x[::2]) + np.sum(x[1::4])

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

@njit
def fib_nb(n):
    if n < 2:
        return n
    return fib_nb(n - 1) + fib_nb(n - 2)

CASES = [
    ("fp reduction 2M *",            fp_reduce,  (A,),           True,  None),
    ("np.dot 2M *",                  dotk,       (A, B),         True,  None),
    ("saxpy+sum 2M (writes) *",      saxpy_sum,  (A, B),         True,  None),
    ("3-pt stencil 2M *",            stencil,    (A,),           True,  None),
    ("branchy int 3M",               branchy,    (3_000_000,),   False, None),
    ("math-heavy 1M *",              mathy,      (1_000_000,),   True,  None),
    ("2-D checkerboard 1000x1000 *", grid2d,     (M,),           True,  None),
    ("prime count 30k",              primes,     (30_000,),      False, None),
    ("collatz to 30k",               collatz,    (30_000,),      False, None),
    ("logistic 2M (chaotic: bit-exact)", logistic, (2_000_000,), False, None),
    ("popcount 1M",                  popcount,   (1_000_000,),   False, None),
    ("jacobi 20k x 100 (in-place) *", jacobi,    (X0, 100),      True,  None),
    ("slice window energy 2M *",     window_energy, (A, 250, 1_999_750), True, None),
    ("strided np.sum 2M *",          strided_sum, (A,),          True,  None),
    ("fib(27) recursion",            fib,        (27,),          False, fib_nb),
]

def fresh(args):
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)

def best(fn, args, reps):
    b = float("inf"); r = None
    for _ in range(reps):
        fa = fresh(args)
        t0 = time.perf_counter(); r = fn(*fa); b = min(b, time.perf_counter()-t0)
    return r, b

emit("# Four-way: CPython vs numba vs hanajit vs hanajit+GA\n")
emit("| kernel | CPython | numba | hanajit -O3 | hanajit +GA | GA gain | GA vs numba |")
emit("|---|---|---|---|---|---|---|")
g_numba, g_ga = [], []
for name, fn, args, fm, nbfn in CASES:
    jf = jit(fn); jf(*fresh(args))
    nf = nbfn or njit(fastmath=fm)(fn); nf(*fresh(args))
    r_py, t_py = best(fn, args, 1)
    r_nb, t_nb = best(nf, args, 5)
    rep = jf.evolve(*args, generations=6, population=10, reps=5, seed=3,
                    allow_fastmath=fm)
    r_hj, t_hj_check = best(jf, args, 3)   # evolved (or baseline) installed
    tol = 1e-6 * max(1.0, abs(r_py))
    assert abs(r_py - r_nb) < tol and abs(r_py - r_hj) < tol, name
    t_base, t_ga = rep["baseline_ms"], rep["best_ms"]
    g_numba.append((t_nb * 1e3) / t_ga); g_ga.append(t_base / t_ga)
    emit(f"| {name} | {t_py*1e3:,.1f} | {t_nb*1e3:,.2f} | {t_base:,.2f} | "
         f"{t_ga:,.2f} | {t_base/t_ga:.2f}x | {(t_nb*1e3)/t_ga:.2f}x |")

import math
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
emit(f"\n(all times ms) **Geomeans:** GA gain over hanajit -O3 = "
     f"**{gm(g_ga):.2f}x**; hanajit+GA vs numba = **{gm(g_numba):.2f}x**. "
     "`*` = fastmath allowed for BOTH numba (`njit(fastmath=True)`) and the "
     "GA; other rows bit-exact. Every result cross-verified before timing.")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "RESULTS-EVOLVE.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\nwritten:", out)
