"""NumPy/SciPy acceleration benchmarks. Writes RESULTS-NUMPY.md.

Honest framing: hanajit does not reimplement numpy's kernels. It wins where
numpy loses — chained expressions that allocate temporaries, and per-call
Python overhead in scipy callbacks. It loses to BLAS where BLAS applies,
and that row is included on purpose.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit
from numba import njit, cfunc, types as nbt
from scipy.integrate import quad
from scipy import LowLevelCallable

lines = []
def emit(s=""):
    print(s); lines.append(s)

def best(fn, *a, reps=5):
    b = float("inf"); r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(*a); b = min(b, time.perf_counter()-t0)
    return r, b

N = 2_000_000
rng = np.random.default_rng(7)
A = rng.uniform(-1, 1, N); B = rng.uniform(-1, 1, N)

emit("# NumPy / SciPy acceleration\n")
emit("| workload | numpy/scipy | hanajit | numba | vs numpy | vs numba |")
emit("|---|---|---|---|---|---|")

# 1. fused elementwise chain (numpy allocates 4 temporaries)
def np_chain(a, b):
    return float((a * b + 2.5 * a - b * b).sum())
@jit(nogil=True)
def hj_fused(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i] + 2.5 * a[i] - b[i] * b[i]
    return s
nb_fused = njit(lambda a, b: (a * b + 2.5 * a - b * b).sum())
def nb_loop_src(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i] + 2.5 * a[i] - b[i] * b[i]
    return s
nb_fusedloop = njit(nb_loop_src)
hj_fused(A[:10], B[:10], 10); nb_fusedloop(A[:10], B[:10], 10)
r1, t_np = best(np_chain, A, B)
r2, t_hj = best(hj_fused, A, B, N)
r3, t_nb = best(nb_fusedloop, A, B, N)
assert abs(r1-r2) < 1e-6 and abs(r1-r3) < 1e-6
emit(f"| fused elementwise+reduce, 2M (4 numpy temporaries) | {t_np*1e3:,.1f} ms | {t_hj*1e3:,.1f} ms | {t_nb*1e3:,.1f} ms | {t_np/t_hj:.1f}x | {t_nb/t_hj:.2f}x |")

# 2. dot product — BLAS should win; included for honesty
@jit
def hj_dot(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s
def nbd(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s
nb_dot = njit(nbd)
hj_dot(A[:10], B[:10], 10); nb_dot(A[:10], B[:10], 10)
r1, t_np = best(lambda: float(np.dot(A, B)))
r2, t_hj = best(hj_dot, A, B, N)
r3, t_nb = best(nb_dot, A, B, N)
assert abs(r1-r2) < 1e-5
emit(f"| dot product 2M (numpy = BLAS: expected numpy win) | {t_np*1e3:,.2f} ms | {t_hj*1e3:,.2f} ms | {t_nb*1e3:,.2f} ms | {t_np/t_hj:.2f}x | {t_nb/t_hj:.2f}x |")

# 3. iterative Jacobi smoothing: 200 sweeps (numpy version churns temporaries)
M = 20_000
X0 = rng.uniform(0, 1, M)
def np_jacobi(x0, sweeps):
    x = x0.copy()
    for _ in range(sweeps):
        x[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
    return float(x.sum())
@jit(nogil=True)
def hj_jacobi(x, n, sweeps):
    for k in range(sweeps):
        prev = x[0]
        for i in range(1, n - 1):
            cur = x[i]
            x[i] = 0.25 * prev + 0.5 * cur + 0.25 * x[i + 1]
            prev = cur
    s = 0.0
    for i in range(n):
        s += x[i]
    return s
def nbj(x, n, sweeps):
    for k in range(sweeps):
        prev = x[0]
        for i in range(1, n - 1):
            cur = x[i]
            x[i] = 0.25 * prev + 0.5 * cur + 0.25 * x[i + 1]
            prev = cur
    s = 0.0
    for i in range(n):
        s += x[i]
    return s
nb_jacobi = njit(nbj)
hj_jacobi(X0.copy(), M, 1); nb_jacobi(X0.copy(), M, 1)
r1, t_np = best(lambda: np_jacobi(X0, 200), reps=3)
r2, t_hj = best(lambda: hj_jacobi(X0.copy(), M, 200), reps=3)
r3, t_nb = best(lambda: nb_jacobi(X0.copy(), M, 200), reps=3)
assert abs(r1-r2) < 1e-6 * abs(r1) and abs(r1-r3) < 1e-6 * abs(r1)
emit(f"| Jacobi smoothing, 20k x 200 sweeps (in-place, seq. dependency) | {t_np*1e3:,.1f} ms | {t_hj*1e3:,.1f} ms | {t_nb*1e3:,.1f} ms | {t_np/t_hj:.1f}x | {t_nb/t_hj:.2f}x |")

# 4. scipy.integrate.quad: Python callback vs C-pointer callbacks
def py_gauss(x):
    return 2.718281828459045 ** (-x * x)
@jit
def hj_gauss(x):
    return 2.718281828459045 ** (-x * x)
llc_hj = hj_gauss.scipy_callable()
nb_cf = cfunc(nbt.float64(nbt.float64))(py_gauss)
llc_nb = LowLevelCallable(nb_cf.ctypes)
def q(f):
    return quad(f, -6, 6, limit=200, epsabs=1e-12, epsrel=1e-12)[0]
r1, t_py = best(q, py_gauss)
r2, t_hj = best(q, llc_hj)
r3, t_nb = best(q, llc_nb)
assert abs(r1-r2) < 1e-9 and abs(r1-r3) < 1e-9
emit(f"| scipy.quad gaussian (LowLevelCallable vs Python callback) | {t_py*1e3:,.2f} ms | {t_hj*1e3:,.2f} ms | {t_nb*1e3:,.2f} ms | {t_py/t_hj:.1f}x | {t_nb/t_hj:.2f}x |")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS-NUMPY.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\nwritten:", out)
