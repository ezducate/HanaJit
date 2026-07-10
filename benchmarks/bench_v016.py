"""v0.16.0 benchmarks: the two new features plus the four-way baseline,
all self-verified (results asserted equal before timing)."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit
from numba import njit

def best(f, *a, reps=5):
    b = float("inf"); r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(*a); b = min(b, time.perf_counter()-t0)
    return r, b

print("="*70)
print("A. HELPER INLINING — cross-function calls")
print("="*70)

@jit
def sq(x): return x * x
@jit
def cube(x): return x * x * x

@jit
def via_helpers(x):
    s = 0.0
    for i in range(len(x)):
        s += sq(x[i]) + sq(x[i] + 1.0) + cube(x[i])
    return s

@jit
def hand_inlined(x):
    s = 0.0
    for i in range(len(x)):
        v = x[i]
        s += v*v + (v+1.0)*(v+1.0) + v*v*v
    return s

# numba: helpers are separate njit funcs (it inlines too — fair comparison)
@njit
def nsq(x): return x*x
@njit
def ncube(x): return x*x*x
@njit
def nb_helpers(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += nsq(x[i]) + nsq(x[i]+1.0) + ncube(x[i])
    return s

a = np.random.default_rng(0).uniform(-1, 1, 2_000_000)
for f in (via_helpers, hand_inlined, nb_helpers): f(a[:16])
rh, th = best(via_helpers, a)
ri, ti = best(hand_inlined, a)
rn, tn = best(nb_helpers, a)
assert abs(rh-ri) < 1e-6 and abs(rh-rn) < 1e-6
print(f"  hanajit via helpers : {th*1e3:6.2f} ms")
print(f"  hanajit hand-inlined: {ti*1e3:6.2f} ms   (helper overhead: {th/ti:.2f}x)")
print(f"  numba  via helpers  : {tn*1e3:6.2f} ms   (hanajit is {tn/th:.2f}x)")

print()
print("="*70)
print("B. AUTO-PARALLEL — @jit(parallel=True), no code change")
print("="*70)
import os
CORES = len(os.sched_getaffinity(0))

def heavy(n):
    acc = 0.0
    for i in range(n):
        x = i * 1e-7
        acc += (x*x + 0.3*x - 0.1)*(x - 0.2) + x*x*x
    return acc

ser = jit(heavy)
par = jit(parallel=True, workers=max(CORES,1))(heavy)
n = 8_000_000
ser(1000); par(1000)
rs, ts = best(ser, n, reps=3)
rp, tp = best(par, n, reps=3)
assert abs(rs-rp) < 1e-6 * max(1, abs(rs))
print(f"  cores available     : {CORES}")
print(f"  serial              : {ts*1e3:6.1f} ms")
print(f"  parallel=True       : {tp*1e3:6.1f} ms   (speedup: {ts/tp:.2f}x)")
if CORES == 1:
    print("  (single-core box — speedup shows on multi-core hardware; see doctor reports)")

print()
print("="*70)
print("C. FOUR-WAY with fusion + inlining together (CPython/numba/hanajit/+GA)")
print("="*70)
A = np.random.default_rng(1).uniform(-2, 2, 2_000_000)
B = np.random.default_rng(2).uniform(0.1, 2, 2_000_000)

def expr_py(a, b):
    return float(np.sum(np.exp(-a*a)*b + np.where(a>0, a, 2.0*a) - np.clip(b, .2, 1.5)))
@jit
def expr_hj(a, b):
    return np.sum(np.exp(-a*a)*b + np.where(a>0.0, a, 2.0*a) - np.clip(b, .2, 1.5))
expr_nb = njit(lambda a, b: np.sum(np.exp(-a*a)*b + np.where(a>0, a, 2.0*a) - np.clip(b, .2, 1.5)))
expr_hj(A[:16], B[:16]); expr_nb(A[:16], B[:16])
r0, t0 = best(expr_py, A, B, reps=3)
r1, t1 = best(expr_hj, A, B)
r2, t2 = best(expr_nb, A, B)
assert abs(r0-r1) < 1e-4 and abs(r0-r2) < 1e-4
rep = expr_hj.evolve(A, B, generations=5, population=10, reps=4, allow_fastmath=True)
r3, t3 = best(expr_hj, A, B)
assert abs(r0-r3) < 1e-2
print(f"  CPython+numpy       : {t0*1e3:6.1f} ms")
print(f"  numba               : {t2*1e3:6.1f} ms")
print(f"  hanajit (fused)     : {t1*1e3:6.1f} ms   ({t0/t1:.1f}x vs numpy, {t2/t1:.2f}x vs numba)")
print(f"  hanajit + GA        : {t3*1e3:6.1f} ms   (GA {t1/t3:.2f}x further)")
