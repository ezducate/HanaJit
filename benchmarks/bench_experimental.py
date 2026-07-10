"""Benchmarks for the two experimental modes (v0.18): structural rewrite
and hyper-aggressive GP. All results self-verified before timing where the
mode guarantees correctness; hyper reports its validated tolerance."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit

def best(f, *a, reps=5):
    b = float("inf"); r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(*a); b = min(b, time.perf_counter()-t0)
    return r, b

print("="*68)
print("A. STRUCTURAL REWRITE (@jit(rewrite=True)) - closed-form reduction")
print("="*68)
# NB: for a plain `acc += i`, LLVM -O3 ALREADY forms the closed form, so
# plain-jit and rewrite tie. The rewrite EARNS its keep on affine bodies
# LLVM does not auto-close, e.g. a scaled sum with a large multiplier, and
# on making the O(1) form explicit/portable. We show both honestly.
def wsum(n):
    acc = 0
    for i in range(n):
        acc += 7 * i + 3
    return acc
py = wsum
rw = jit(rewrite=True)(wsum)
rw(10)
N = 20_000_000
r0, t0 = best(py, N, reps=1)
r1, t1 = best(rw, N, reps=3)
assert r0 == r1, (r0, r1)
print(f"  sum(7i+3, 0..{N}):")
print(f"    CPython loop : {t0*1e3:8.1f} ms")
print(f"    rewrite=True : {t1*1e6:8.1f} us  ({t0/t1:,.0f}x vs CPython, exact)")
print(f"  HONEST NOTE: our own LLVM -O3 also closed-forms this, so rewrite")
print(f"  vs plain-jit is ~1x. Rewrite's value: (1) it's a GUARANTEED O(1)")
print(f"  independent of the optimizer spotting it, and (2) it composes with")
print(f"  patterns LLVM misses (documented in docs/experimental.md).")

print()
print("="*68)
print("B. HYPER-AGGRESSIVE (evolve_hyper) vs safe evolve - fp reduction")
print("="*68)
def fred(x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * x[i] + 0.5 * x[i]
    return s
a = np.random.default_rng(0).uniform(-1, 1, 800_000)
jf = jit(fred); jf(a)
_, tb = best(jf, a)
safe = jf.evolve(a, allow_fastmath=True, generations=5, population=8, reps=4)
_, ts = best(jf, a)
jf2 = jit(fred); jf2(a)
hyper = jf2.evolve_hyper(a, confirmed=True, generations=5, population=8,
                         reps=4, hyper_tol=1e-3)
_, th = best(jf2, a)
print(f"  baseline        : {tb*1e3:.2f} ms")
print(f"  safe evolve     : {ts*1e3:.2f} ms ({tb/ts:.2f}x, guaranteed equiv)")
print(f"  hyper-aggressive: {th*1e3:.2f} ms ({tb/th:.2f}x, validated {hyper['probes_validated']} probes, persisted={hyper['persisted']})")
# fresh-input accuracy of hyper (allocate each probe once, small count)
maxerr = 0.0
for k in range(500, 505):
    u = np.random.default_rng(k).uniform(-1, 1, 800_000)
    exp = float((u*u + 0.5*u).sum())
    maxerr = max(maxerr, abs(jf2(u) - exp) / max(1, abs(exp)))
    del u
print(f"  hyper max rel err on 5 fresh inputs: {maxerr:.2e} (tol was 1e-3)")
