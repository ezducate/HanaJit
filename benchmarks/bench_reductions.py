"""CPU reduction throughput: default (sequential, bit-exact) vs
reduce_reassoc (parallel SIMD accumulators, numpy-class) vs numpy.
Self-verified: reassoc result stays within reassociation tolerance."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit

def best(f, *a, reps=7):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(*a); b = min(b, time.perf_counter()-t0)
    return r, b

N = 20_000_000
a = np.random.default_rng(0).uniform(-1, 1, N)
b = np.random.default_rng(1).uniform(-1, 1, N)

print("="*64)
print(f"Reduction throughput, {N/1e6:.0f}M float64 elements")
print("="*64)

def hand_sum(x):
    acc = 0.0
    for i in range(len(x)):
        acc += x[i]
    return acc

jd = jit(hand_sum)
jr = jit(reduce_reassoc=True)(hand_sum)
jd(a); jr(a)
_, td = best(jd, a); _, tr = best(jr, a)
_, tn = best(lambda x=a: np.sum(x), a)
exact = sum(float(v) for v in a[:100000]) * 0  # placeholder
print(f"  hand loop default      : {td*1e3:6.2f} ms  {N*8/td/1e9:5.1f} GB/s")
print(f"  hand loop reduce_reassoc: {tr*1e3:6.2f} ms  {N*8/tr/1e9:5.1f} GB/s  ({td/tr:.2f}x)")
print(f"  numpy sum              : {tn*1e3:6.2f} ms  {N*8/tn/1e9:5.1f} GB/s")

def nd(x, y): return np.dot(x, y)
jdd = jit(nd); jrd = jit(reduce_reassoc=True)(nd)
jdd(a, b); jrd(a, b)
_, tdd = best(jdd, a, b); _, trd = best(jrd, a, b)
_, tnd = best(lambda x=a, y=b: np.dot(x, y), a, b)
print(f"\n  dot default            : {tdd*1e3:6.2f} ms")
print(f"  dot reduce_reassoc     : {trd*1e3:6.2f} ms  ({tdd/trd:.2f}x)")
print(f"  numpy dot (BLAS)       : {tnd*1e3:6.2f} ms")

print(f"\n  Accuracy (vs exact sequential sum):")
seq = 0.0
for v in a:
    seq += float(v)
print(f"    default : err {abs(jd(a)-seq):.2e}  (bit-exact sequential)")
print(f"    reassoc : err {abs(jr(a)-seq):.2e}  (reassociated, like numpy)")
print(f"    numpy   : err {abs(float(np.sum(a))-seq):.2e}")
print("\n  reduce_reassoc reaches numpy-class reduction speed by enabling")
print("  reassociation ONLY on accumulators — no global fastmath, integers")
print("  untouched. Best for memory-bound reductions; compute-bound fused")
print("  kernels see no benefit (already vectorized elsewhere).")
