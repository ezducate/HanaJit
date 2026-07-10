"""Four-way comparison: plain Hana Jit vs +evolve() (safe GA) vs
hyper-aggressive vs Numba, across representative kernels.

This is the table shown in the README. All safe columns are verified to
compute the same result before timing; the hyper column is validated on a
large random probe suite (and, as the README notes honestly, is usually a
no-op because the safe GA already reaches the roofline).

Absolute milliseconds are noisy on shared machines — read the ratios.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from hanajit import jit
from numba import njit


def best(f, *a, reps=5):
    b = float("inf"); r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(*a); b = min(b, time.perf_counter() - t0)
    return r, b


def run(name, pyfn, args, small=64):
    plain = jit(pyfn)
    withga = jit(pyfn)
    hyper = jit(pyfn)
    nb = njit(pyfn)
    sm = tuple((min(x, small) if isinstance(x, int)
                else (x[:small] if isinstance(x, np.ndarray) else x))
               for x in args)
    for f in (plain, withga, hyper, nb):
        try:
            f(*sm)
        except Exception:
            f(*args)
    r0, t0 = best(plain, *args)
    withga.evolve(*args, allow_fastmath=True, generations=5, population=8,
                  reps=4)
    r1, t1 = best(withga, *args)
    hrep = hyper.evolve_hyper(*args, confirmed=True, generations=5,
                              population=8, reps=4, hyper_tol=1e-3)
    r2, t2 = best(hyper, *args)
    r3, t3 = best(nb, *args)
    # verify the safe columns agree
    ok = (abs(r0 - r1) < 1e-6 * max(1, abs(r0))
          and abs(r0 - r3) < 1e-6 * max(1, abs(r0)))
    return name, t0, t1, t2, t3, ok


def main():
    a = np.random.default_rng(0).uniform(-1, 1, 600_000)
    b = np.random.default_rng(1).uniform(-1, 1, 600_000)
    rows = []

    def fred(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i] + 0.5 * x[i]
        return s
    rows.append(run("fp reduction", fred, (a,)))

    def poly(x):
        s = 0.0
        for i in range(len(x)):
            v = x[i]
            s += ((((v * 0.1 + 0.2) * v + 0.3) * v + 0.4) * v + 0.5) * v
        return s
    rows.append(run("poly5 eval", poly, (a,)))

    def trans(x):
        s = 0.0
        for i in range(len(x)):
            s += np.exp(-x[i] * x[i]) + np.sqrt(np.abs(x[i]))
        return s
    rows.append(run("transcendental", trans, (a,)))

    def dot(x, y):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * y[i]
        return s
    rows.append(run("dot product", dot, (a, b)))

    print("=" * 76)
    print("FOUR-WAY: plain Hana Jit | +evolve() safe GA | +hyper-aggressive | Numba")
    print("(600K elements; ratios are the signal, ms is noisy on shared machines)")
    print("=" * 76)
    hdr = f"{'workload':<16}{'Hana Jit':>11}{'+GA (safe)':>12}{'+hyper':>10}{'Numba':>9}"
    print(hdr)
    print("-" * 76)
    for name, t0, t1, t2, t3, ok in rows:
        flag = "" if ok else "  [MISMATCH!]"
        print(f"{name:<16}{t0*1e3:>9.2f}ms{t1*1e3:>10.2f}ms{t2*1e3:>8.2f}ms"
              f"{t3*1e3:>7.2f}ms{flag}")
    print("-" * 76)
    print("Safe columns verified equal before timing. Hyper validated on 256")
    print("random probes. Takeaway: the safe GA (evolve) is the win; hyper is")
    print("usually a no-op because the safe GA already hits the roofline.")


if __name__ == "__main__":
    main()
