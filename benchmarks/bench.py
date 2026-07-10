"""hanajit benchmark suite: hanajit vs CPython vs numba.

Run:  python benchmarks/bench.py            (full run, writes RESULTS.md)
      python benchmarks/bench.py --quick    (smaller iteration counts)

Sections:
  1. Compute workloads (recursion, int/float loops, control flow)
  2. Dispatch overhead (generic call path + .specialize())
  3. Time-to-first-result: cold start, with and without disk caches
  4. GIL release under threading
  5. cProfile breakdown of the dispatch layers
"""
import cProfile
import io
import os
import pstats
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
SCALE = 0.1 if QUICK else 1.0

from hanajit import jit  # noqa: E402
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def best(fn, *args, reps=5):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        b = min(b, time.perf_counter() - t0)
    return r, b


# ---------------- workload definitions (pure Python source of truth) --------
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


if HAVE_NUMBA:
    @njit
    def fib_nb(n):
        if n < 2:
            return n
        return fib_nb(n - 1) + fib_nb(n - 2)


def int_loop(n):
    acc = 0
    for i in range(n):
        acc += i % 7 - i % 3
    return acc


def logistic(n):
    acc = 0.0
    x = 0.5
    for i in range(n):
        x = 3.999 * x * (1.0 - x)
        acc += x
    return acc


def mandel_point(cr, ci, maxit):
    zr = 0.0
    zi = 0.0
    it = 0
    while it < maxit and zr * zr + zi * zi <= 4.0:
        zr2 = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = zr2
        it += 1
    return it


def collatz_total(limit):
    total = 0
    for n in range(2, limit):
        m = n
        while m != 1:
            if m % 2 == 0:
                m = m // 2
            else:
                m = 3 * m + 1
            total += 1
    return total


def count_primes(n):
    count = 0
    for i in range(2, n):
        is_p = 1
        j = 2
        while j * j <= i:
            if i % j == 0:
                is_p = 0
                break
            j += 1
        count += is_p
    return count


def run_workloads():
    emit("## 1. Compute workloads\n")
    emit("| workload | CPython | hanajit | numba | fj vs py | fj vs numba |")
    emit("|---|---|---|---|---|---|")
    N = int(20_000_000 * SCALE)
    cases = [
        ("fib(%d) recursion" % (28 if QUICK else 30), fib,
         (28 if QUICK else 30,)),
        ("int loop %dM (mod/branch)" % (N // 10**6), int_loop, (N,)),
        ("logistic map %dM (float)" % (N // 10**6), logistic, (N,)),
        ("mandelbrot point x2000", None, None),   # special-cased below
        ("collatz total to %d" % int(30000 * SCALE), collatz_total,
         (int(30000 * SCALE),)),
        ("prime count to %d" % int(30000 * SCALE), count_primes,
         (int(30000 * SCALE),)),
    ]
    for name, pyfn, args in cases:
        if pyfn is None:
            def batch(fn):
                def run():
                    s = 0
                    for k in range(2000):
                        s += fn(-0.74 + k * 1e-6, 0.11, 300)
                    return s
                return run
            jf = jit(mandel_point); jf(0.0, 0.0, 2)
            r_py, t_py = best(batch(mandel_point))
            r_fj, t_fj = best(batch(jf))
            if HAVE_NUMBA:
                nf = njit(mandel_point); nf(0.0, 0.0, 2)
                r_nb, t_nb = best(batch(nf))
            else:
                r_nb = r_fj; t_nb = float("nan")
        else:
            jf = jit(pyfn)
            small = tuple(min(a, 100) if isinstance(a, int) else a
                          for a in args)
            jf(*small)
            r_fj, t_fj = best(jf, *args)
            r_py, t_py = best(pyfn, *args, reps=2)
            if HAVE_NUMBA:
                nf = fib_nb if pyfn is fib else njit(pyfn)
                nf(*small)
                r_nb, t_nb = best(nf, *args)
            else:
                r_nb = r_fj; t_nb = float("nan")
        assert r_py == r_fj == r_nb or abs(r_py - r_fj) < 1e-6, name
        emit(f"| {name} | {t_py*1e3:,.1f} ms | {t_fj*1e3:,.1f} ms | "
             f"{t_nb*1e3:,.1f} ms | {t_py/t_fj:,.1f}x | {t_nb/t_fj:.2f}x |")
    emit()


def run_dispatch():
    emit("## 2. Dispatch overhead (200k tiny calls)\n")

    def tiny(a, b):
        return a * b + 1
    jf = jit(tiny); jf(1, 2)
    fast = jf.specialize(int, int)
    nf = None
    if HAVE_NUMBA:
        nf = njit(tiny); nf(1, 2)

    def many(fn):
        t0 = time.perf_counter()
        for i in range(200000):
            fn(i, 3)
        return time.perf_counter() - t0

    rows = [("plain Python function", many(tiny)),
            ("hanajit generic (native vectorcall)", many(jf)),
            ("hanajit .specialize()", many(fast))]
    if nf:
        rows.append(("numba dispatcher", many(nf)))
    emit("| call path | 200k calls | per call |")
    emit("|---|---|---|")
    for name, t in rows:
        emit(f"| {name} | {t*1e3:.1f} ms | {t/200000*1e9:.0f} ns |")
    emit()


COLD_SRC = """
import sys, time, warnings; warnings.filterwarnings("ignore")
t_start = time.perf_counter()
MODE = sys.argv[1]
def kernel(n):
    acc = 0.0
    for i in range(n):
        acc += (i % 13) * 0.25
    return acc
if MODE == "python":
    fn = kernel
elif MODE.startswith("hanajit"):
    from hanajit import jit
    fn = jit(cache=("cache" in MODE))(kernel)
elif MODE.startswith("numba"):
    from numba import njit
    fn = njit(cache=("cache" in MODE))(kernel)
r = fn(1000)
print(f"{(time.perf_counter()-t_start)*1e3:.1f}")
"""


def run_cold_start():
    emit("## 3. Time to first result (fresh process: import + compile + run)\n")
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "cold.py")
        with open(script, "w") as f:
            f.write(COLD_SRC)
        env = dict(os.environ, HANAJIT_CACHE_DIR=os.path.join(td, "fj"))
        emit("| mode | time-to-first-result |")
        emit("|---|---|")
        modes = ["python", "hanajit", "hanajit-cache (cold)",
                 "hanajit-cache (warm)"]
        if HAVE_NUMBA:
            modes += ["numba", "numba-cache (cold)", "numba-cache (warm)"]
        for mode in modes:
            arg = mode.split(" ")[0]
            out = subprocess.run([sys.executable, script, arg], env=env,
                                 capture_output=True, text=True, cwd=td)
            emit(f"| {mode} | {out.stdout.strip()} ms |")
    emit()


def run_gil():
    emit("## 4. GIL release (main-thread iterations while a background "
         "kernel computes)\n")
    import threading

    def spin(n):
        acc = 0.0
        for i in range(n):
            acc += (i % 7) * 0.5
        return acc
    N = int(400_000_000 * SCALE)
    jg = jit(spin)
    jn = jit(nogil=True)(spin)
    jg(10); jn(10)
    emit("| kernel | main-thread ticks during kernel |")
    emit("|---|---|")
    for name, fn in [("@jit (GIL held)", jg), ("@jit(nogil=True)", jn)]:
        t = threading.Thread(target=fn, args=(N,)); t.start()
        ticks = 0
        while t.is_alive():
            ticks += 1
        t.join()
        emit(f"| {name} | {ticks:,} |")
    emit()


def run_profile():
    emit("## 5. cProfile: where dispatch time goes\n")

    def tiny(a, b):
        return a * b + 1
    jf_native = jit(tiny)
    jf_python = jit(native_dispatch=False)(tiny)
    jf_native(1, 2); jf_python(1, 2)

    for name, fn in [("native vectorcall dispatch", jf_native),
                     ("python Dispatcher fallback", jf_python)]:
        pr = cProfile.Profile()
        pr.enable()
        for i in range(100000):
            fn(i, 3)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(4)
        body = "\n".join(l for l in buf.getvalue().splitlines()
                         if l.strip())[:900]
        emit(f"### {name}\n```\n{body}\n```\n")
    emit("Native dispatch shows no Python-level frames per call — the "
         "interpreter never runs between the call site and the kernel.")


if __name__ == "__main__":
    import platform
    emit(f"# hanajit benchmark results\n")
    emit(f"- Python {sys.version.split()[0]}, {platform.machine()}, "
         f"{os.cpu_count()} core(s)")
    if HAVE_NUMBA:
        import numba
        emit(f"- numba {numba.__version__}")
    import hanajit
    emit(f"- hanajit {hanajit.__version__}"
         + (" (--quick)" if QUICK else "") + "\n")
    run_workloads()
    run_dispatch()
    run_cold_start()
    run_gil()
    run_profile()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {out}")
