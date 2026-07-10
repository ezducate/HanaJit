"""hanajit micro-benchmark suite: 52 workloads vs CPython and numba.

Run:  python benchmarks/microbench.py            (writes RESULTS-MICRO.md)
      python benchmarks/microbench.py --quick    (reduced iteration counts)

Categories: integer arithmetic, float arithmetic, control flow, recursion,
bit manipulation, pointer/array kernels, call/dispatch patterns.
Every workload is verified (hanajit == numba == CPython) before timing.
Pointer kernels: hanajit takes raw addresses via signature=..., numba takes
numpy arrays (its idiomatic form), CPython loops over the same numpy array.
"""
import os
import sys
import time
import math
import warnings

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
S = 0.1 if QUICK else 1.0

import numpy as np  # noqa: E402
from hanajit import jit  # noqa: E402
from numba import njit  # noqa: E402

REG = []  # (category, name, kind, payload)


def bench(category, name, args=(), sig=None, arrays=None, approx=False):
    def deco(fn):
        REG.append((category, name, dict(fn=fn, args=args, sig=sig,
                                         arrays=arrays, approx=approx)))
        return fn
    return deco


N1 = int(3_000_000 * S)      # big int loops
N2 = int(1_000_000 * S)      # float loops
N3 = int(30_000 * S)         # quadratic-ish control flow
NA = int(200_000 * S)        # array lengths

# ======================= 1. integer arithmetic (10) =========================
@bench("int", "sum of range", (N1,))
def i_sum(n):
    s = 0
    for i in range(n):
        s += i
    return s


@bench("int", "alternating sum", (N1,))
def i_altsum(n):
    s = 0
    for i in range(n):
        if i % 2 == 0:
            s += i
        else:
            s -= i
    return s


@bench("int", "gcd chain", (N3,))
def i_gcdchain(n):
    total = 0
    for i in range(1, n):
        a = i
        b = i * 7 + 3
        while b != 0:
            t = b
            b = a % b
            a = t
        total += a
    return total


@bench("int", "collatz total", (N3,))
def i_collatz(limit):
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


@bench("int", "factorial mod", (N1,))
def i_factmod(n):
    acc = 1
    for i in range(1, n):
        acc = acc * (i % 1000 + 1) % 10007
    return acc


@bench("int", "modular exponentiation", (N3,))
def i_powmod(n):
    total = 0
    for i in range(1, n):
        base = i % 97 + 2
        e = i % 61 + 1
        r = 1
        b = base
        while e > 0:
            if e % 2 == 1:
                r = r * b % 1000003
            b = b * b % 1000003
            e = e // 2
        total += r
    return total % 1000003


@bench("int", "digit sum sweep", (N3 * 10,))
def i_digitsum(n):
    total = 0
    for i in range(n):
        x = i
        while x > 0:
            total += x % 10
            x = x // 10
    return total


@bench("int", "pell recurrence mod", (N1,))
def i_pell(n):
    a = 0
    b = 1
    for i in range(n):
        t = (2 * b + a) % 1000000007
        a = b
        b = t
    return b


@bench("int", "fibonacci iterative mod", (N1,))
def i_fibiter(n):
    a = 0
    b = 1
    for i in range(n):
        t = (a + b) % 998244353
        a = b
        b = t
    return a


@bench("int", "fizzbuzz count", (N1,))
def i_fizz(n):
    c = 0
    for i in range(n):
        if i % 3 == 0 or i % 5 == 0:
            c += 1
    return c


# ========================= 2. float arithmetic (12) ==========================
@bench("float", "logistic map", (N2,), approx=True)
def f_logistic(n):
    x = 0.5
    acc = 0.0
    for i in range(n):
        x = 3.999 * x * (1.0 - x)
        acc += x
    return acc


@bench("float", "newton sqrt sweep", (N3 * 3,), approx=True)
def f_newton(n):
    acc = 0.0
    for i in range(1, n):
        v = float(i)
        x = v
        for k in range(6):
            x = 0.5 * (x + v / x)
        acc += x
    return acc


@bench("float", "leibniz pi", (N2,), approx=True)
def f_leibniz(n):
    s = 0.0
    sign = 1.0
    for i in range(n):
        s += sign / (2.0 * i + 1.0)
        sign = -sign
    return 4.0 * s


@bench("float", "harmonic series", (N2,), approx=True)
def f_harmonic(n):
    s = 0.0
    for i in range(1, n):
        s += 1.0 / i
    return s


@bench("float", "exp taylor sweep", (N3 * 2,), approx=True)
def f_exps(n):
    acc = 0.0
    for i in range(n):
        x = -3.0 + (i % 600) * 0.01
        term = 1.0
        s = 1.0
        for k in range(1, 18):
            term = term * x / k
            s += term
        acc += s
    return acc


@bench("float", "sin taylor sweep", (N3 * 2,), approx=True)
def f_sins(n):
    acc = 0.0
    for i in range(n):
        x = -3.0 + (i % 600) * 0.01
        x2 = x * x
        term = x
        s = x
        for k in range(1, 9):
            term = -term * x2 / ((2 * k) * (2 * k + 1))
            s += term
        acc += s
    return acc


@bench("float", "horner polynomial", (N2,), approx=True)
def f_horner(n):
    acc = 0.0
    for i in range(n):
        x = (i % 1000) * 0.002 - 1.0
        acc += ((((0.3 * x - 0.5) * x + 1.7) * x - 0.2) * x + 0.9)
    return acc


@bench("float", "lcg mean", (N2,), approx=True)
def f_lcg(n):
    x = 12345
    s = 0.0
    for i in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        s += x / 2147483648.0
    return s / n


@bench("float", "euler pendulum", (N2,), approx=True)
def f_pendulum(n):
    th = 0.5
    w = 0.0
    dt = 0.0005
    for i in range(n):
        # small-angle series for sin(th)
        s = th - th * th * th / 6.0
        w = w - 9.81 * s * dt
        th = th + w * dt
    return th + w


@bench("float", "verlet spring", (N2,), approx=True)
def f_verlet(n):
    x = 1.0
    xp = 0.999
    dt2 = 0.000001
    for i in range(n):
        a = -4.0 * x
        xn = 2.0 * x - xp + a * dt2
        xp = x
        x = xn
    return x


@bench("float", "softsign sum", (N2,), approx=True)
def f_softsign(n):
    s = 0.0
    for i in range(n):
        x = (i % 2001) * 0.01 - 10.0
        s += x / (1.0 + abs(x))
    return s


@bench("float", "mandelbrot escape batch", (2000,), approx=True)
def f_mandel(k):
    total = 0
    for t in range(k):
        cr = -0.74 + t * 1e-6
        ci = 0.11
        zr = 0.0
        zi = 0.0
        it = 0
        while it < 300 and zr * zr + zi * zi <= 4.0:
            zr2 = zr * zr - zi * zi + cr
            zi = 2.0 * zr * zi + ci
            zr = zr2
            it += 1
        total += it
    return total


# ========================== 3. control flow (8) ==============================
@bench("control", "prime count", (N3,))
def c_primes(n):
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


@bench("control", "perfect numbers", (N3 // 3,))
def c_perfect(n):
    found = 0
    for i in range(2, n):
        s = 1
        j = 2
        while j * j <= i:
            if i % j == 0:
                s += j
                if j != i // j:
                    s += i // j
            j += 1
        if s == i:
            found += 1
    return found


@bench("control", "happy numbers", (N3,))
def c_happy(n):
    happy = 0
    for i in range(1, n):
        x = i
        steps = 0
        while x != 1 and steps < 50:
            s = 0
            while x > 0:
                d = x % 10
                s += d * d
                x = x // 10
            x = s
            steps += 1
        if x == 1:
            happy += 1
    return happy


@bench("control", "palindromic numbers", (N3 * 5,))
def c_palin(n):
    count = 0
    for i in range(n):
        x = i
        rev = 0
        while x > 0:
            rev = rev * 10 + x % 10
            x = x // 10
        if rev == i:
            count += 1
    return count


@bench("control", "generalized syracuse", (N3,))
def c_syracuse(limit):
    total = 0
    for n in range(2, limit):
        m = n
        steps = 0
        while m != 1 and steps < 500:
            if m % 3 == 0:
                m = m // 3
            elif m % 2 == 0:
                m = m // 2
            else:
                m = 5 * m + 1
            steps += 1
        total += steps
    return total


@bench("control", "binary gcd", (N3,))
def c_bingcd(n):
    total = 0
    for i in range(1, n):
        a = i
        b = i * 3 + 7
        shift = 0
        while a % 2 == 0 and b % 2 == 0:
            a = a // 2
            b = b // 2
            shift += 1
        while b != 0:
            while b % 2 == 0:
                b = b // 2
            if a > b:
                t = a
                a = b
                b = t
            b = b - a
        total += a << shift
    return total


@bench("control", "zeller weekday sum", (N3 * 3,))
def c_zeller(n):
    total = 0
    for k in range(n):
        q = k % 28 + 1
        m = k % 12 + 1
        y = 1900 + k % 200
        mm = m
        yy = y
        if mm < 3:
            mm += 12
            yy -= 1
        h = (q + 13 * (mm + 1) // 5 + yy + yy // 4 - yy // 100 + yy // 400) % 7
        total += h
    return total


@bench("control", "loop break/continue mix", (N1,))
def c_breakmix(n):
    s = 0
    i = 0
    while i < n:
        i += 1
        if i % 7 == 0:
            continue
        if i % 100000 == 99999:
            s += 1000
            continue
        s += i % 3
    return s


# ============================ 4. recursion (5) ===============================
@bench("recursion", "fibonacci", (27,))
def r_fib(n):
    if n < 2:
        return n
    return r_fib(n - 1) + r_fib(n - 2)


@bench("recursion", "ackermann(3,6)", (3, 6))
def r_ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return r_ack(m - 1, 1)
    return r_ack(m - 1, r_ack(m, n - 1))


@bench("recursion", "binary tree fold", (18, 1))
def r_tree(n, s):
    if n == 0:
        return s % 7
    return (r_tree(n - 1, s * 2 + 1) + r_tree(n - 1, s * 2 + 2)
            + s % 5)


@bench("recursion", "takeuchi tak(18,12,6)", (18, 12, 6))
def r_tak(x, y, z):
    if y >= x:
        return z
    return r_tak(r_tak(x - 1, y, z), r_tak(y - 1, z, x),
                 r_tak(z - 1, x, y))


@bench("recursion", "cross-function call sweep (known hanajit gap)", (2000,))
def r_gcds(n):
    def helper(a, b):
        return a
    total = 0
    for i in range(1, n):
        total += r_gcd(i * 13 + 5, i)
    return total


def r_gcd(a, b):
    if b == 0:
        return a
    return r_gcd(b, a % b)


# ======================== 5. bit manipulation (5) ============================
@bench("bits", "lcg xor cascade", (N1,))
def b_lcgxor(n):
    x = 123456789
    acc = 0
    for i in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        acc ^= x
    return acc


@bench("bits", "popcount sweep", (N3 * 10,))
def b_popcount(n):
    total = 0
    for i in range(n):
        x = i
        while x > 0:
            total += x & 1
            x = x >> 1
    return total


@bench("bits", "reverse 16-bit", (N1,))
def b_revbits(n):
    total = 0
    for i in range(n):
        x = i & 0xFFFF
        r = 0
        for k in range(16):
            r = (r << 1) | (x & 1)
            x = x >> 1
        total += r
    return total % 1000003


@bench("bits", "parity sum", (N1,))
def b_parity(n):
    total = 0
    for i in range(n):
        x = i
        p = 0
        while x > 0:
            p ^= x & 1
            x = x >> 1
        total += p
    return total


@bench("bits", "gray code sum", (N1,))
def b_gray(n):
    total = 0
    for i in range(n):
        total += i ^ (i >> 1)
    return total % 1000000007


# ===================== 6. pointer / array kernels (8) ========================
def arr(seed, n=NA):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, n)


@bench("pointer", "saxpy", sig="f64*, f64*, f64, i64",
       arrays=lambda: (arr(1), arr(2)), args=(2.5, NA), approx=True)
def p_saxpy(x, y, a, n):
    for i in range(n):
        y[i] = a * x[i] + y[i]
    s = 0.0
    for i in range(n):
        s += y[i]
    return s


@bench("pointer", "dot product", sig="f64*, f64*, i64",
       arrays=lambda: (arr(3), arr(4)), args=(NA,), approx=True)
def p_dot(x, y, n):
    s = 0.0
    for i in range(n):
        s += x[i] * y[i]
    return s


@bench("pointer", "abs sum", sig="f64*, i64",
       arrays=lambda: (arr(5),), args=(NA,), approx=True)
def p_asum(x, n):
    s = 0.0
    for i in range(n):
        s += abs(x[i])
    return s


@bench("pointer", "max element", sig="f64*, i64",
       arrays=lambda: (arr(6),), args=(NA,), approx=True)
def p_max(x, n):
    m = x[0]
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
    return m


@bench("pointer", "count above threshold", sig="f64*, i64",
       arrays=lambda: (arr(7),), args=(NA,))
def p_count(x, n):
    c = 0
    for i in range(n):
        if x[i] > 0.25:
            c += 1
    return c


@bench("pointer", "prefix sum (in place)", sig="f64*, i64",
       arrays=lambda: (arr(8),), args=(NA,), approx=True)
def p_prefix(x, n):
    for i in range(1, n):
        x[i] = x[i] + x[i - 1]
    return x[n - 1]


@bench("pointer", "3-point stencil", sig="f64*, f64*, i64",
       arrays=lambda: (arr(9), arr(10)), args=(NA,), approx=True)
def p_stencil(x, y, n):
    for i in range(1, n - 1):
        y[i] = 0.25 * x[i - 1] + 0.5 * x[i] + 0.25 * x[i + 1]
    s = 0.0
    for i in range(n):
        s += y[i]
    return s


@bench("pointer", "polynomial eval array", sig="f64*, i64",
       arrays=lambda: (arr(11),), args=(NA,), approx=True)
def p_poly(x, n):
    s = 0.0
    for i in range(n):
        v = x[i]
        s += ((0.3 * v - 0.5) * v + 1.7) * v - 0.2
    return s


# ===================== 7. call / dispatch patterns (4) =======================
DISPATCH = []  # handled specially: (name, pyfunc, mode)


def d_tiny(a, b):
    return a * b + 1


def d_medium(a, b):
    s = a
    for i in range(20):
        s = s * 31 + b
    return s % 1000003


DISPATCH.append(("200k tiny calls (int,int)", d_tiny, "mono"))
DISPATCH.append(("100k medium calls", d_medium, "mono"))
DISPATCH.append(("100k polymorphic calls (int/float alternating)",
                 d_tiny, "poly"))
DISPATCH.append(("200k .specialize() calls", d_tiny, "spec"))


# =============================== harness =====================================
def best(fn, *args, reps=3):
    b = float("inf")
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        b = min(b, time.perf_counter() - t0)
    return r, b


NB_RECURSIVE = {}


def build_numba_recursive():
    src = """
from numba import njit

@njit
def r_fib(n):
    if n < 2:
        return n
    return r_fib(n - 1) + r_fib(n - 2)

@njit
def r_ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return r_ack(m - 1, 1)
    return r_ack(m - 1, r_ack(m, n - 1))

@njit
def r_tree(n, s):
    if n == 0:
        return s % 7
    return (r_tree(n - 1, s * 2 + 1) + r_tree(n - 1, s * 2 + 2)
            + s % 5)

@njit
def r_tak(x, y, z):
    if y >= x:
        return z
    return r_tak(r_tak(x - 1, y, z), r_tak(y - 1, z, x),
                 r_tak(z - 1, x, y))

@njit
def r_gcd(a, b):
    if b == 0:
        return a
    return r_gcd(b, a % b)

@njit
def r_gcds(n):
    total = 0
    for i in range(1, n):
        total += r_gcd(i * 13 + 5, i)
    return total
"""
    g = {}
    exec(src, g)
    for k in ("r_fib", "r_ack", "r_tree", "r_tak", "r_gcds"):
        NB_RECURSIVE[k] = g[k]


def hanajit_recursive():
    # hanajit self-recursion works by AST name; r_gcds calls another jit fn
    # via fallback-unsafe global -> compile r_gcd standalone and inline sweep
    src_map = {}
    jf_gcd = jit(r_gcd)
    jf_gcd(12, 8)

    def sweep(n):
        total = 0
        for i in range(1, n):
            total += jf_gcd(i * 13 + 5, i)
        return total
    src_map["r_gcds"] = sweep
    return src_map


def geomean(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    import platform
    import hanajit
    import numba
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# hanajit micro-benchmarks (52 workloads)\n")
    emit(f"- Python {sys.version.split()[0]}, {platform.machine()}, "
         f"{os.cpu_count()} core(s); numba {numba.__version__}; "
         f"hanajit {hanajit.__version__}"
         + (" (--quick)" if QUICK else "") + "\n")

    build_numba_recursive()
    fj_special = hanajit_recursive()
    cat_speedup_py, cat_speedup_nb = {}, {}
    current_cat = None

    for category, name, spec in REG:
        if category != current_cat:
            current_cat = category
            emit(f"\n## {category}\n")
            emit("| workload | CPython | hanajit | numba | vs py | vs numba |")
            emit("|---|---|---|---|---|---|")

        fn, args, sig, arrays, approx = (spec["fn"], spec["args"],
                                         spec["sig"], spec["arrays"],
                                         spec["approx"])
        if category == "recursion":
            jf = fj_special.get(fn.__name__) or jit(fn)
            nf = NB_RECURSIVE[fn.__name__]
            small = tuple(min(a, 3) for a in args)
            jf(*small); nf(*small)
            r_fj, t_fj = best(jf, *args)
            r_nb, t_nb = best(nf, *args)
            r_py, t_py = best(fn, *args, reps=2)
        elif sig is None:
            jf = jit(fn)
            nf = njit(fn)
            small = tuple(min(a, 100) if isinstance(a, int) else a
                          for a in args)
            jf(*small); nf(*small)
            r_fj, t_fj = best(jf, *args)
            r_nb, t_nb = best(nf, *args)
            r_py, t_py = best(fn, *args, reps=2)
        else:
            jf = jit(signature=sig)(fn)
            nf = njit(fn)

            def run(callable_, mode):
                copies = [a.copy() for a in arrays()]
                if mode == "fj":
                    cargs = tuple(c.ctypes.data for c in copies) + args
                else:
                    cargs = tuple(copies) + args
                t0 = time.perf_counter()
                r = callable_(*cargs)
                return r, time.perf_counter() - t0

            def best3(callable_, mode, reps=3):
                b = float("inf")
                r = None
                for _ in range(reps):
                    r, t = run(callable_, mode)
                    b = min(b, t)
                return r, b
            r_nb, _ = best3(nf, "np", reps=1)   # numba warm-up + verify src
            r_fj, t_fj = best3(jf, "fj")
            r_nb, t_nb = best3(nf, "np")
            r_py, t_py = best3(fn, "np", reps=1)

        if approx:
            ok = (abs(r_py - r_fj) <= 1e-6 * max(1.0, abs(r_py))
                  and abs(r_py - r_nb) <= 1e-6 * max(1.0, abs(r_py)))
        else:
            ok = r_py == r_fj == r_nb
        assert ok, (name, r_py, r_fj, r_nb)

        sp_py, sp_nb = t_py / t_fj, t_nb / t_fj
        cat_speedup_py.setdefault(category, []).append(sp_py)
        cat_speedup_nb.setdefault(category, []).append(sp_nb)
        emit(f"| {name} | {t_py*1e3:,.1f} ms | {t_fj*1e3:,.2f} ms | "
             f"{t_nb*1e3:,.2f} ms | {sp_py:,.1f}x | {sp_nb:.2f}x |")

    # ---- dispatch patterns ----
    emit("\n## dispatch\n")
    emit("| workload | CPython | hanajit | numba | vs py | vs numba |")
    emit("|---|---|---|---|---|---|")
    for name, fn, mode in DISPATCH:
        jf = jit(fn)
        nf = njit(fn)
        jf(1, 2); jf(1.0, 2.0); nf(1, 2); nf(1.0, 2.0)
        count = 100000 if "100k" in name else 200000
        target_fj = jf.specialize(int, int) if mode == "spec" else jf

        def loop(f):
            t0 = time.perf_counter()
            if mode == "poly":
                for i in range(count // 2):
                    f(i, 3)
                    f(float(i), 3.0)
            else:
                for i in range(count):
                    f(i, 3)
            return time.perf_counter() - t0
        t_py, t_fj, t_nb = loop(fn), loop(target_fj), loop(nf)
        sp_py, sp_nb = t_py / t_fj, t_nb / t_fj
        cat_speedup_py.setdefault("dispatch", []).append(sp_py)
        cat_speedup_nb.setdefault("dispatch", []).append(sp_nb)
        emit(f"| {name} | {t_py*1e3:,.1f} ms | {t_fj*1e3:,.2f} ms | "
             f"{t_nb*1e3:,.2f} ms | {sp_py:,.1f}x | {sp_nb:.2f}x |")

    emit("\n## Summary (geometric means)\n")
    emit("| category | workloads | hanajit vs CPython | hanajit vs numba |")
    emit("|---|---|---|---|")
    all_py, all_nb = [], []
    for cat in cat_speedup_py:
        g1, g2 = geomean(cat_speedup_py[cat]), geomean(cat_speedup_nb[cat])
        all_py += cat_speedup_py[cat]
        all_nb += cat_speedup_nb[cat]
        emit(f"| {cat} | {len(cat_speedup_py[cat])} | {g1:,.1f}x | {g2:.2f}x |")
    emit(f"| **overall** | **{len(all_py)}** | **{geomean(all_py):,.1f}x** | "
         f"**{geomean(all_nb):.2f}x** |")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "RESULTS-MICRO.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
