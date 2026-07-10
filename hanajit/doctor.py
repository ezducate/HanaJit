"""hanajit doctor — platform diagnostic & report generator.

Run on each machine:   python -m hanajit.doctor
Writes hanajit_report_<os>_<arch>.md with PASS/FAIL/SKIP per check,
environment details, timings, and full tracebacks for failures — paste the
file back for debugging. Risky sections (GPU emission, Metal/ptxas
toolchains, cache subprocesses) run in child processes so a hard abort
can't kill the report.
"""
import faulthandler
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
import traceback
import warnings

faulthandler.enable()
warnings.filterwarnings("ignore")

RESULTS = []          # (section, name, status, detail_ms_or_msg)
FAILURES = []         # (name, traceback_text)


def check(section, name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                msg = fn()
                dt = (time.perf_counter() - t0) * 1e3
                RESULTS.append((section, name, "PASS",
                                msg or f"{dt:.1f} ms"))
            except SkipCheck as e:
                RESULTS.append((section, name, "SKIP", str(e)))
            except Exception:
                RESULTS.append((section, name, "FAIL", "see traceback"))
                FAILURES.append((name, traceback.format_exc()))
        run.__name__ = name
        CHECKS.append(run)
        return run
    return deco


class SkipCheck(Exception):
    pass


CHECKS = []


def subproc(code, timeout=120):
    """Run a snippet in a child interpreter FROM A FILE (inspect.getsource
    needs real files — `-c` code would silently fall back to CPython)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n" + textwrap.dedent(code))
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, encoding="utf-8",
                           errors="replace")
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- checks
@check("core", "import + version")
def c_import():
    import hanajit
    import llvmlite
    return f"hanajit {hanajit.__version__}, llvmlite {llvmlite.__version__}"


@check("core", "scalar compile + run")
def c_scalar():
    from hanajit import jit

    @jit
    def f(a, b):
        acc = 0
        for i in range(a):
            acc += i % b
        return acc
    assert f(1000, 7) == sum(i % 7 for i in range(1000))
    assert len(f.cache) == 1, "fell back instead of compiling"
    return f"dispatch={type(f).__name__}"


@check("core", "native vectorcall dispatcher")
def c_dispatch():
    import struct
    from hanajit import jit

    @jit
    def g(a):
        return a * 2
    g(1)
    name = type(g).__name__
    expected_native = (sys.implementation.name == "cpython"
                       and sys.version_info >= (3, 12)
                       and struct.calcsize("P") == 8
                       and not sysconfig.get_config_var("Py_GIL_DISABLED"))
    if expected_native and name != "HanaFunction":
        raise AssertionError(f"expected native dispatch, got {name}")
    return f"{name} (native expected: {expected_native})"


@check("core", "recursion + math intrinsics")
def c_math():
    import math
    from hanajit import jit

    @jit
    def f(x):
        return math.sqrt(x) + math.exp(-x) + math.sin(x)
    v = f(2.5)
    ref = math.sqrt(2.5) + math.exp(-2.5) + math.sin(2.5)
    assert abs(v - ref) < 1e-12

    @jit
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)
    assert fib(20) == 6765


@check("numpy", "array args: 1-D/2-D/strided/fortran")
def c_arrays():
    import numpy as np
    from hanajit import jit

    @jit
    def s1(x):
        t = 0.0
        for i in range(len(x)):
            t += x[i]
        return t

    @jit
    def s2(m):
        t = 0.0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                t += m[i, j]
        return t
    a = np.arange(1000, dtype=np.float64)
    m = a.reshape(10, 100)
    assert abs(s1(a) - a.sum()) < 1e-9
    assert abs(s1(a[::2]) - a[::2].sum()) < 1e-9
    assert abs(s2(m) - m.sum()) < 1e-9
    assert abs(s2(np.asfortranarray(m)) - m.sum()) < 1e-9
    assert len(s1.cache) == 2 and len(s2.cache) == 2


@check("numpy", "slicing / reshape / reductions")
def c_views():
    import numpy as np
    from hanajit import jit

    @jit
    def f(x):
        w = x[100:900]
        return np.sum(w) + np.max(x) + np.mean(x.reshape(50, 20))
    a = np.random.default_rng(0).uniform(-1, 1, 1000)
    ref = a[100:900].sum() + a.max() + a.reshape(50, 20).mean()
    assert abs(f(a) - ref) < 1e-9
    assert len(f.cache) == 1


@check("threads", "nogil releases the GIL")
def c_nogil():
    import threading
    from hanajit import jit

    @jit(nogil=True)
    def spin(n):
        acc = 0.0
        for i in range(n):
            acc += (i % 7) * 0.5
        return acc
    spin(10)
    t = threading.Thread(target=spin, args=(300_000_000,))
    t.start()
    ticks = 0
    while t.is_alive():
        ticks += 1
    t.join()
    assert ticks > 1000, f"main thread starved (ticks={ticks})"
    return f"{ticks:,} main-thread ticks during kernel"


@check("threads", "prange correctness + scaling")
def c_prange():
    import numpy as np
    from hanajit import jit, prange
    a = np.random.default_rng(1).uniform(-1, 1, 4_000_000)

    @jit
    def serial(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i]
        return s
    ref = serial(a)
    t0 = time.perf_counter(); serial(a)
    t_serial = time.perf_counter() - t0
    rows = [f"serial {t_serial*1e3:.1f}ms"]
    for w in sorted({1, 2, os.cpu_count() or 1}):
        @jit(workers=w)
        def par(x):
            s = 0.0
            for i in prange(len(x)):
                s += x[i] * x[i]
            return s
        assert abs(par(a) - ref) < 1e-6 * abs(ref)
        best = min(_t(par, a) for _ in range(3))
        rows.append(f"w{w} {best*1e3:.1f}ms ({t_serial/best:.2f}x)")
    return " | ".join(rows)


def _t(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


@check("perf", "dispatch overhead + specialize")
def c_overhead():
    from hanajit import jit

    @jit
    def tiny(a, b):
        return a * b + 1
    tiny(1, 2)
    fast = tiny.specialize(int, int)

    def loop(f):
        t0 = time.perf_counter()
        for i in range(200000):
            f(i, 3)
        return (time.perf_counter() - t0) * 1e9 / 200000
    return (f"generic {loop(tiny):.0f} ns/call, "
            f"specialize {loop(fast):.0f} ns/call")


@check("perf", "evolve() (GA post-optimizer, short run)")
def c_evolve():
    import numpy as np
    from hanajit import jit

    @jit
    def red(x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * x[i] + 0.5 * x[i]
        return s
    a = np.random.default_rng(2).uniform(-1, 1, 1_000_000)
    ref = red(a)
    r = red.evolve(a, generations=3, population=6, reps=3,
                   allow_fastmath=True)
    assert abs(red(a) - ref) < 1e-6 * abs(ref)
    return (f"{r['baseline_ms']:.2f} -> {r['best_ms']:.2f} ms "
            f"({r['speedup']:.2f}x)")


@check("cache", "disk cache cold/warm (subprocess)")
def c_cache():
    with tempfile.TemporaryDirectory() as td:
        code = """
        import time, warnings; warnings.filterwarnings("ignore")
        t0 = time.perf_counter()
        from hanajit import jit
        @jit(cache=True)
        def k(n):
            acc = 0.0
            for i in range(n):
                acc += (i % 13) * 0.25
            return acc
        r = k(1000)
        print(f"{(time.perf_counter()-t0)*1e3:.1f} {r}")
        """
        env_line = f"import os; os.environ['HANAJIT_CACHE_DIR']={td!r}\n"
        full = env_line + textwrap.dedent(code)
        rc1, o1, e1 = subproc(full)
        rc2, o2, e2 = subproc(full)
        assert rc1 == 0 and rc2 == 0, (e1 or e2)[-500:]
        t1, r1 = o1.split(); t2, r2 = o2.split()
        assert r1 == r2
        return f"cold {t1} ms, warm {t2} ms"


@check("scipy", "LowLevelCallable via quad")
def c_scipy():
    try:
        import scipy  # noqa
    except ImportError:
        raise SkipCheck("scipy not installed")
    rc, out, err = subproc("""
        import warnings; warnings.filterwarnings("ignore")
        from hanajit import jit
        from scipy.integrate import quad
        import numpy as np
        def g(x):
            return 2.718281828459045 ** (-x * x)
        jf = jit(g)
        got, _ = quad(jf.scipy_callable(), -6, 6)
        ref, _ = quad(lambda x: np.exp(-x*x), -6, 6)
        assert abs(got - ref) < 1e-9, (got, ref)
        print(f"{got:.10f}")
        """)
    assert rc == 0, err[-500:]
    return f"quad = {out}"


def _gpu_check(vendor, marker):
    bdim = "block_dim()" if vendor != "amd" else "256"
    code = f"""
        import warnings; warnings.filterwarnings("ignore")
        from hanajit import jit
        @jit(target={vendor!r}, signature="f64*, f64*, f64, i64")
        def sax(x, y, a, n):
            i = block_id() * {bdim} + thread_id()
            if i < n:
                y[i] = a * x[i] + y[i]
            return 0
        art = sax.inspect_gpu()
        assert art is not None, "no artifact (eager emission failed)"
        v, text, native = art
        print("NATIVE" if native else "SOURCE",
              "MARKER_OK" if {marker.lower()!r} in text.lower() else "MARKER_MISSING")
        print(text[:160].replace(chr(10), " | "))
        """
    rc, out, err = subproc(code)
    assert rc == 0, f"emission crashed: {err[-400:]}"
    lines = out.splitlines()
    assert "MARKER_OK" in lines[0], out[:300]
    return f"{lines[0].split()[0]}: {lines[1][:90]}"


@check("gpu", "CUDA PTX emission")
def c_cuda():
    return _gpu_check("cuda", "tid")


@check("gpu", "AMD GCN emission")
def c_amd():
    return _gpu_check("amd", "amdgcn")


@check("gpu", "Intel SPIR-V emission (no thread intrinsics yet)")
def c_intel():
    code = """
        import warnings; warnings.filterwarnings("ignore")
        from hanajit import jit
        @jit(target="intel", signature="f64*, f64, i64")
        def scale(x, a, n):
            for i in range(n):
                x[i] = a * x[i]
            return 0
        art = scale.inspect_gpu()
        assert art is not None
        v, text, native = art
        print("NATIVE" if native else "SOURCE",
              "MARKER_OK" if "opcapability" in text.lower() else "MISSING")
        print(text[:160].replace(chr(10), " | "))
        """
    rc, out, err = subproc(code)
    assert rc == 0, f"emission crashed: {err[-400:]}"
    lines = out.splitlines()
    assert "MARKER_OK" in lines[0], out[:300]
    return f"{lines[0].split()[0]}: {lines[1][:90]}"


@check("gpu", "Metal MSL emission")
def c_metal():
    return _gpu_check("metal", "kernel void")


@check("gpu", "ptxas assembles our PTX (NVIDIA toolchain)")
def c_ptxas():
    if not shutil.which("ptxas"):
        raise SkipCheck("ptxas not on PATH (install CUDA toolkit)")
    rc, out, err = subproc("""
        import warnings, subprocess, tempfile, os
        warnings.filterwarnings("ignore")
        from hanajit import jit
        def make(arch=None):
            @jit(target="cuda", signature="f64*, f64*, f64, i64",
                 gpu_arch=arch)
            def sax(x, y, a, n):
                i = block_id() * block_dim() + thread_id()
                if i < n:
                    y[i] = a * x[i] + y[i]
                return 0
            v, ptx, native = sax.inspect_gpu()
            return ptx, native
        ptx, native = make()
        assert native
        import re
        # architectures this installed ptxas accepts, newest first
        h = subprocess.run(["ptxas", "--help"], capture_output=True,
                           text=True)
        # architectures this ptxas mentions anywhere (help text lists both
        # supported AND deprecated archs, so this is only a candidate pool,
        # never trusted directly - we PROBE each one).
        blob = (h.stdout or "") + (h.stderr or "")
        mentioned = sorted(set(re.findall("sm_[0-9]+", blob)),
                           key=lambda a: int(a[3:]), reverse=True)
        m = re.search("[.]target[ \\t]+(sm_[0-9]+)", ptx)
        want = m.group(1) if m else "sm_75"
        wn = int(want[3:])
        # candidate order: our target first, then every mentioned arch >=
        # ours (newest first, forward-compatible), then any remaining as a
        # last resort. Empirical: we actually try to assemble and keep the
        # first that returns 0 - help-text prose can't fool this.
        # try our default target first; then the LOWEST supported archs
        # (widest GPU compatibility) up to newest, then anything below.
        higher = sorted([a for a in mentioned if int(a[3:]) >= wn],
                        key=lambda a: int(a[3:]))            # low -> high
        lower = sorted([a for a in mentioned if int(a[3:]) < wn],
                       key=lambda a: int(a[3:]), reverse=True)
        candidates = []
        for a in [want] + higher + lower:
            if a not in candidates:
                candidates.append(a)
        with tempfile.TemporaryDirectory() as td:
            used, ok, errs = None, False, []
            for a in candidates:
                ptx_a, _ = make(a)            # emit PTX targeting this arch
                p = os.path.join(td, "k_" + a + ".ptx")
                open(p, "w").write(ptx_a)
                r = subprocess.run(["ptxas", "-arch=" + a, p, "-o",
                                    os.path.join(td, "k_" + a + ".cubin")],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    used, ok = a, True
                    break
                msg = ((r.stderr or "") + (r.stdout or "")).strip()
                errs.append(a + ": " + (msg[-100:] or "rc="
                                        + str(r.returncode)))
            assert ok, (f"default target {want}; tried {candidates[:8]}; "
                        + " | ".join(errs[:4]))
        print(f"cubin OK (assembled at {used}; hanajit emits matching PTX "
              f"via gpu_arch)")
        """)
    assert rc == 0, err[-500:]
    return out


@check("gpu", "llvm-mc assembles our AMD GCN (LLVM AMDGPU backend)")
def c_llvmmc():
    import shutil
    if not shutil.which("llvm-mc"):
        raise SkipCheck("llvm-mc not on PATH (install LLVM/clang or ROCm)")
    rc, out, err = subproc("""
        import warnings, subprocess, tempfile, os
        warnings.filterwarnings("ignore")
        from hanajit import jit
        @jit(target="amd", signature="f64*, f64*, f64, i64")
        def sax(x, y, a, n):
            i = block_id() * block_dim() + thread_id()
            if i < n:
                y[i] = a * x[i] + y[i]
            return 0
        v, asm, native = sax.inspect_gpu()
        assert native, "AMD emission fell back"
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "k.s")
            open(p, "w", encoding="utf-8").write(asm)
            r = subprocess.run(
                ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx90a",
                 "-filetype=obj", p, "-o", os.path.join(td, "k.o")],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace")
            msg = ((r.stderr or "") + (r.stdout or "")).strip()
            assert r.returncode == 0, msg[-260:] or ("rc=" + str(r.returncode))
            sz = os.path.getsize(os.path.join(td, "k.o"))
        print("object OK (" + str(sz) + " bytes, gfx90a)")
        """)
    assert rc == 0, err[-500:]
    return out


@check("gpu", "xcrun compiles our Metal source (macOS)")
def c_xcrun():
    if sys.platform != "darwin":
        raise SkipCheck("not macOS")
    if not shutil.which("xcrun"):
        raise SkipCheck("xcrun missing (install Xcode CLT)")
    rc, out, err = subproc("""
        import warnings, subprocess, tempfile, os
        warnings.filterwarnings("ignore")
        from hanajit import jit
        @jit(target="metal", signature="f64*, f64*, f64, i64")
        def sax(x, y, a, n):
            i = block_id() * block_dim() + thread_id()
            if i < n:
                y[i] = a * x[i] + y[i]
            return 0
        _, msl, _ = sax.inspect_gpu()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "k.metal")
            open(p, "w").write(msl)
            r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c",
                                p, "-o", os.path.join(td, "k.air")],
                               capture_output=True, text=True)
            if "missing Metal Toolchain" in (r.stderr or ""):
                print("NO_METAL_TOOLCHAIN")
                raise SystemExit(0)
            assert r.returncode == 0, r.stderr[-400:]
        print("AIR OK")
        """)
    if "NO_METAL_TOOLCHAIN" in out:
        # Xcode 16+ ships the Metal compiler as a separate download; MSL
        # emission itself passed — this is an environment step, not a bug.
        raise SkipCheck("Metal Toolchain not installed — run: "
                        "xcodebuild -downloadComponent MetalToolchain "
                        "(then re-run doctor)")
    assert rc == 0, err[-500:]
    return out


@check("gpu", "hardware detection")
def c_detect():
    from hanajit import detect
    return "; ".join(f"{t} ({why})" for t, why in detect())


@check("cross", "arm64-apple-darwin codegen")
def c_arm64():
    from hanajit import jit
    from hanajit.backends.cpu import emit_cross

    @jit
    def k(a, x):
        s = 0.0
        for i in range(a):
            s += x * i
        return s
    k(10, 1.5)
    asm = emit_cross(next(iter(k.modules.values())), "arm64-apple-darwin",
                     cpu="apple-m1")
    assert "ret" in asm
    return f"{len(asm)} bytes of arm64 asm"


# ---------------------------------------------------------------- report
def main():
    import llvmlite.binding as llvm
    for c in CHECKS:
        print(f"  running: {c.__name__} ...", flush=True)
        c()

    tag = f"{platform.system().lower()}_{platform.machine().lower()}"
    out = f"hanajit_report_{tag}.md"
    npv = scv = nbv = "not installed"
    try:
        import numpy; npv = numpy.__version__
    except ImportError:
        pass
    try:
        import scipy; scv = scipy.__version__
    except ImportError:
        pass
    try:
        import numba; nbv = numba.__version__
    except ImportError:
        pass
    try:
        cpu = llvm.get_host_cpu_name()
    except Exception:
        cpu = "?"
    import hanajit
    env = {
        "hanajit": hanajit.__version__,
        "python": sys.version, "implementation": sys.implementation.name,
        "platform": platform.platform(), "machine": platform.machine(),
        "cores": os.cpu_count(), "host_cpu(llvm)": cpu,
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "numpy": npv, "scipy": scv, "numba": nbv,
    }
    lines = [f"# hanajit doctor — {tag}", "",
             "```json", json.dumps(env, indent=2, default=str), "```", "",
             "| section | check | status | detail |", "|---|---|---|---|"]
    for sec, name, st, detail in RESULTS:
        lines.append(f"| {sec} | {name} | **{st}** | {detail} |")
    if FAILURES:
        lines.append("\n## Failure tracebacks\n")
        for name, tb in FAILURES:
            lines.append(f"### {name}\n```\n{tb[-3000:]}\n```")
    open(out, "w").write("\n".join(lines) + "\n")

    n_fail = sum(1 for *_, s, _ in RESULTS if s == "FAIL")
    n_pass = sum(1 for *_, s, _ in RESULTS if s == "PASS")
    n_skip = sum(1 for *_, s, _ in RESULTS if s == "SKIP")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"report written to current directory: {out}  <- send this file back for debugging")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
