"""@jit decorator: Numba-style lazy specialization with full-ecosystem fallback.

- Parses the function with CPython's own parser (ast.parse on source).
- On first call, infers types from the actual arguments, lowers to LLVM IR,
  and compiles per type-signature (so f(1, 2) and f(1.0, 2.0) get separate
  native specializations).
- Anything outside the compilable subset (objects, lists, numpy calls,
  imports, closures over arbitrary state...) raises UnsupportedError and
  transparently falls back to the original CPython function — the whole
  Python ecosystem keeps working, you just don't get native speed there.
"""
import ast
import functools
import inspect
import textwrap
import warnings

from .errors import UnsupportedError
from .typeinfer import (TypeInferencer, I64, F64, F32, BOOL, PF64, PI64,
                        AF64, AI64, POINTER_ELEM, ARRAY_ELEM, arr_ty)
from .codegen import CodeGen
from .backends import cpu as cpu_backend

PYTYPE = {int: I64, float: F64, bool: BOOL}
SIGTOK = {"i64": I64, "f64": F64, "bool": BOOL, "f64*": PF64, "i64*": PI64,
          "f64[]": AF64, "i64[]": AI64, "f64[:]": AF64, "i64[:]": AI64,
          "int": I64, "float": F64}
ABS_TO_PY = {I64: int, F64: float, BOOL: bool, PF64: int, PI64: int}


def _parse_signature(sig):
    tys = []
    for tok in sig.split(","):
        t = SIGTOK.get(tok.strip())
        if t is None:
            raise UnsupportedError(f"unknown type in signature: {tok.strip()!r}")
        tys.append(t)
    return tuple(tys)


def _get_func_ast(pyfunc):
    try:
        src = getattr(pyfunc, "__hanajit_source__", None) or \
            textwrap.dedent(inspect.getsource(pyfunc))
    except (OSError, TypeError) as e:  # REPL / exec'd code has no source
        raise UnsupportedError(f"source unavailable: {e}")
    tree = ast.parse(src)  # CPython's own parser
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise UnsupportedError("expected a plain function definition")
    if fn.decorator_list:
        fn.decorator_list = []
    return fn, src


def _get_compiled_ast(pyfunc, rewrite=False):
    """AST for codegen: helpers inlined; optionally structural rewrites."""
    fn, src = _get_func_ast(pyfunc)
    from . import inline as _inline
    fn = _inline.inline_calls(fn)
    if rewrite:
        from . import rewrite as _rw
        fn, _stats = _rw.rewrite(fn)
    return fn, src


_NP_BASE = {"float64": "f64", "int64": "i64", "float32": "f32"}
_ARR_TO_PTR = {AF64: PF64, AI64: PI64}


def _array_abstract(a):
    """Abstract type for a numpy array argument, or raise UnsupportedError
    (which routes the call to the interpreter, where numpy works anyway)."""
    t = type(a)
    if t.__module__ != "numpy" or t.__name__ != "ndarray":
        raise UnsupportedError(f"unsupported argument type: {t.__name__}")
    base = _NP_BASE.get(str(a.dtype))
    if base is None:
        raise UnsupportedError(f"unsupported array dtype: {a.dtype}")
    it = a.itemsize
    if a.ndim == 1:
        if a.strides[0] == it:
            return arr_ty(base, 1, True)
        if a.strides[0] % it == 0:
            return arr_ty(base, 1, False)
    elif a.ndim == 2:
        s0, s1 = a.strides
        if s1 == it and s0 == it * a.shape[1]:
            return arr_ty(base, 2, True)
        if s0 % it == 0 and s1 % it == 0:
            return arr_ty(base, 2, False)
    raise UnsupportedError(
        f"unsupported array layout: ndim={a.ndim}, strides={a.strides}")


def _arr_abi(a, kind):
    it = a.itemsize
    if kind.endswith("1c]"):
        return [a.ctypes.data, a.shape[0]]
    if kind.endswith("1s]"):
        return [a.ctypes.data, a.shape[0], a.strides[0] // it]
    if kind.endswith("2c]"):
        return [a.ctypes.data, a.shape[0], a.shape[1]]
    return [a.ctypes.data, a.shape[0], a.shape[1],
            a.strides[0] // it, a.strides[1] // it]


def _is_ndarray(a):
    t = type(a)
    return t.__module__ == "numpy" and t.__name__ == "ndarray"


def _hybrid_key(args):
    # arrays key on full kind (dtype + ndim + layout): a matrix and its
    # transpose must never share a compiled caller
    return tuple(_array_abstract(a) if _is_ndarray(a) else type(a)
                 for a in args)


def _make_array_caller(native, sig):
    def call(*args):
        conv = []
        for a, t in zip(args, sig):
            if t in ARRAY_ELEM:                 # -> (ptr, shape..., strides...)
                conv += _arr_abi(a, t)
            elif t in POINTER_ELEM and _is_ndarray(a):
                conv.append(a.ctypes.data)      # raw-pointer signature
            else:
                conv.append(a)
        # `args` holds references, keeping the arrays alive for the call
        return native(*conv)
    return call


def _abstract_types(args):
    tys = []
    for a in args:
        t = PYTYPE.get(type(a))
        if t is None:
            t = _array_abstract(a)
        tys.append(t)
    return tuple(tys)


class Dispatcher:
    def __init__(self, pyfunc, target="cpu", fallback=True, verbose=False,
                 fastmath=False, nogil=False, signature=None, cache=False,
                 gpu_arch=None, rewrite=False, reduce_reassoc=False):
        self.pyfunc = pyfunc
        self.gpu_arch = gpu_arch
        self.rewrite = rewrite
        self.reduce_reassoc = reduce_reassoc
        if target == "auto":
            target = self._resolve_auto(pyfunc)
        self.target = target
        self.fallback = fallback
        self.verbose = verbose
        self.fastmath = fastmath
        self.nogil = nogil
        self.disk_cache = cache
        self.signature = _parse_signature(signature) if signature else None
        self._specs = []         # (py_types, kernel_addr, arg_tys, ret_ty)
        self._arr_fast = {}      # dtype-aware key -> array-converting callable
        self._sig_ret = {}       # abstract sig -> return type
        self._cache_keys = {}    # abstract sig -> disk cache key (if any)
        self._proxy = None
        self.cache = {}          # abstract signature -> native callable
        self._fast = {}          # tuple of python types -> native callable
        self.modules = {}        # signature -> llvm ir module (for inspection)
        self.gave_up = False
        functools.update_wrapper(self, pyfunc)

    @staticmethod
    def _resolve_auto(pyfunc):
        """auto -> detected GPU for kernels using thread intrinsics
        (they cannot execute on the CPU), else cpu (the only target
        hanajit executes directly)."""
        from .backends import detect
        from .typeinfer import GPU_INTRINSICS
        try:
            fn_ast, _ = _get_func_ast(pyfunc)
            uses_gpu = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in GPU_INTRINSICS for n in ast.walk(fn_ast))
        except UnsupportedError:
            return "cpu"
        if not uses_gpu:
            return "cpu"
        gpu = detect.best_gpu()
        if gpu is None:
            raise ValueError(
                f"@jit(target='auto'): {pyfunc.__name__!r} uses GPU thread "
                "intrinsics but no GPU runtime was detected on this machine "
                "(and it cannot run on the CPU). Pass target='cuda'/'amd'/"
                "'intel'/'metal' explicitly to emit device code anyway.")
        return gpu

    def _compile(self, sig):
        fn_ast, src = _get_compiled_ast(self.pyfunc, rewrite=self.rewrite)
        arg_names = [a.arg for a in fn_ast.args.args]
        if len(arg_names) != len(sig):
            raise UnsupportedError("defaults/varargs/kwargs not supported")

        cache_key = None
        if self.disk_cache and self.target == "cpu":
            import llvmlite
            from llvmlite import binding as _llvm
            from . import cache as _cache
            from . import __version__ as _ver
            cache_key = self._cache_keys[sig] = _cache.make_key(src, sig, {
                "cpu": _llvm.get_host_cpu_name(),
                "llvmlite": llvmlite.__version__, "hanajit": _ver,
                "fastmath": self.fastmath, "nogil": self.nogil})
            hit = _cache.load(cache_key)
            if hit is not None:
                try:
                    native, kaddr, ret_type = cpu_backend.load_cached(
                        hit, fn_ast.name, sig)
                    self._pending = (kaddr, sig, ret_type)
                    if self.verbose:
                        print(f"[hanajit] loaded {fn_ast.name}{sig} "
                              "from disk cache")
                    return native
                except Exception:
                    pass  # corrupt/stale entry: recompile and overwrite

        arg_types = dict(zip(arg_names, sig))
        var_types, ret_type = TypeInferencer(fn_ast, arg_types).run()
        self._sig_ret[sig] = ret_type
        if (self.target in ("cuda", "amd", "intel", "metal")
                and any(t in ARRAY_ELEM for t in sig)):
            raise UnsupportedError(
                "GPU kernels take raw pointers: use 'f64*' in signature=, "
                "not 'f64[]'")
        if self.target == "metal":
            from .backends import metal
            src = metal.transpile(fn_ast, arg_types, var_types, ret_type)
            self._gpu_artifact = ("metal", src, False)
            raise UnsupportedError(
                "metal target emits MSL source (see .inspect_gpu()); "
                "validate with xcrun on macOS — host execution falls back")
        gpu = self.target if self.target in ("cuda", "amd", "intel") else None
        module = CodeGen(fn_ast, arg_types, var_types, ret_type,
                         reduce_reassoc=self.reduce_reassoc,
                         fastmath=self.fastmath, gpu=gpu).generate()
        self.modules[sig] = module
        if self.target == "cpu":
            native, kaddr = cpu_backend.compile_module(
                module, fn_ast.name, sig, ret_type, nogil=self.nogil,
                cache_key=cache_key)
            self._pending = (kaddr, sig, ret_type)
        elif self.target in ("cuda", "amd", "intel"):
            from .backends import gpu as gpu_backend
            text, is_native = gpu_backend.emit(
                module, fn_ast.name, self.target,
                cpu=getattr(self, "gpu_arch", None))
            self._gpu_artifact = (self.target, text, is_native)
            raise UnsupportedError(
                f"{self.target} target emits device code only in v0.1 "
                "(see .inspect_gpu()); host execution falls back to CPython")
        else:
            raise UnsupportedError(f"unknown target {self.target!r}")
        if self.verbose:
            print(f"[hanajit] compiled {self.pyfunc.__name__}{sig} -> native ({ret_type})")
        return native

    def __call__(self, *args, **kwargs):
        fn = self._fast.get(tuple(map(type, args))) if not kwargs else None
        if fn is not None:
            return fn(*args)
        return self._slow_call(*args, **kwargs)

    def _slow_call(self, *args, **kwargs):
        if kwargs or self.gave_up:
            return self.pyfunc(*args, **kwargs)
        if any(_is_ndarray(a) for a in args):
            try:
                return self._array_call(args)
            except UnsupportedError as e:
                if not self.fallback:
                    raise
                self.gave_up = True
                warnings.warn(f"hanajit: falling back to CPython for "
                              f"{self.pyfunc.__name__!r} ({e})", stacklevel=2)
                return self.pyfunc(*args)
        try:
            key = tuple(map(type, args))
            if self.signature is not None:
                if len(args) != len(self.signature):
                    raise UnsupportedError("arity mismatch with signature")
                sig = self.signature
            else:
                sig = _abstract_types(args)
            native = self.cache.get(sig)
            if native is None:
                native = self.cache[sig] = self._compile(sig)
                if self._pending is not None:
                    kaddr, asig, rty = self._pending
                    self._pending = None
                    self._specs.append((key, kaddr, asig, rty, self.nogil))
                    self._refresh_proxy()
            self._fast[key] = native
            return native(*args)
        except UnsupportedError as e:
            if not self.fallback:
                raise
            self.gave_up = True
            warnings.warn(
                f"hanajit: falling back to CPython for "
                f"{self.pyfunc.__name__!r} ({e})", stacklevel=2)
            return self.pyfunc(*args, **kwargs)

    def _array_call(self, args):
        key = _hybrid_key(args)
        fn = self._arr_fast.get(key)
        if fn is None:
            if self.signature is not None:
                if len(args) != len(self.signature):
                    raise UnsupportedError("arity mismatch with signature")
                sig = self.signature
                for a, t in zip(args, sig):
                    if _is_ndarray(a):
                        got = _array_abstract(a)
                        if got != t and _ARR_TO_PTR.get(got) != t:
                            raise UnsupportedError(
                                f"array dtype {a.dtype} does not match "
                                f"declared {t}")
            else:
                sig = _abstract_types(args)
            native = self.cache.get(sig)
            if native is None:
                native = self.cache[sig] = self._compile(sig)
                self._pending = None  # pointer sigs skip the native dispatcher
            fn = self._arr_fast[key] = _make_array_caller(native, sig)
        return fn(*args)

    def _build_ir(self, sig, fastmath=False):
        """Regenerate this signature's IR (e.g. with fastmath flags)."""
        fn_ast, _ = _get_compiled_ast(self.pyfunc)
        arg_types = dict(zip([a.arg for a in fn_ast.args.args], sig))
        var_types, ret_type = TypeInferencer(fn_ast, arg_types).run()
        gpu = self.target if self.target in ("cuda", "amd", "intel") else None
        return str(CodeGen(fn_ast, arg_types, var_types, ret_type,
                           fastmath=fastmath, gpu=gpu).generate())

    def evolve_hyper(self, *example_args, confirmed=False, **kw):
        """HYPER-AGGRESSIVE optimization. Applies unsafe fp transforms
        (reassociation, FMA contraction, reciprocals, approximate
        functions) and keeps whatever is fastest that passes a large RANDOM
        differential suite. The result is validated ONLY on those random
        inputs and MAY BE WRONG on untested inputs. It is per-session and is
        NEVER written to the disk cache. CPU only.

        Requires confirmed=True to proceed."""
        if not confirmed:
            raise UnsupportedError(
                "evolve_hyper() applies UNSAFE optimizations that may produce "
                "wrong results on untested inputs. It validates only on random "
                "probes and never caches the result. If you accept this, call "
                "with confirmed=True.")
        from . import evolve as _ev
        kw.setdefault("hyper_probes", 256)
        return _ev.evolve(self, example_args, hyper_aggressive=True,
                          _hyper_confirmed=True, **kw)

    def _install_evolved_hyper(self, sig, genome, ir_fast_text):
        """Install a hyper winner into LIVE dispatch only, never disk."""
        from .backends import cpu
        from .evolve import _apply_loop_md, _apply_hyper_attrs
        fn_ast, _ = _get_func_ast(self.pyfunc)
        module = _apply_loop_md(ir_fast_text, genome.get("uc", 0),
                                genome.get("vw", 0))
        module = _apply_hyper_attrs(module, genome)
        ret_type = self._sig_ret[sig]
        native, kaddr = cpu.compile_module(
            _IRText(module), fn_ast.name, sig, ret_type,
            opt=genome["speed"], nogil=self.nogil, cache_key=None,
            tuning={"loop_vectorization": bool(genome["lv"]),
                    "slp_vectorization": bool(genome["slp"]),
                    "loop_unrolling": bool(genome["unroll"]),
                    "loop_interleaving": bool(genome["inter"])},
            tm_opt=genome["tm"])
        self.cache[sig] = native
        self._arr_fast.clear()
        self._refresh_proxy()

    def evolve(self, *example_args, **kw):
        """Genetic search over semantics-preserving compilation strategies
        for the specialization matching example_args. See hanajit/evolve.py.
        Returns a report dict; installs the winner only if it beats the
        baseline on this machine."""
        from . import evolve as _ev
        return _ev.evolve(self, example_args, **kw)

    def _install_evolved(self, sig, genome, ir_text):
        from .backends import cpu, fastcall  # noqa
        fn_ast, _ = _get_func_ast(self.pyfunc)
        module = ir_text
        # recompile through the normal path (fastcall wrapper included)
        # with the winning tuning applied
        ret_type = self._sig_ret[sig]
        cache_key = None
        if self.disk_cache and self.target == "cpu":
            import llvmlite
            from llvmlite import binding as _llvm
            from . import cache as _cache
            from . import __version__ as _ver
            _, src = _get_func_ast(self.pyfunc)
            cache_key = _cache.make_key(src, sig, {
                "cpu": _llvm.get_host_cpu_name(),
                "llvmlite": llvmlite.__version__, "hanajit": _ver,
                "fastmath": self.fastmath, "nogil": self.nogil})
        native, kaddr = cpu.compile_module(
            _IRText(module),
            fn_ast.name, sig, ret_type, opt=genome["speed"],
            nogil=self.nogil, cache_key=cache_key, tuning={
                "loop_vectorization": bool(genome["lv"]),
                "slp_vectorization": bool(genome["slp"]),
                "loop_unrolling": bool(genome["unroll"]),
                "loop_interleaving": bool(genome["inter"])},
            tm_opt=genome["tm"])
        self.cache[sig] = native
        self._arr_fast.clear()
        for key, nat in list(self._fast.items()):
            if _abstract_types_from_key(key) == sig:
                self._fast[key] = native
        for i, (kt, ka, asig, rty, ng) in enumerate(self._specs):
            if asig == sig:
                self._specs[i] = (kt, kaddr, asig, rty, ng)
        self._refresh_proxy()

    def scipy_callable(self, nargs=1):
        """Expose a float64 specialization as a scipy.LowLevelCallable.

        The kernel must take `nargs` floats and return a float — the
        signature scipy.integrate.quad (nargs=1) and friends expect.
        scipy then calls the JITed code directly as a C function pointer:
        zero Python per evaluation.
        """
        import ctypes
        from scipy import LowLevelCallable
        sig = (F64,) * nargs
        if sig not in self.cache:
            self.cache[sig] = self._compile(sig)
            pend = self._pending
            self._pending = None
            if pend is not None:
                self._specs.append(((float,) * nargs, pend[0], sig, pend[2],
                                    self.nogil))
                self._refresh_proxy()
        kaddr = None
        for py_types, ka, asig, rty, _ in self._specs:
            if asig == sig:
                if rty != F64:
                    raise UnsupportedError(
                        "scipy_callable requires a float return")
                kaddr = ka
        if kaddr is None:
            raise UnsupportedError("no float64 kernel address available "
                                   "(cached compile?) — use cache=False")
        proto = ctypes.CFUNCTYPE(ctypes.c_double,
                                 *[ctypes.c_double] * nargs)
        cf = proto(kaddr)
        if not hasattr(self, "_scipy_keepalive"):
            self._scipy_keepalive = []
        self._scipy_keepalive.append(cf)
        return LowLevelCallable(cf)

    _pending = None

    def _refresh_proxy(self):
        if self._proxy is None:
            return
        from .backends import vectorcall as vc, cpu
        mod = vc.build_dispatch_module(self._specs, id(self._fallback_ref))
        addr = cpu.compile_raw(mod, "hana_dispatch")
        vc.install(self._proxy, addr)

    def make_proxy(self):
        """Wrap this dispatcher in a HanaFunction: a callable whose
        vectorcall slot is LLVM-JITed native dispatch (C-level overhead)."""
        from .backends import vectorcall as vc
        HanaFunction = vc.get_proxy_type()
        proxy = HanaFunction()
        self._proxy = proxy
        self._fallback_ref = self._slow_call  # keep bound method alive
        proxy.dispatcher = self
        proxy.cache, proxy._fast, proxy.modules = (self.cache, self._fast,
                                                   self.modules)
        for name in ("specialize", "inspect_llvm", "inspect_asm",
                     "inspect_gpu", "export_fpga", "scipy_callable",
                     "evolve", "evolve_hyper"):
            setattr(proxy, name, getattr(self, name))
        proxy.__wrapped__ = self.pyfunc
        proxy.__name__ = self.pyfunc.__name__
        self._refresh_proxy()  # installs empty dispatcher -> all-fallback
        return proxy

    def specialize(self, *types):
        """Return the raw JITed builtin for the given Python argument types.

        Bypasses the dispatcher entirely: C-level call overhead (faster than
        numba's dispatcher in our benchmarks). No type checking beyond
        CPython's unboxing — use in hot loops where types are stable.

            fast_fib = fib.specialize(int)
            fast_fib(32)
        """
        try:
            sig = tuple(PYTYPE[t] for t in types)
        except KeyError as e:
            raise UnsupportedError(f"specialize(): unsupported type {e}")
        native = self.cache.get(sig)
        if native is None:
            native = self.cache[sig] = self._compile(sig)
            self._fast[tuple(types)] = native
        return native

    # ---- introspection ----
    def inspect_llvm(self, sig=None):
        if not self.modules:
            raise RuntimeError("call the function once first")
        sig = sig or next(iter(self.modules))
        return str(self.modules[sig])

    def inspect_asm(self, sig=None):
        sig = sig or next(iter(self.modules))
        return cpu_backend.emit_assembly(self.modules[sig])

    def inspect_gpu(self):
        return getattr(self, "_gpu_artifact", None)

    def export_fpga(self, path_prefix, sig=None):
        from .backends import fpga as fpga_backend
        sig = sig or next(iter(self.modules))
        return fpga_backend.export_for_hls(
            self.modules[sig], self.pyfunc.__name__, path_prefix)


class _IRText:
    """Minimal shim: compile_module stringifies its module argument."""
    def __init__(self, text):
        self._t = text

    def __str__(self):
        return self._t


def _abstract_types_from_key(key):
    out = []
    for k in key:
        if isinstance(k, type):
            out.append(PYTYPE.get(k))
        else:  # abstract array-kind string from hybrid keys
            out.append(k if k in ARRAY_ELEM else None)
    return tuple(out)


def pmap(fn, argtuples, workers=None):
    """Run fn over argument tuples in a thread pool. With @jit(nogil=True)
    kernels this achieves true multi-core parallelism (no GIL contention)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: fn(*t), argtuples))


def jit(func=None, *, target="cpu", fallback=True, verbose=False,
        fastmath=False, native_dispatch=True, nogil=False, signature=None,
        cache=False, workers=None, parallel=False, gpu_arch=None,
        rewrite=False, reduce_reassoc=False):
    _parallel = parallel  # avoid shadowing by `from . import parallel`
    """Decorate a function for JIT compilation.

    @jit
    def f(x, y): ...

    @jit(target="cuda")          # emit PTX/NVPTX IR (experimental)
    @jit(fallback=False)         # raise instead of falling back
    """
    def wrap(f):
        try:
            fn_ast, src = _get_func_ast(f)
            from . import inline as _inline
            _inline.register(f.__name__, fn_ast)   # available as a helper
            uses_prange = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "prange" for n in ast.walk(fn_ast))
        except UnsupportedError:
            uses_prange = False
        par_ast = locals().get("fn_ast")
        if _parallel and par_ast is not None and target in ("cpu", "auto") \
                and not uses_prange:
            from . import autopar
            try:
                par_ast = autopar.rewrite_range_to_prange(fn_ast)
                uses_prange = True
            except UnsupportedError as e:
                if verbose:
                    print(f"[hanajit] parallel=True not applicable ({e}); "
                          "compiling serially")
        if uses_prange and target in ("cpu", "auto"):
            from . import parallel
            try:
                return parallel.make_parallel(
                    f, par_ast, dict(fallback=fallback, fastmath=fastmath,
                                     cache=cache), workers=workers)
            except UnsupportedError as e:
                if verbose:
                    print(f"[hanajit] loop not parallelized ({e}); "
                          "compiling serially")
        d = Dispatcher(f, target=target, fallback=fallback, verbose=verbose,
                       fastmath=fastmath, nogil=nogil, signature=signature,
                       cache=cache, gpu_arch=gpu_arch, rewrite=rewrite,
                       reduce_reassoc=reduce_reassoc)
        if d.signature is not None:  # eager compile: GPU emission / ptr kernels
            try:
                d._slow_call  # noqa - ensure attrs exist
                sig = d.signature
                if d.target == "cpu":
                    native = d.cache[sig] = d._compile(sig)
                    key = tuple(ABS_TO_PY[t] for t in sig)
                    d._fast[key] = native
                    if d._pending is not None:
                        kaddr, asig, rty = d._pending
                        d._pending = None
                        d._specs.append((key, kaddr, asig, rty, d.nogil))
                else:
                    d._compile(sig)   # raises UnsupportedError after emission
            except UnsupportedError:
                pass
        if d.target == "cpu" and native_dispatch:
            try:
                return d.make_proxy()
            except Exception:
                pass  # fall back to the Python dispatcher
        return d
    return wrap(func) if func is not None else wrap
