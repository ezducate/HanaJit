"""CPU backend: LLVM MCJIT via llvmlite.binding, called through ctypes.

Compatible with both legacy (<0.45) and new-pass-manager llvmlite APIs.
"""
import ctypes
from llvmlite import binding as llvm
from ..typeinfer import I64, F64, BOOL, PF64, PI64, AF64, AI64, ARRAY_ELEM

_initialized = False
_engines = []  # keep engines alive (they own the JITed code memory)

CTYPES = {I64: ctypes.c_int64, F64: ctypes.c_double, BOOL: ctypes.c_bool,
          PF64: ctypes.c_void_p, PI64: ctypes.c_void_p}


def _proto(ret_type, arg_types):
    from ..codegen import arr_meta_count
    argt = []
    for t in arg_types:
        if t in ARRAY_ELEM:
            argt += ([ctypes.c_void_p]
                     + [ctypes.c_int64] * arr_meta_count(t))
        else:
            argt.append(CTYPES[t])
    return ctypes.CFUNCTYPE(CTYPES[ret_type], *argt)


def _init():
    global _initialized
    if _initialized:
        return
    for fn in ("initialize", "initialize_native_target",
               "initialize_native_asmprinter"):
        try:
            getattr(llvm, fn)()
        except (AttributeError, RuntimeError):
            pass  # newer llvmlite auto-initializes
    _initialized = True


def _optimize(mod, tm, opt, tuning=None):
    try:  # llvmlite >= 0.45: new pass manager
        pto = llvm.create_pipeline_tuning_options(speed_level=opt)
        for knob, val in (tuning or {}).items():
            try:
                setattr(pto, knob, val)
            except (AttributeError, TypeError):
                pass  # knob not in this llvmlite build
        pb = llvm.create_pass_builder(tm, pto)
        pb.getModulePassManager().run(mod, pb)
    except AttributeError:
        try:
            pto = llvm.PipelineTuningOptions()
            pto.speed_level = opt
            pb = llvm.PassBuilder(tm, pto)
            pb.getModulePassManager().run(mod, pb)
        except Exception:
            try:  # legacy pass manager
                pmb = llvm.PassManagerBuilder()
                pmb.opt_level = opt
                pm = llvm.ModulePassManager()
                pmb.populate(pm)
                pm.run(mod)
            except Exception:
                pass  # run unoptimized rather than fail


def _host_target_machine(opt):
    target = llvm.Target.from_default_triple()
    try:  # tune codegen for the actual host CPU (AVX etc.), like numba
        return target.create_target_machine(
            cpu=llvm.get_host_cpu_name(),
            features=llvm.get_host_cpu_features().flatten(), opt=opt)
    except Exception:
        return target.create_target_machine(opt=opt)


def load_cached(cached, func_name, arg_types):
    """Rebuild callables from cached object code. Returns
    (callable, kernel_addr, ret_type)."""
    obj_bytes, meta = cached
    _init()
    from . import fastcall as fc
    fc.register()  # CPython API symbols must resolve before linking
    tm = _host_target_machine(3)
    backing = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing, tm)
    engine.add_object_file(llvm.ObjectFileRef.from_data(obj_bytes))
    engine.finalize_object()
    _engines.append(engine)
    ret_type = meta["ret"]
    kernel_addr = engine.get_function_address(func_name)
    if meta.get("wrapper"):
        try:
            waddr = engine.get_function_address(meta["wrapper"])
            return fc.make_builtin(waddr, func_name), kernel_addr, ret_type
        except Exception:
            pass
    return (_proto(ret_type, arg_types)(kernel_addr), kernel_addr,
            ret_type)


def compile_module(module, func_name, arg_types, ret_type, opt=3,
                   fastcall=True, nogil=False, cache_key=None,
                   tuning=None, tm_opt=None):
    """Compile an llvmlite ir.Module.

    Returns (callable, is_fastcall). With fastcall=True the callable is a
    genuine CPython builtin (C-level dispatch); otherwise ctypes.
    """
    _init()
    wrapper_name = None
    if any(t in ARRAY_ELEM for t in arg_types):
        fastcall = False   # arrays use the expanded ctypes ABI
    if fastcall:
        try:
            from . import fastcall as fc
            fc.register()
            kernel = None
            for f in module.functions:
                if f.name == func_name:
                    kernel = f
                    break
            wrapper_name = fc.add_fastcall_wrapper(
                module, kernel, arg_types, ret_type, nogil=nogil)
        except Exception:
            wrapper_name = None
    tm = _host_target_machine(tm_opt if tm_opt is not None else opt)
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    _optimize(mod, tm, opt, tuning=tuning)

    engine = llvm.create_mcjit_compiler(mod, tm)
    captured = {}
    if cache_key is not None:
        engine.set_object_cache(
            notify_func=lambda m, buf: captured.setdefault("obj", buf))
    engine.finalize_object()
    _engines.append(engine)
    if cache_key is not None and "obj" in captured:
        from .. import cache as _cache
        _cache.save(cache_key, captured["obj"],
                    {"ret": ret_type, "args": list(arg_types),
                     "wrapper": wrapper_name})

    kernel_addr = engine.get_function_address(func_name)
    if wrapper_name is not None:
        try:
            from . import fastcall as fc
            waddr = engine.get_function_address(wrapper_name)
            return fc.make_builtin(waddr, func_name), kernel_addr
        except Exception:
            pass
    return _proto(ret_type, arg_types)(kernel_addr), kernel_addr


def compile_raw(module, func_name, opt=3):
    """Compile a module and return the raw address of one function."""
    _init()
    tm = _host_target_machine(opt)
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    _optimize(mod, tm, opt)
    engine = llvm.create_mcjit_compiler(mod, tm)
    engine.finalize_object()
    _engines.append(engine)
    return engine.get_function_address(func_name)


def emit_assembly(module, opt=3):
    """Return native assembly text for inspection."""
    _init()
    tm = _host_target_machine(opt)
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    return tm.emit_assembly(mod)


def emit_cross(module, triple, cpu=""):
    """Cross-compile a module's IR for another architecture (portability
    check, e.g. arm64-apple-darwin for Apple Silicon)."""
    _init()
    try:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
    except (AttributeError, RuntimeError):
        pass
    target = llvm.Target.from_triple(triple)
    tm = target.create_target_machine(cpu=cpu)
    ir_text = str(module)
    mod = llvm.parse_assembly(ir_text)
    _optimize(mod, tm, 3)
    return tm.emit_assembly(mod)
