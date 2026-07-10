"""Native dispatch: a callable proxy whose CPython *vectorcall* slot points
at LLVM-JITed dispatch code.

Numba beats ctypes-based dispatch because its Dispatcher is a C extension
type. We get the same class of speed with zero compile-time dependencies:

1. A heap type `HanaFunction` is created via PyType_FromSpec (ctypes) with
   Py_TPFLAGS_HAVE_VECTORCALL, a per-instance vectorcall pointer, and an
   instance __dict__ (so the public API hangs off the same object).
2. For each @jit function we JIT a dispatcher in LLVM:
   check kwnames/nargs -> compare Py_TYPE(arg) pointers against the known
   specializations -> unbox via CPython C-API -> call the kernel by direct
   function-pointer -> box the result. Unknown signatures tail-call the
   Python slow path via PyObject_Vectorcall (which compiles a new
   specialization, rebuilds this dispatcher, and swaps the instance's
   vectorcall pointer -- an atomic 8-byte store).

So `f(1, 2)` runs dispatch, unboxing, kernel, and boxing entirely in
native code.
"""
import ctypes as C
from llvmlite import ir, binding as llvm
from ..typeinfer import I64, F64, BOOL, POINTER_ELEM

I8P = ir.IntType(8).as_pointer()
I64T = ir.IntType(64)
LLTY = {I64: I64T, F64: ir.DoubleType(), BOOL: ir.IntType(1)}
NARGS_MASK = (1 << 63) - 1  # clear PY_VECTORCALL_ARGUMENTS_OFFSET

# ---------------------------------------------------------------- type setup
T_PYSSIZET, READONLY = 19, 1
Py_tp_call, Py_tp_members = 50, 72
Py_TPFLAGS_HAVE_VECTORCALL = 1 << 11
DICT_OFF, VC_OFF, BASICSIZE = 16, 24, 32


class _Slot(C.Structure):
    _fields_ = [("slot", C.c_int), ("pfunc", C.c_void_p)]


class _Spec(C.Structure):
    _fields_ = [("name", C.c_char_p), ("basicsize", C.c_int),
                ("itemsize", C.c_int), ("flags", C.c_uint),
                ("slots", C.POINTER(_Slot))]


class _Member(C.Structure):
    _fields_ = [("name", C.c_char_p), ("type", C.c_int),
                ("offset", C.c_ssize_t), ("flags", C.c_int),
                ("doc", C.c_char_p)]


_keepalive = []
_HanaFunction = None


def _type_addr(name):
    return C.addressof(C.c_void_p.in_dll(C.pythonapi, name))


TYPE_ADDR = None  # python type -> address of its PyTypeObject


def _check_supported():
    import sys, struct, sysconfig
    if sys.implementation.name != "cpython":
        raise RuntimeError("native dispatch requires CPython")
    if sys.version_info < (3, 12):
        # PyType_FromSpec only honors Py_TPFLAGS_HAVE_VECTORCALL since 3.12
        raise RuntimeError("native dispatch requires CPython >= 3.12")
    if struct.calcsize("P") != 8:
        raise RuntimeError("native dispatch requires a 64-bit build")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        # free-threaded builds change the PyObject header layout
        raise RuntimeError("native dispatch unsupported on free-threaded builds")


def get_proxy_type():
    """Create (once) and return the HanaFunction heap type."""
    global _HanaFunction, TYPE_ADDR
    if _HanaFunction is not None:
        return _HanaFunction
    _check_supported()
    TYPE_ADDR = {int: _type_addr("PyLong_Type"),
                 float: _type_addr("PyFloat_Type"),
                 bool: _type_addr("PyBool_Type")}
    members = (_Member * 3)(
        _Member(b"__vectorcalloffset__", T_PYSSIZET, VC_OFF, READONLY, None),
        _Member(b"__dictoffset__", T_PYSSIZET, DICT_OFF, READONLY, None),
        _Member(None, 0, 0, 0, None))
    vc_call = C.cast(C.pythonapi.PyVectorcall_Call, C.c_void_p).value
    slots = (_Slot * 3)(_Slot(Py_tp_call, vc_call),
                        _Slot(Py_tp_members, C.cast(members, C.c_void_p).value),
                        _Slot(0, None))
    spec = _Spec(b"hanajit.HanaFunction", BASICSIZE, 0,
                 Py_TPFLAGS_HAVE_VECTORCALL, C.cast(slots, C.POINTER(_Slot)))
    _keepalive.extend([members, slots, spec])
    C.pythonapi.PyType_FromSpec.restype = C.py_object
    C.pythonapi.PyType_FromSpec.argtypes = [C.POINTER(_Spec)]
    _HanaFunction = C.pythonapi.PyType_FromSpec(C.byref(spec))
    return _HanaFunction


def install(proxy, dispatch_addr):
    """Atomically point the instance's vectorcall slot at native code."""
    C.cast(id(proxy) + VC_OFF, C.POINTER(C.c_void_p))[0] = dispatch_addr


def _register_symbols():
    from .fastcall import _register_symbols as reg
    reg()
    addr = C.cast(C.pythonapi.PyObject_Vectorcall, C.c_void_p).value
    llvm.add_symbol("PyObject_Vectorcall", addr)


# ------------------------------------------------------------- IR generation
def _declare(module, name, ret, args):
    for f in module.functions:
        if f.name == name:
            return f
    return ir.Function(module, ir.FunctionType(ret, args), name=name)


def build_dispatch_module(specs, fallback_obj_addr, name="hana_dispatch"):
    """specs: list of (py_types tuple, kernel_addr, abstract_arg_types,
    abstract_ret_type). Returns an ir.Module with one vectorcall function."""
    _register_symbols()
    get_proxy_type()
    module = ir.Module(name="hanajit.dispatch")
    fnty = ir.FunctionType(I8P, [I8P, I8P.as_pointer(), I64T, I8P])
    fn = ir.Function(module, fnty, name=name)
    self_, argv, nargsf, kwnames = fn.args

    as_i64 = _declare(module, "PyLong_AsLongLong", I64T, [I8P])
    as_f64 = _declare(module, "PyFloat_AsDouble", ir.DoubleType(), [I8P])
    from_i64 = _declare(module, "PyLong_FromLongLong", I8P, [I64T])
    from_f64 = _declare(module, "PyFloat_FromDouble", I8P, [ir.DoubleType()])
    from_bool = _declare(module, "PyBool_FromLong", I8P, [I64T])
    err_occ = _declare(module, "PyErr_Occurred", I8P, [])
    vcall = _declare(module, "PyObject_Vectorcall", I8P,
                     [I8P, I8P.as_pointer(), I64T, I8P])

    entry = fn.append_basic_block("entry")
    fallback = fn.append_basic_block("fallback")
    b = ir.IRBuilder(entry)
    null = ir.Constant(I8P, None)

    nargs = b.and_(nargsf, ir.Constant(I64T, NARGS_MASK))
    has_kw = b.icmp_unsigned("!=", kwnames, null)

    arity = len(specs[0][0]) if specs else -1
    bad_arity = (b.icmp_signed("!=", nargs, ir.Constant(I64T, arity))
                 if specs else ir.Constant(ir.IntType(1), 1))
    slow = b.or_(has_kw, bad_arity)

    # preload arg slots + their type pointers once
    check0 = fn.append_basic_block("typecheck")
    b.cbranch(slow, fallback, check0)
    b.position_at_end(check0)
    slots, typeptrs = [], []
    for i in range(max(arity, 0)):
        s = b.load(b.gep(argv, [ir.Constant(I64T, i)]))
        slots.append(s)
        tp_slot = b.bitcast(b.gep(s, [ir.Constant(I64T, 8)]), I8P.as_pointer())
        typeptrs.append(b.load(tp_slot))  # ob_type at offset 8

    # chain of signature checks
    for k, (py_types, kaddr, arg_tys, ret_ty, nogil) in enumerate(specs):
        body = fn.append_basic_block(f"sig{k}.body")
        nxt = fn.append_basic_block(f"sig{k}.next")
        match = ir.Constant(ir.IntType(1), 1)
        for i, pt in enumerate(py_types):
            want = ir.Constant(I64T, TYPE_ADDR[pt]).inttoptr(I8P)
            match = b.and_(match, b.icmp_unsigned("==", typeptrs[i], want))
        b.cbranch(match, body, nxt)

        b.position_at_end(body)
        unboxed = []
        for i, ty in enumerate(arg_tys):
            if ty == F64:
                unboxed.append(b.call(as_f64, [slots[i]]))
            else:
                v = b.call(as_i64, [slots[i]])
                if ty == BOOL:
                    v = b.icmp_signed("!=", v, ir.Constant(I64T, 0))
                elif ty in POINTER_ELEM:
                    from ..codegen import LLTY as _LL
                    v = b.inttoptr(v, _LL[ty])
                unboxed.append(v)
        ok = fn.append_basic_block(f"sig{k}.call")
        errbb = fn.append_basic_block(f"sig{k}.err")
        b.cbranch(b.icmp_unsigned("==", b.call(err_occ, []), null), ok, errbb)
        b.position_at_end(errbb)
        b.ret(null)
        b.position_at_end(ok)
        kty = ir.FunctionType(LLTY[ret_ty], [LLTY[t] for t in arg_tys])
        kname = f"hana_k_{kaddr:x}"
        llvm.add_symbol(kname, kaddr)   # resolve external to the JITed kernel
        from ..codegen import LLTY as _LLK
        kfn = _declare(module, kname, _LLK[ret_ty], [_LLK[t] for t in arg_tys])
        if nogil:
            save = _declare(module, "PyEval_SaveThread", I8P, [])
            restore = _declare(module, "PyEval_RestoreThread",
                               ir.VoidType(), [I8P])
            ts = b.call(save, [])
            res = b.call(kfn, unboxed)
            b.call(restore, [ts])
        else:
            res = b.call(kfn, unboxed)
        if ret_ty == F64:
            obj = b.call(from_f64, [res])
        elif ret_ty == BOOL:
            obj = b.call(from_bool, [b.zext(res, I64T)])
        else:
            obj = b.call(from_i64, [res])
        b.ret(obj)
        b.position_at_end(nxt)
    b.branch(fallback)

    b.position_at_end(fallback)
    fb = ir.Constant(I64T, fallback_obj_addr).inttoptr(I8P)
    b.ret(b.call(vcall, [fb, argv, nargsf, kwnames]))
    return module
