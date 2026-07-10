"""C-level dispatch: JIT a CPython METH_FASTCALL wrapper in LLVM.

Instead of calling native code through ctypes (~0.45us/call), we generate
an LLVM function with the CPython fastcall ABI:

    PyObject *wrapper(PyObject *self, PyObject *const *args, Py_ssize_t n)

that unboxes arguments (PyLong_AsLongLong / PyFloat_AsDouble), calls the
JITed kernel directly (same module -> inlinable), boxes the result, and is
then installed as a genuine builtin via PyCFunction_NewEx. Call overhead
drops to builtin-function level — the same trick Numba's C dispatcher uses.
"""
import ctypes
from llvmlite import ir, binding as llvm
from ..typeinfer import I64, F64, BOOL, PF64, PI64, POINTER_ELEM

I8P = ir.IntType(8).as_pointer()
I64T = ir.IntType(64)
F64T = ir.DoubleType()

_PYAPI = ["PyLong_AsLongLong", "PyFloat_AsDouble", "PyLong_FromLongLong",
          "PyFloat_FromDouble", "PyBool_FromLong", "PyErr_Occurred",
          "PyErr_SetString", "PyEval_SaveThread", "PyEval_RestoreThread"]
_registered = False
_keepalive = []


def _register_symbols():
    """Point MCJIT at the CPython API symbols in this process."""
    global _registered
    if _registered:
        return
    for name in _PYAPI:
        addr = ctypes.cast(getattr(ctypes.pythonapi, name), ctypes.c_void_p).value
        llvm.add_symbol(name, addr)
    exc = ctypes.c_void_p.in_dll(ctypes.pythonapi, "PyExc_TypeError")
    llvm.add_symbol("PyExc_TypeError", ctypes.addressof(exc))
    _registered = True


def _declare(module, name, ret, args):
    for f in module.functions:
        if f.name == name:
            return f
    return ir.Function(module, ir.FunctionType(ret, args), name=name)


def add_fastcall_wrapper(module, kernel, arg_types, ret_type, nogil=False):
    """Append a METH_FASTCALL wrapper for `kernel` to the module."""
    wname = kernel.name + "__fastcall"
    wty = ir.FunctionType(I8P, [I8P, I8P.as_pointer(), I64T])
    w = ir.Function(module, wty, name=wname)
    self_, argv, nargs = w.args

    as_i64 = _declare(module, "PyLong_AsLongLong", I64T, [I8P])
    as_f64 = _declare(module, "PyFloat_AsDouble", F64T, [I8P])
    from_i64 = _declare(module, "PyLong_FromLongLong", I8P, [I64T])
    from_f64 = _declare(module, "PyFloat_FromDouble", I8P, [F64T])
    from_bool = _declare(module, "PyBool_FromLong", I8P, [I64T])
    err_occ = _declare(module, "PyErr_Occurred", I8P, [])
    err_set = _declare(module, "PyErr_SetString", ir.VoidType(), [I8P, I8P])
    exc_type = ir.GlobalVariable(module, I8P, "PyExc_TypeError")

    msg_text = f"{kernel.name}() takes {len(arg_types)} positional arguments\0"
    msg = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), len(msg_text)),
                            name=wname + ".argmsg")
    msg.global_constant = True
    msg.initializer = ir.Constant(msg.type.pointee,
                                  bytearray(msg_text.encode()))

    entry = w.append_basic_block("entry")
    bad = w.append_basic_block("badargs")
    good = w.append_basic_block("unbox")
    fail = w.append_basic_block("unboxfail")
    ok = w.append_basic_block("call")
    b = ir.IRBuilder(entry)

    b.cbranch(b.icmp_signed("==", nargs, ir.Constant(I64T, len(arg_types))),
              good, bad)

    b.position_at_end(bad)
    b.call(err_set, [b.load(exc_type), b.bitcast(msg, I8P)])
    b.ret(ir.Constant(I8P, None))

    b.position_at_end(good)
    unboxed = []
    for i, ty in enumerate(arg_types):
        slot = b.load(b.gep(argv, [ir.Constant(I64T, i)]))
        if ty == F64:
            unboxed.append(b.call(as_f64, [slot]))
        else:
            v = b.call(as_i64, [slot])
            if ty == BOOL:
                v = b.icmp_signed("!=", v, ir.Constant(I64T, 0))
            elif ty in POINTER_ELEM:  # address passed as int
                from ..codegen import LLTY
                v = b.inttoptr(v, LLTY[ty])
            unboxed.append(v)
    b.cbranch(b.icmp_unsigned("==", b.call(err_occ, []),
                              ir.Constant(I8P, None)), ok, fail)

    b.position_at_end(fail)
    b.ret(ir.Constant(I8P, None))

    b.position_at_end(ok)
    if nogil:
        save = _declare(module, "PyEval_SaveThread", I8P, [])
        restore = _declare(module, "PyEval_RestoreThread", ir.VoidType(), [I8P])
        tstate = b.call(save, [])       # release the GIL
        res = b.call(kernel, unboxed)   # pure native compute, no CPython API
        b.call(restore, [tstate])       # reacquire before boxing
    else:
        res = b.call(kernel, unboxed)
    if ret_type == F64:
        obj = b.call(from_f64, [res])
    elif ret_type == BOOL:
        obj = b.call(from_bool, [b.zext(res, I64T)])
    else:
        obj = b.call(from_i64, [res])
    b.ret(obj)
    return wname


METH_FASTCALL = 0x0080


class _PyMethodDef(ctypes.Structure):
    _fields_ = [("ml_name", ctypes.c_char_p), ("ml_meth", ctypes.c_void_p),
                ("ml_flags", ctypes.c_int), ("ml_doc", ctypes.c_char_p)]


def make_builtin(wrapper_addr, name):
    """Wrap a JITed fastcall address as a real CPython builtin function."""
    _register_symbols()
    nm = name.encode()
    mdef = _PyMethodDef(nm, wrapper_addr, METH_FASTCALL, None)
    _keepalive.append((nm, mdef))
    new = ctypes.pythonapi.PyCFunction_NewEx
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.POINTER(_PyMethodDef), ctypes.c_void_p,
                    ctypes.c_void_p]
    return new(ctypes.byref(mdef), None, None)


def register():
    _register_symbols()
