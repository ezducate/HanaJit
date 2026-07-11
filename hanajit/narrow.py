"""Narrow-integer compute mode (EXPERIMENTAL, opt-in).

This is the integer companion to float32 mode: for large, memory-bandwidth-
bound integer reductions, storing the data as int8 / int16 / int32 (instead of
int64) moves fewer bytes and lets a SIMD register hold more lanes per load.

The compiled kernel loads narrow elements as a vector, sign-extends each lane
to int64 IN REGISTER, and accumulates in a wide int64 vector accumulator. Because
accumulation is always 64-bit, the result is bit-exact with the int64 sum — there
is no accumulator overflow (the failure mode of naive narrowing).

Scope and honesty:
  * Helps ONLY memory-bound integer reductions over large arrays. On compute-
    bound or small-array code, narrowing does nothing (or costs a little).
  * Measured speedups on a memory-bound sum are roughly int8 ~2.3-2.6x,
    int16 ~2.0-2.3x, int32 ~1.5-1.7x over an int64 baseline. These are
    bandwidth-dependent and will differ on your hardware.
  * This mode is EXPERIMENTAL and gated behind an explicit confirmed=True,
    exactly like hyper mode. Unlike hyper mode the RESULT is exact; what is
    "experimental" is the codegen path and the narrow-storage requirement.
  * int4 / int2 are NOT supported on CPU: there are no sub-byte SIMD load
    instructions, so they require bit-unpacking whose cost eats the bandwidth
    saving. They belong on the GPU/accelerator roadmap, not here.

The mode currently accelerates the sum reduction over a 1-D contiguous integer
array. Other reductions fall back to the normal compiler with a warning.
"""
import ctypes
import numpy as np
import llvmlite.binding as llvm
import llvmlite.ir as ir

# supported narrow widths (bits) -> (numpy dtype, ctypes type, SIMD vector width)
_WIDTHS = {
    8:  (np.int8,  ctypes.c_int8,  32),
    16: (np.int16, ctypes.c_int16, 16),
    32: (np.int32, ctypes.c_int32, 8),
}

# cache of compiled kernels: bits -> (cfun, engine)
_KERNELS = {}


def supported_dtype(arr):
    """Return the bit width if arr is a 1-D contiguous narrow int array, else None."""
    if not isinstance(arr, np.ndarray):
        return None
    if arr.ndim != 1 or not arr.flags["C_CONTIGUOUS"]:
        return None
    if arr.dtype == np.int8:
        return 8
    if arr.dtype == np.int16:
        return 16
    if arr.dtype == np.int32:
        return 32
    return None


def _build_narrow_sum_ir(bits):
    """Emit LLVM IR for a sum reduction over int<bits> with i64 accumulation.

    Vector loop: load VW narrow lanes, sext to <VW x i64>, add into a vector
    accumulator. Then horizontally reduce and run a scalar tail. Accumulation is
    64-bit throughout, so the result cannot overflow the narrow width.
    """
    _, _, VW = _WIDTHS[bits]
    narrow = ir.IntType(bits)
    i64 = ir.IntType(64)
    vecN = ir.VectorType(narrow, VW)
    vec64 = ir.VectorType(i64, VW)

    mod = ir.Module(name=f"hanajit_narrow_sum_{bits}")
    mod.triple = llvm.get_process_triple()
    fnty = ir.FunctionType(i64, [narrow.as_pointer(), i64])
    fn = ir.Function(mod, fnty, name=f"narrow_sum_{bits}")
    ptr, n = fn.args

    entry = fn.append_basic_block("entry")
    vcheck = fn.append_basic_block("vcheck")
    vbody = fn.append_basic_block("vbody")
    hred = fn.append_basic_block("hred")
    tcheck = fn.append_basic_block("tcheck")
    tbody = fn.append_basic_block("tbody")
    done = fn.append_basic_block("done")

    b = ir.IRBuilder(entry)
    VWc = ir.Constant(i64, VW)
    zero = ir.Constant(i64, 0)
    nvec = b.udiv(n, VWc)
    vend = b.mul(nvec, VWc)
    accv0 = ir.Constant(vec64, None)
    b.branch(vcheck)

    b.position_at_end(vcheck)
    iv = b.phi(i64, "iv")
    accv = b.phi(vec64, "accv")
    iv.add_incoming(zero, entry)
    accv.add_incoming(accv0, entry)
    b.cbranch(b.icmp_unsigned("<", iv, vend), vbody, hred)

    b.position_at_end(vbody)
    eptr = b.gep(ptr, [iv])
    vptr = b.bitcast(eptr, vecN.as_pointer())
    v = b.load(vptr)
    vw = b.sext(v, vec64)
    accv_next = b.add(accv, vw)
    iv_next = b.add(iv, VWc)
    iv.add_incoming(iv_next, vbody)
    accv.add_incoming(accv_next, vbody)
    b.branch(vcheck)

    b.position_at_end(hred)
    parts = [b.extract_element(accv, ir.Constant(i64, l)) for l in range(VW)]
    hsum = parts[0]
    for p in parts[1:]:
        hsum = b.add(hsum, p)
    b.branch(tcheck)

    b.position_at_end(tcheck)
    ti = b.phi(i64, "ti")
    tacc = b.phi(i64, "tacc")
    ti.add_incoming(vend, hred)
    tacc.add_incoming(hsum, hred)
    b.cbranch(b.icmp_unsigned("<", ti, n), tbody, done)

    b.position_at_end(tbody)
    ep = b.gep(ptr, [ti])
    ev = b.load(ep)
    ew = b.sext(ev, i64)
    tacc_next = b.add(tacc, ew)
    ti_next = b.add(ti, ir.Constant(i64, 1))
    ti.add_incoming(ti_next, tbody)
    tacc.add_incoming(tacc_next, tbody)
    b.branch(tcheck)

    b.position_at_end(done)
    res = b.phi(i64)
    res.add_incoming(tacc, tcheck)
    b.ret(res)
    return str(mod)


def _compile_kernel(bits):
    if bits in _KERNELS:
        return _KERNELS[bits]
    src = _build_narrow_sum_ir(bits)
    m = llvm.parse_assembly(src)
    m.verify()
    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine(opt=3)
    ee = llvm.create_mcjit_compiler(m, tm)
    ee.finalize_object()
    addr = ee.get_function_address(f"narrow_sum_{bits}")
    _, ctype, _ = _WIDTHS[bits]
    cfun = ctypes.CFUNCTYPE(
        ctypes.c_int64, ctypes.POINTER(ctype), ctypes.c_int64)(addr)
    _KERNELS[bits] = (cfun, ee)   # keep ee alive to keep the code mapped
    return _KERNELS[bits]


def narrow_sum(arr):
    """Compute the exact int64 sum of a narrow int array via the widening kernel.

    Returns a Python int identical to int(arr.astype(np.int64).sum()).
    Raises ValueError if arr is not a supported narrow array.
    """
    bits = supported_dtype(arr)
    if bits is None:
        raise ValueError(
            "narrow mode supports 1-D contiguous int8/int16/int32 arrays only")
    cfun, _ee = _compile_kernel(bits)
    _, ctype, _ = _WIDTHS[bits]
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    return int(cfun(ptr, arr.shape[0]))
