"""Export a typed HanaJit kernel as a standalone x86-64 executable.

The executable is a small native CLI: positional arguments are parsed as the
declared scalar signature, the compiled kernel is called, and its result is
printed to stdout.  It contains no Python or HanaJit runtime.  In CUDA modes a
one-thread CUDA version of the same scalar kernel is embedded as PTX and loaded
through the NVIDIA driver API at runtime; ``optional`` falls back to the CPU,
while ``required`` exits if CUDA is unavailable.

"Self-contained" here means one application file with no Python installation,
HanaJit package, CUDA toolkit, or sidecar PTX.  Normal operating-system runtime
libraries are still used, and CUDA execution naturally requires an installed
NVIDIA driver.
"""
import ast
import copy
import functools
import os
import platform
import re
import shlex
import shutil
import subprocess
import warnings
from collections import namedtuple
from pathlib import Path

from ..codegen import CodeGen
from ..errors import UnsupportedError
from ..typeinfer import I64, F64, BOOL, PF64, PI64, POINTER_ELEM, ARRAY_ELEM
from ..typeinfer import TypeInferencer, MATH_FNS, MATH_MODULES


ExecutableExport = namedtuple(
    "ExecutableExport", "executable object source build ptx")

_CTY = {I64: "int64_t", F64: "double", BOOL: "bool"}
_CUDA_MODES = {"off", "optional", "required"}
_C_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _x86_64():
    machine = platform.machine().lower()
    return machine in ("amd64", "x86_64", "x64")


def _normalize_mode(cuda):
    if cuda is True:
        cuda = "optional"
    elif cuda is False or cuda is None:
        cuda = "off"
    cuda = str(cuda).lower()
    if cuda not in _CUDA_MODES:
        raise ValueError(
            "cuda must be 'off', 'optional', or 'required'")
    if cuda != "off" and sys_platform_is_macos():
        raise UnsupportedError(
            "standalone CUDA export is unavailable on macOS")
    return cuda


def sys_platform_is_macos():
    import sys
    return sys.platform == "darwin"


def _paths(output):
    exe = Path(output).expanduser().resolve()
    if os.name == "nt" and exe.suffix.lower() != ".exe":
        exe = Path(str(exe) + ".exe")
    base = exe.with_suffix("") if exe.suffix else exe
    obj = Path(str(base) + (".obj" if os.name == "nt" else ".o"))
    src = Path(str(base) + ".standalone.c")
    build = Path(str(base) + (".build.bat" if os.name == "nt"
                              else ".build.sh"))
    ptx = Path(str(base) + ".ptx")
    exe.parent.mkdir(parents=True, exist_ok=True)
    return exe, obj, src, build, ptx


class _CScalarGen:
    """Lower the already-typed scalar subset to portable C11.

    Integer helpers use unsigned arithmetic where C signed overflow would be
    undefined, preserving HanaJit's i64 wraparound contract.  Floor division
    and modulo use the same sign adjustment as the LLVM backend.
    """
    BIN = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}
    CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
           ast.Gt: ">", ast.GtE: ">="}

    def __init__(self, fn_ast, arg_types, var_types, ret_type):
        self.f = fn_ast
        self.arg_types = arg_types
        self.var_types = var_types
        self.ret_type = ret_type
        self.names = {name: f"hj_v{i}" for i, name in
                      enumerate(var_types)}
        self.out = []
        self.depth = 1
        self.loop_id = 0
        self.ti = TypeInferencer(fn_ast, arg_types)
        self.ti.run()

    def line(self, s=""):
        self.out.append("    " * self.depth + s)

    def v(self, name):
        return self.names[name]

    def etype(self, node):
        return self.ti.expr(node)

    def generate(self):
        params = ", ".join(
            f"{_CTY[self.arg_types[a.arg]]} {self.v(a.arg)}"
            for a in self.f.args.args) or "void"
        self.out = [f"static {_CTY[self.ret_type]} hj_cpu_kernel({params}) {{"]
        for name, ty in self.var_types.items():
            if name in self.arg_types:
                continue
            zero = "0.0" if ty == F64 else ("false" if ty == BOOL else "0")
            self.line(f"{_CTY[ty]} {self.v(name)} = {zero};")
        for stmt in self.f.body:
            self.stmt(stmt)
        zero = "0.0" if self.ret_type == F64 else (
            "false" if self.ret_type == BOOL else "0")
        self.line(f"return {zero};")
        self.out.append("}")
        return "\n".join(self.out)

    def stmt(self, node):
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if not isinstance(t, ast.Name):
                raise UnsupportedError(
                    "standalone scalar executable cannot store through pointers")
            self.line(f"{self.v(t.id)} = {self.expr(node.value)};")
            return
        if isinstance(node, ast.AugAssign):
            left = ast.Name(id=node.target.id, ctx=ast.Load())
            value = ast.BinOp(left=left, op=node.op, right=node.value)
            self.line(f"{self.v(node.target.id)} = {self.expr(value)};")
            return
        if isinstance(node, ast.Return):
            self.line(f"return {self.expr(node.value)};")
            return
        if isinstance(node, ast.If):
            self.line(f"if ({self.truth(node.test)}) {{")
            self.depth += 1
            for s in node.body:
                self.stmt(s)
            self.depth -= 1
            if node.orelse:
                self.line("} else {")
                self.depth += 1
                for s in node.orelse:
                    self.stmt(s)
                self.depth -= 1
            self.line("}")
            return
        if isinstance(node, ast.While):
            self.line(f"while ({self.truth(node.test)}) {{")
            self.depth += 1
            for s in node.body:
                self.stmt(s)
            self.depth -= 1
            self.line("}")
            return
        if isinstance(node, ast.For):
            args = node.iter.args
            start = "0" if len(args) == 1 else self.expr(args[0])
            stop = self.expr(args[0] if len(args) == 1 else args[1])
            step = "1" if len(args) < 3 else self.expr(args[2])
            k = self.loop_id
            self.loop_id += 1
            self.line(f"int64_t hj_start{k} = {start};")
            self.line(f"int64_t hj_stop{k} = {stop};")
            self.line(f"int64_t hj_step{k} = {step};")
            iv = self.v(node.target.id)
            self.line(f"for ({iv} = hj_start{k}; "
                      f"hj_step{k} > 0 ? {iv} < hj_stop{k} : "
                      f"{iv} > hj_stop{k}; "
                      f"{iv} = hj_add_i64({iv}, hj_step{k})) {{")
            self.depth += 1
            for s in node.body:
                self.stmt(s)
            self.depth -= 1
            self.line("}")
            return
        if isinstance(node, ast.Break):
            self.line("break;")
            return
        if isinstance(node, ast.Continue):
            self.line("continue;")
            return
        if isinstance(node, (ast.Expr, ast.Pass)):
            return
        raise UnsupportedError(
            f"standalone: unsupported statement {type(node).__name__}")

    def truth(self, node):
        e = self.expr(node)
        t = self.etype(node)
        if t == BOOL:
            return e
        if t == F64:
            return f"hj_truth_f64({e})"
        return f"(({e}) != 0)"

    def expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, int):
                bits = node.value & ((1 << 64) - 1)
                return f"((int64_t)UINT64_C({bits}))"
            if isinstance(node.value, float):
                if node.value != node.value:
                    return "NAN"
                if node.value == float("inf"):
                    return "INFINITY"
                if node.value == float("-inf"):
                    return "(-INFINITY)"
                return repr(node.value)
            raise UnsupportedError("standalone: unsupported constant")
        if isinstance(node, ast.Name):
            return self.v(node.id)
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand)
            t = self.etype(node.operand)
            if isinstance(node.op, ast.USub):
                return f"hj_neg_i64({v})" if t != F64 else f"(-({v}))"
            if isinstance(node.op, ast.UAdd):
                return f"(+({v}))"
            if isinstance(node.op, ast.Not):
                return f"(!({self.truth(node.operand)}))"
            raise UnsupportedError("standalone: unsupported unary operator")
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            lt, rt, out = (self.etype(node.left), self.etype(node.right),
                           self.etype(node))
            if isinstance(node.op, ast.Div):
                return f"(((double)({a})) / ((double)({b})))"
            if isinstance(node.op, ast.FloorDiv):
                if out == F64:
                    return f"floor(((double)({a})) / ((double)({b})))"
                return f"hj_floor_div_i64({a}, {b})"
            if isinstance(node.op, ast.Mod):
                if out == F64:
                    return f"hj_mod_f64((double)({a}), (double)({b}))"
                return f"hj_mod_i64({a}, {b})"
            if isinstance(node.op, ast.Pow):
                p = f"pow((double)({a}), (double)({b}))"
                return p if out == F64 else f"((int64_t)({p}))"
            if isinstance(node.op, ast.Add):
                return (f"(({a}) + ({b}))" if out == F64
                        else f"hj_add_i64({a}, {b})")
            if isinstance(node.op, ast.Sub):
                return (f"(({a}) - ({b}))" if out == F64
                        else f"hj_sub_i64({a}, {b})")
            if isinstance(node.op, ast.Mult):
                return (f"(({a}) * ({b}))" if out == F64
                        else f"hj_mul_i64({a}, {b})")
            op = self.BIN.get(type(node.op))
            if op:
                return f"(({a}) {op} ({b}))"
            if isinstance(node.op, ast.LShift):
                return f"((int64_t)((uint64_t)({a}) << (uint64_t)({b})))"
            if isinstance(node.op, ast.RShift):
                return f"(({a}) >> ({b}))"
            raise UnsupportedError("standalone: unsupported binary operator")
        if isinstance(node, ast.Compare):
            op = self.CMP.get(type(node.ops[0]))
            if op is None:
                raise UnsupportedError("standalone: unsupported comparison")
            if F64 in (self.etype(node.left),
                       self.etype(node.comparators[0])):
                fn = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt",
                      ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}[
                          type(node.ops[0])]
                return (f"hj_fcmp_{fn}((double)({self.expr(node.left)}), "
                        f"(double)({self.expr(node.comparators[0])}))")
            return (f"(({self.expr(node.left)}) {op} "
                    f"({self.expr(node.comparators[0])}))")
        if isinstance(node, ast.BoolOp):
            op = " & " if isinstance(node.op, ast.And) else " | "
            return "(" + op.join(f"({self.truth(v)})" for v in node.values) + ")"
        if isinstance(node, ast.IfExp):
            return (f"({self.truth(node.test)} ? {self.expr(node.body)} : "
                    f"{self.expr(node.orelse)})")
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in MATH_MODULES
                and node.attr in ("pi", "e")):
            return ("3.141592653589793238462643383279502884"
                    if node.attr == "pi" else
                    "2.718281828459045235360287471352662498")
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in MATH_MODULES
                    and node.func.attr in MATH_FNS + ("abs",)):
                if node.func.attr == "abs":
                    v = self.expr(node.args[0])
                    return (f"fabs({v})" if self.etype(node.args[0]) == F64
                            else f"hj_abs_i64({v})")
                return (f"{node.func.attr}(" +
                        ", ".join(self.expr(a) for a in node.args) + ")")
            if not isinstance(node.func, ast.Name):
                raise UnsupportedError("standalone: unsupported call")
            name = node.func.id
            if name == self.f.name:
                return "hj_cpu_kernel(" + ", ".join(
                    self.expr(a) for a in node.args) + ")"
            if name == "abs":
                v = self.expr(node.args[0])
                return (f"fabs({v})" if self.etype(node.args[0]) == F64
                        else f"hj_abs_i64({v})")
            if name == "float":
                return f"((double)({self.expr(node.args[0])}))"
            if name == "int":
                return f"((int64_t)({self.expr(node.args[0])}))"
            raise UnsupportedError(f"standalone: unsupported call {name!r}")
        raise UnsupportedError(
            f"standalone: unsupported expression {type(node).__name__}")


class _ReturnToOutput(ast.NodeTransformer):
    def __init__(self, output_name):
        self.output_name = output_name

    def visit_Return(self, node):
        if node.value is None:
            raise UnsupportedError(
                "standalone CUDA export does not support bare return")
        store = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id=self.output_name, ctx=ast.Load()),
                slice=ast.Constant(value=0), ctx=ast.Store())],
            value=self.visit(node.value))
        ret = ast.Return(value=ast.Constant(value=0))
        return [ast.copy_location(store, node), ast.copy_location(ret, node)]


def _cuda_ptx(fn_ast, sig, ret_type, fastmath=False,
              reduce_reassoc=False, arch=None):
    """Build a one-thread CUDA entry that writes the scalar return value."""
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == fn_ast.name for n in ast.walk(fn_ast)):
        raise UnsupportedError(
            "standalone CUDA export does not support recursive kernels")
    from ..typeinfer import GPU_INTRINSICS
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id in GPU_INTRINSICS for n in ast.walk(fn_ast)):
        raise UnsupportedError(
            "standalone scalar executables cannot use GPU thread intrinsics")

    tree = copy.deepcopy(fn_ast)
    tree.name = "__hanajit_cuda_" + fn_ast.name
    out_name = "__hj_standalone_out"
    tree.args.args.append(ast.arg(arg=out_name))
    tree = _ReturnToOutput(out_name).visit(tree)
    ast.fix_missing_locations(tree)

    names = [a.arg for a in tree.args.args]
    out_ty = PF64 if ret_type == F64 else PI64
    arg_types = dict(zip(names, tuple(sig) + (out_ty,)))
    var_types, cuda_ret = TypeInferencer(tree, arg_types).run()
    module = CodeGen(tree, arg_types, var_types, cuda_ret,
                     fastmath=fastmath, reduce_reassoc=reduce_reassoc,
                     gpu="cuda").generate()
    from . import gpu
    text, native = gpu.emit(module, tree.name, "cuda", cpu=arch)
    if not native:
        raise UnsupportedError(
            "CUDA PTX emission is unavailable in this llvmlite build")
    return tree.name, text


def _ptx_array(ptx):
    data = ptx.encode("utf-8") + b"\0"
    rows = []
    for i in range(0, len(data), 12):
        rows.append("    " + ", ".join(f"0x{x:02x}" for x in data[i:i + 12]))
    return ",\n".join(rows)


def _parse_lines(sig):
    out = []
    for i, t in enumerate(sig):
        n = f"a{i}"
        if t == I64:
            out += [f"    int64_t {n};",
                    f"    if (!hj_parse_i64(argv[{i + 1}], &{n})) "
                    f"return hj_bad_arg({i + 1}, argv[{i + 1}]);"]
        elif t == F64:
            out += [f"    double {n};",
                    f"    if (!hj_parse_f64(argv[{i + 1}], &{n})) "
                    f"return hj_bad_arg({i + 1}, argv[{i + 1}]);"]
        else:
            out += [f"    bool {n};",
                    f"    if (!hj_parse_bool(argv[{i + 1}], &{n})) "
                    f"return hj_bad_arg({i + 1}, argv[{i + 1}]);"]
    return "\n".join(out)


def _cuda_source(kernel_name, ptx, sig, ret_type):
    args_decl = ", ".join(f"{_CTY[t]} a{i}" for i, t in enumerate(sig))
    if args_decl:
        args_decl += ", "
    params = ", ".join([f"&a{i}" for i in range(len(sig))] + ["&dout"])
    # The CUDA lowering writes booleans through an i64 pointer.  Keep that
    # device ABI independent of C's one-byte bool representation.
    if ret_type == BOOL:
        result_decl = "    int64_t hj_cuda_bool_result = 0;"
        result_size = "sizeof(hj_cuda_bool_result)"
        result_copy = "&hj_cuda_bool_result"
        result_finish = "    *out = hj_cuda_bool_result != 0;"
    else:
        result_decl = ""
        result_size = "sizeof(*out)"
        result_copy = "out"
        result_finish = ""
    return f"""
#ifdef _WIN32
#  include <windows.h>
#  define HJ_CUDAAPI __stdcall
static void *hj_open_cuda(void) {{ return (void *)LoadLibraryA("nvcuda.dll"); }}
static void *hj_cuda_sym(void *h, const char *n) {{
    return (void *)(uintptr_t)GetProcAddress((HMODULE)h, n);
}}
#else
#  include <dlfcn.h>
#  define HJ_CUDAAPI
static void *hj_open_cuda(void) {{ return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); }}
static void *hj_cuda_sym(void *h, const char *n) {{ return dlsym(h, n); }}
#endif

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef uint64_t CUdeviceptr;
typedef CUresult (HJ_CUDAAPI *hj_cuInit_t)(unsigned int);
typedef CUresult (HJ_CUDAAPI *hj_cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (HJ_CUDAAPI *hj_cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef CUresult (HJ_CUDAAPI *hj_cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (HJ_CUDAAPI *hj_cuModuleLoadDataEx_t)(CUmodule *, const void *, unsigned int, void *, void *);
typedef CUresult (HJ_CUDAAPI *hj_cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (HJ_CUDAAPI *hj_cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (HJ_CUDAAPI *hj_cuMemFree_t)(CUdeviceptr);
typedef CUresult (HJ_CUDAAPI *hj_cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **);
typedef CUresult (HJ_CUDAAPI *hj_cuCtxSynchronize_t)(void);
typedef CUresult (HJ_CUDAAPI *hj_cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);

static const unsigned char hj_embedded_ptx[] = {{
{_ptx_array(ptx)}
}};

#define HJ_LOAD(var, type, symbol) do {{ \
    *(void **)(&(var)) = hj_cuda_sym(lib, symbol); \
    if (!(var)) goto done; \
}} while (0)

static int hj_cuda_run({args_decl}{_CTY[ret_type]} *out) {{
    int ok = 0;
    void *lib = hj_open_cuda();
    CUdevice dev = 0;
    CUcontext ctx = 0;
    CUmodule mod = 0;
    CUfunction fn = 0;
    CUdeviceptr dout = 0;
{result_decl}
    hj_cuInit_t cuInit = 0;
    hj_cuDeviceGet_t cuDeviceGet = 0;
    hj_cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = 0;
    hj_cuCtxSetCurrent_t cuCtxSetCurrent = 0;
    hj_cuModuleLoadDataEx_t cuModuleLoadDataEx = 0;
    hj_cuModuleGetFunction_t cuModuleGetFunction = 0;
    hj_cuMemAlloc_t cuMemAlloc = 0;
    hj_cuMemFree_t cuMemFree = 0;
    hj_cuLaunchKernel_t cuLaunchKernel = 0;
    hj_cuCtxSynchronize_t cuCtxSynchronize = 0;
    hj_cuMemcpyDtoH_t cuMemcpyDtoH = 0;
    if (!lib) return 0;
    HJ_LOAD(cuInit, hj_cuInit_t, "cuInit");
    HJ_LOAD(cuDeviceGet, hj_cuDeviceGet_t, "cuDeviceGet");
    HJ_LOAD(cuDevicePrimaryCtxRetain, hj_cuDevicePrimaryCtxRetain_t, "cuDevicePrimaryCtxRetain");
    HJ_LOAD(cuCtxSetCurrent, hj_cuCtxSetCurrent_t, "cuCtxSetCurrent");
    HJ_LOAD(cuModuleLoadDataEx, hj_cuModuleLoadDataEx_t, "cuModuleLoadDataEx");
    HJ_LOAD(cuModuleGetFunction, hj_cuModuleGetFunction_t, "cuModuleGetFunction");
    HJ_LOAD(cuMemAlloc, hj_cuMemAlloc_t, "cuMemAlloc_v2");
    HJ_LOAD(cuMemFree, hj_cuMemFree_t, "cuMemFree_v2");
    HJ_LOAD(cuLaunchKernel, hj_cuLaunchKernel_t, "cuLaunchKernel");
    HJ_LOAD(cuCtxSynchronize, hj_cuCtxSynchronize_t, "cuCtxSynchronize");
    HJ_LOAD(cuMemcpyDtoH, hj_cuMemcpyDtoH_t, "cuMemcpyDtoH_v2");
    if (cuInit(0) || cuDeviceGet(&dev, 0) ||
            cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) goto done;
    if (cuModuleLoadDataEx(&mod, hj_embedded_ptx, 0, 0, 0) ||
            cuModuleGetFunction(&fn, mod, "{kernel_name}") ||
            cuMemAlloc(&dout, {result_size})) goto done;
    {{ void *params[] = {{ {params} }};
      if (cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, params, 0) ||
              cuCtxSynchronize() ||
              cuMemcpyDtoH({result_copy}, dout, {result_size}))
          goto done; }}
{result_finish}
    ok = 1;
done:
    if (dout && cuMemFree) cuMemFree(dout);
    return ok;
}}
"""


def _c_source(kernel_source, sig, ret_type, cuda_mode,
              kernel_name=None, ptx=None):
    parse = _parse_lines(sig)
    callargs = ", ".join(f"a{i}" for i in range(len(sig)))
    usage = " ".join({I64: "<i64>", F64: "<f64>", BOOL: "<bool>"}[t]
                     for t in sig)
    if ret_type == F64:
        print_result = 'printf("%.17g\\n", result);'
    elif ret_type == BOOL:
        print_result = 'printf("%d\\n", result ? 1 : 0);'
    else:
        print_result = 'printf("%" PRId64 "\\n", result);'

    cuda_part = ""
    run = f"    {_CTY[ret_type]} result = hj_cpu_kernel({callargs});"
    if cuda_mode != "off":
        cuda_part = _cuda_source(kernel_name, ptx, sig, ret_type)
        cuda_args = (callargs + ", ") if callargs else ""
        required = "1" if cuda_mode == "required" else "0"
        run = f"""    {_CTY[ret_type]} result;
    if (!hj_cuda_run({cuda_args}&result)) {{
        if ({required}) {{
            fputs("CUDA execution required but the NVIDIA driver or kernel is unavailable\\n", stderr);
            return 70;
        }}
        result = hj_cpu_kernel({callargs});
    }}"""

    return f"""/* Generated by hanajit.export_executable. */
#define _USE_MATH_DEFINES
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int64_t hj_add_i64(int64_t a, int64_t b) {{
    return (int64_t)((uint64_t)a + (uint64_t)b);
}}
static int64_t hj_sub_i64(int64_t a, int64_t b) {{
    return (int64_t)((uint64_t)a - (uint64_t)b);
}}
static int64_t hj_mul_i64(int64_t a, int64_t b) {{
    return (int64_t)((uint64_t)a * (uint64_t)b);
}}
static int64_t hj_neg_i64(int64_t a) {{
    return (int64_t)(UINT64_C(0) - (uint64_t)a);
}}
static int64_t hj_abs_i64(int64_t a) {{
    return a < 0 ? hj_neg_i64(a) : a;
}}
static int64_t hj_floor_div_i64(int64_t a, int64_t b) {{
    int64_t q = a / b, r = a % b;
    return ((a ^ b) < 0 && r != 0) ? hj_sub_i64(q, 1) : q;
}}
static int64_t hj_mod_i64(int64_t a, int64_t b) {{
    int64_t r = a % b;
    return ((a ^ b) < 0 && r != 0) ? hj_add_i64(r, b) : r;
}}
static double hj_mod_f64(double a, double b) {{
    double r = fmod(a, b);
    return ((r < 0.0) != (b < 0.0) && r != 0.0) ? r + b : r;
}}
static bool hj_truth_f64(double a) {{ return !isnan(a) && a != 0.0; }}
static bool hj_fcmp_eq(double a, double b) {{ return !isnan(a) && !isnan(b) && a == b; }}
static bool hj_fcmp_ne(double a, double b) {{ return !isnan(a) && !isnan(b) && a != b; }}
static bool hj_fcmp_lt(double a, double b) {{ return !isnan(a) && !isnan(b) && a < b; }}
static bool hj_fcmp_le(double a, double b) {{ return !isnan(a) && !isnan(b) && a <= b; }}
static bool hj_fcmp_gt(double a, double b) {{ return !isnan(a) && !isnan(b) && a > b; }}
static bool hj_fcmp_ge(double a, double b) {{ return !isnan(a) && !isnan(b) && a >= b; }}

{kernel_source}

static int hj_parse_i64(const char *s, int64_t *out) {{
    char *end = 0; long long v; errno = 0;
    v = strtoll(s, &end, 10);
    if (errno || !end || end == s || *end) return 0;
    *out = (int64_t)v; return 1;
}}
static int hj_parse_f64(const char *s, double *out) {{
    char *end = 0; double v; errno = 0;
    v = strtod(s, &end);
    if (errno || !end || end == s || *end) return 0;
    *out = v; return 1;
}}
static int hj_parse_bool(const char *s, bool *out) {{
    int64_t v;
    if (!hj_parse_i64(s, &v) || (v != 0 && v != 1)) return 0;
    *out = v != 0; return 1;
}}
static int hj_bad_arg(int n, const char *s) {{
    fprintf(stderr, "invalid argument %d: %s\\n", n, s);
    return 64;
}}
{cuda_part}
int main(int argc, char **argv) {{
    if (argc != {len(sig) + 1}) {{
        fprintf(stderr, "usage: %s {usage}\\n", argv[0]);
        return 64;
    }}
{parse}
{run}
    {print_result}
    return 0;
}}
"""


def _msvc_install():
    """Return (cl.exe, vcvars64.bat), including non-developer shells."""
    roots = []
    vswhere = Path(os.environ.get(
        "ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            r = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                roots.append(Path(r.stdout.strip().splitlines()[-1]))
        except Exception:
            pass
    for edition in ("BuildTools", "Community", "Professional",
                    "Enterprise"):
        roots.append(Path(os.environ.get("ProgramFiles",
                                         r"C:\Program Files")) /
                     "Microsoft Visual Studio" / "2022" / edition)
    for root in roots:
        vcvars = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        tools = root / "VC" / "Tools" / "MSVC"
        if not vcvars.is_file() or not tools.is_dir():
            continue
        matches = sorted(tools.glob("*/bin/Hostx64/x64/cl.exe"), reverse=True)
        if matches:
            return matches[0], vcvars
    return None


def _msvc_environment(vcvars):
    import tempfile
    fd, env_script = tempfile.mkstemp(prefix="hanajit_vcvars_",
                                      suffix=".bat")
    os.close(fd)
    try:
        Path(env_script).write_text(
            f'@call "{vcvars}" >nul\r\n@set\r\n', encoding="utf-8")
        r = subprocess.run(
            ["cmd.exe", "/d", "/c", env_script],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        if r.returncode != 0:
            return None
        env = dict(os.environ)
        for line in r.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
        return env
    except Exception:
        return None
    finally:
        try:
            Path(env_script).unlink()
        except OSError:
            pass


def _compiler():
    return _compiler_cached(os.environ.get("HANAJIT_CC"))


@functools.lru_cache(maxsize=4)
def _compiler_cached(override):
    if override:
        argv = shlex.split(override, posix=os.name != "nt")
        return {"argv": argv,
                "msvc": Path(argv[0]).stem.lower() == "cl",
                "env": None, "vcvars": None}
    candidates = (["clang", "gcc", "cc", "cl"] if os.name != "nt"
                  else ["clang", "gcc", "cl"])
    for c in candidates:
        p = shutil.which(c)
        if p:
            return {"argv": [p], "msvc": c == "cl",
                    "env": None, "vcvars": None}
    if os.name == "nt":
        found = _msvc_install()
        if found:
            cl, vcvars = found
            env = _msvc_environment(vcvars)
            if env is not None:
                return {"argv": [str(cl)], "msvc": True,
                        "env": env, "vcvars": str(vcvars)}
    return None


def _build_commands(compiler, is_msvc, src, obj, exe, with_cuda):
    if is_msvc:
        compile_ = compiler + ["/nologo", "/O2", "/MT", "/std:c11",
                               "/c", str(src), "/Fo:" + str(obj)]
        link = compiler + ["/nologo", "/O2", "/MT", str(obj),
                           "/Fe:" + str(exe)]
        return [compile_, link]
    compile_ = compiler + ["-std=c11", "-O3", "-fwrapv",
                           "-march=x86-64", "-c", str(src), "-o", str(obj)]
    link = compiler + [str(obj), "-o", str(exe)]
    if os.name != "nt":
        link.append("-lm")
        if with_cuda:
            link.append("-ldl")
    return [compile_, link]


def _write_build_script(path, commands, vcvars=None):
    if os.name == "nt":
        lines = "\r\n".join(subprocess.list2cmdline(c) for c in commands)
        pre = f'call "{vcvars}" >nul\r\n' if vcvars else ""
        path.write_text("@echo off\r\n" + pre + lines + "\r\n",
                        encoding="utf-8", newline="")
    else:
        lines = "\n".join(shlex.join(c) for c in commands)
        path.write_text("#!/bin/sh\nset -eu\n" + lines + "\n",
                        encoding="utf-8")
        path.chmod(0o755)


def export_executable(module, fn_ast, arg_types, var_types, sig, ret_type,
                      output, *,
                      cuda="off", fastmath=False, reduce_reassoc=False,
                      cuda_arch=None):
    """Write and, when a host compiler exists, link a standalone CLI.

    Returns ``ExecutableExport(executable, object, source, build, ptx)``.
    ``executable`` and ``object`` are None when no C toolchain is available;
    the self-contained C source and exact build script are still written.
    """
    if not _x86_64():
        raise UnsupportedError(
            f"standalone executables require x86-64, got {platform.machine()}")
    mode = _normalize_mode(cuda)
    if any(t in POINTER_ELEM or t in ARRAY_ELEM for t in sig):
        raise UnsupportedError(
            "standalone CLI export currently supports scalar signatures only")
    if any(t not in _CTY for t in sig) or ret_type not in _CTY:
        raise UnsupportedError(
            "standalone CLI export supports i64/f64/bool only")
    if not _C_IDENT.match(fn_ast.name):
        raise UnsupportedError(
            "standalone executable function names must be ASCII C identifiers")

    exe, obj, src, build, ptx_path = _paths(output)
    # ``module`` has already been generated by _typed_kernel, which is the
    # authoritative validation that the function belongs to HanaJit's static
    # subset.  Native object format is delegated to the platform C compiler:
    # some llvmlite distributions intentionally expose MCJIT's ELF cache
    # format even on Windows, which is not a COFF link input.
    del module
    kernel_source = _CScalarGen(
        fn_ast, arg_types, var_types, ret_type).generate()
    kernel_name = ptx_text = None
    if mode != "off":
        kernel_name, ptx_text = _cuda_ptx(
            fn_ast, sig, ret_type, fastmath=fastmath,
            reduce_reassoc=reduce_reassoc, arch=cuda_arch)
        ptx_path.write_text(ptx_text, encoding="utf-8")
    else:
        ptx_path = None
    src.write_text(_c_source(kernel_source, sig, ret_type, mode,
                             kernel_name, ptx_text), encoding="utf-8")

    compiler = _compiler()
    script_compiler = (compiler["argv"] if compiler else ["clang"])
    is_msvc = compiler["msvc"] if compiler else False
    commands = _build_commands(script_compiler, is_msvc, src, obj, exe,
                               mode != "off")
    _write_build_script(build, commands,
                        compiler.get("vcvars") if compiler else None)

    linked = None
    if compiler:
        try:
            failure = None
            for command in commands:
                r = subprocess.run(command, capture_output=True, text=True,
                                   timeout=120, env=compiler.get("env"))
                if r.returncode != 0:
                    failure = r
                    break
            if failure is None and exe.exists():
                linked = str(exe)
            else:
                r = failure
                warnings.warn(
                    "hanajit: standalone linker failed; object/source/build "
                    "artifacts were written. " + (r.stderr or r.stdout)[-500:])
        except Exception as e:
            warnings.warn(
                f"hanajit: standalone linker failed ({e}); run {build}")
    else:
        warnings.warn(
            "hanajit: no C compiler found; standalone source/build artifacts "
            f"were written. Install clang or set HANAJIT_CC, then "
            f"run {build}")
    return ExecutableExport(linked, str(obj) if obj.exists() else None,
                            str(src), str(build),
                            str(ptx_path) if ptx_path else None)
