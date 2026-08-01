"""SPIR-V binary generator: typed Python AST -> SPIR-V module.

LLVM's SPIR-V backends cannot produce launchable binaries from our IR
(no object writer for the OpenCL flavor; the shader flavor hard-aborts
on buffer indexing), so kernels destined for Level Zero or Vulkan are
generated directly from the typed AST — the same source of truth the
metal and fpga transpilers use. The compile subset is small enough that
direct generation is exact.

Two flavors, one generator:

- flavor="opencl": Physical64/OpenCL memory model, OpEntryPoint Kernel,
  pointer kernel arguments (CrossWorkgroup), builtin workitem variables.
  Consumed by Level Zero (zeModuleCreate, ZE_MODULE_FORMAT_IL_SPIRV).
- flavor="vulkan": Logical/GLSL450, OpEntryPoint GLCompute + LocalSize,
  pointer arguments become DescriptorSet-0 storage buffers (binding =
  argument position), scalar arguments live in one std430 push-constant
  block, thread builtins are uvec3 Inputs widened to i64.

Semantics match the CPU backend: locals are Function-storage variables
(no SSA phis needed), integer / and % follow Python floor semantics
including negative operands, `/` is true division, bool ops are
non-short-circuit — identical to codegen.py's documented behavior.

Every opcode and enum below was extracted from the official Khronos
machine-readable grammars (spirv.core / OpenCL.std.100 / GLSL.std.450).

The generated modules validate against the drivers they target; they are
deliberately unoptimized (the driver compiler optimizes at module load).
"""
import ast
import struct

from ..errors import UnsupportedError
from ..typeinfer import (I64, F64, BOOL, PF64, PI64, GPU_INTRINSICS,
                         MATH_FNS, MATH_MODULES)

MAGIC = 0x07230203

# ---- core opcodes (spirv.core.grammar.json) ----
OpName_ = 5
OpExtInstImport = 11
OpExtInst = 12
OpMemoryModel = 14
OpEntryPoint = 15
OpExecutionMode = 16
OpCapability = 17
OpTypeVoid = 19
OpTypeBool = 20
OpTypeInt = 21
OpTypeFloat = 22
OpTypeVector = 23
OpTypeRuntimeArray = 29
OpTypeStruct = 30
OpTypePointer = 32
OpTypeFunction = 33
OpConstantTrue = 41
OpConstantFalse = 42
OpConstant = 43
OpConstantComposite = 44
OpFunction = 54
OpFunctionParameter = 55
OpFunctionEnd = 56
OpVariable = 59
OpLoad = 61
OpStore = 62
OpAccessChain = 65
OpInBoundsPtrAccessChain = 70
OpDecorate = 71
OpMemberDecorate = 72
OpCompositeExtract = 81
OpConvertFToS = 110
OpConvertSToF = 111
OpUConvert = 113
OpSConvert = 114
OpFConvert = 115
OpSNegate = 126
OpFNegate = 127
OpIAdd, OpFAdd, OpISub, OpFSub = 128, 129, 130, 131
OpIMul, OpFMul, OpSDiv, OpFDiv = 132, 133, 135, 136
OpSRem = 138
OpFRem = 140
OpLogicalOr, OpLogicalAnd, OpLogicalNot = 166, 167, 168
OpSelect = 169
OpIEqual, OpINotEqual = 170, 171
OpSGreaterThan, OpSGreaterThanEqual = 173, 175
OpSLessThan, OpSLessThanEqual = 177, 179
OpFOrdEqual, OpFOrdNotEqual = 180, 182
OpFOrdLessThan, OpFOrdGreaterThan = 184, 186
OpFOrdLessThanEqual, OpFOrdGreaterThanEqual = 188, 190
OpShiftRightArithmetic, OpShiftLeftLogical = 195, 196
OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd, OpNot = 197, 198, 199, 200
OpControlBarrier = 224
OpAtomicCompareExchange = 230
OpAtomicIAdd = 234
OpLoopMerge, OpSelectionMerge, OpLabel = 246, 247, 248
OpBranch, OpBranchConditional = 249, 250
OpReturn = 253
OpTypeArray = 28
OpBitcast = 124
OpExtension = 10
OpAtomicFAddEXT = 6035

# ---- enums ----
CAP_SHADER, CAP_ADDRESSES, CAP_LINKAGE = 1, 4, 5
CAP_KERNEL, CAP_FLOAT64, CAP_INT64 = 6, 10, 11
CAP_INT64_ATOMICS = 12
CAP_ATOMIC_FLOAT64_ADD_EXT = 6034
EXT_ATOMIC_FLOAT_ADD = "SPV_EXT_shader_atomic_float_add"
SCOPE_DEVICE = 1
AM_LOGICAL, AM_PHYSICAL64 = 0, 2
MM_GLSL450, MM_OPENCL = 1, 2
EM_GLCOMPUTE, EM_KERNEL = 5, 6
MODE_LOCALSIZE = 17
SC_INPUT, SC_FUNCTION, SC_CROSSWORKGROUP = 1, 7, 5
SC_PUSHCONSTANT, SC_STORAGEBUFFER, SC_WORKGROUP = 9, 12, 4
SCOPE_WORKGROUP = 2
# AcquireRelease | WorkgroupMemory
SEM_WG_ACQREL = 0x0008 | 0x0100
DEC_BLOCK, DEC_ARRAYSTRIDE, DEC_BUILTIN = 2, 6, 11
DEC_BINDING, DEC_DESCRIPTORSET, DEC_OFFSET = 33, 34, 35
BI_WORKGROUPSIZE, BI_WORKGROUPID, BI_LOCALINVOCATIONID = 25, 26, 27

# extended-instruction opcodes per math function, per flavor
CL_MATH = {"sqrt": 61, "exp": 19, "log": 37, "sin": 57, "cos": 14,
           "tan": 62, "floor": 25, "ceil": 12, "pow": 48, "fabs": 23,
           "atan2": 7}
GL_MATH = {"sqrt": 31, "exp": 27, "log": 28, "sin": 13, "cos": 14,
           "tan": 15, "floor": 8, "ceil": 9, "pow": 26, "fabs": 4,
           "atan2": 25}


def _words_for_string(s):
    b = s.encode("utf-8") + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


class SpirvGen:
    def __init__(self, func_ast, arg_types, var_types, ret_type,
                 flavor, local_size=(64, 1, 1)):
        if flavor not in ("opencl", "vulkan"):
            raise ValueError(f"unknown SPIR-V flavor {flavor!r}")
        self.f = func_ast
        self.arg_types = arg_types
        self.var_types = var_types
        self.flavor = flavor
        self.local_size = local_size
        self._id = 0
        # module sections, assembled in spec order at the end
        self.sec_ext = []        # OpExtInstImport
        self.sec_entry = []      # OpEntryPoint
        self.sec_modes = []      # OpExecutionMode
        self.sec_dec = []        # decorations
        self.sec_types = []      # types/constants/global variables
        self.body = []           # function instructions
        self.types = {}          # cache: key -> id
        self.consts = {}
        self.vars = {}           # python name -> (function var id, abstract ty)
        self.bufs = {}           # pointer-arg name -> handle (flavor-specific)
        self.builtin_vars = {}   # builtin enum -> variable id
        self.interface = []
        self.loop_stack = []     # (continue_label, merge_label)
        self.terminated = False
        self.ext_id = None
        self.extra_caps = set()  # capabilities required by used features
        self.extensions = set()  # OpExtension strings
        self._var_pos = None     # insertion point for late Function vars

    def nid(self):
        self._id += 1
        return self._id

    def ins(self, sec, opcode, *operands):
        sec.append(((len(operands) + 1) << 16 | opcode, *operands))

    # ------------------------------ types ------------------------------
    def ty(self, key):
        # SPIR-V forbids duplicate non-aggregate types: "u32" is the same
        # OpTypeInt 32 0 as "i32" (signedness 0 = no semantics), so both
        # keys must resolve to one id.
        if key == "u32":
            key = "i32"
        t = self.types.get(key)
        if t is not None:
            return t
        i = self.nid()
        if key == "void":
            self.ins(self.sec_types, OpTypeVoid, i)
        elif key == "bool":
            self.ins(self.sec_types, OpTypeBool, i)
        elif key == "i32":
            self.ins(self.sec_types, OpTypeInt, i, 32, 0)
        elif key == "i64":
            self.ins(self.sec_types, OpTypeInt, i, 64, 0)
        elif key == "f32":
            self.ins(self.sec_types, OpTypeFloat, i, 32)
        elif key == "f64":
            self.ins(self.sec_types, OpTypeFloat, i, 64)
        elif key == "v3u32":
            self.ins(self.sec_types, OpTypeVector, i, self.ty("u32"), 3)
        elif key == "v3i64":
            self.ins(self.sec_types, OpTypeVector, i, self.ty("i64"), 3)
        elif isinstance(key, tuple) and key[0] == "ptr":
            _, sc, inner = key
            self.ins(self.sec_types, OpTypePointer, i, sc, self.ty(inner))
        elif isinstance(key, tuple) and key[0] == "rtarr":
            self.ins(self.sec_types, OpTypeRuntimeArray, i, self.ty(key[1]))
            self.ins(self.sec_dec, OpDecorate, i, DEC_ARRAYSTRIDE, 8)
        elif isinstance(key, tuple) and key[0] == "arr":
            # fixed-size array (workgroup memory); no ArrayStride — that
            # decoration is only for the buffer-interface storage classes
            _, ety, n = key
            self.ins(self.sec_types, OpTypeArray, i, self.ty(ety),
                     self.const(("i32", n)))
        else:
            raise UnsupportedError(f"spirv: unknown type key {key!r}")
        self.types[key] = i
        return i

    ABSTY = {I64: "i64", F64: "f64", BOOL: "bool"}

    def const(self, key):
        c = self.consts.get(key)
        if c is not None:
            return c
        kind, v = key
        i = self.nid()
        if kind == "i32":
            self.ins(self.sec_types, OpConstant, self.ty("i32"), i,
                     v & 0xFFFFFFFF)
        elif kind == "u32":
            self.ins(self.sec_types, OpConstant, self.ty("u32"), i,
                     v & 0xFFFFFFFF)
        elif kind == "i64":
            lo = v & 0xFFFFFFFF
            hi = (v >> 32) & 0xFFFFFFFF
            self.ins(self.sec_types, OpConstant, self.ty("i64"), i, lo, hi)
        elif kind == "f64":
            bits = struct.unpack("<Q", struct.pack("<d", v))[0]
            self.ins(self.sec_types, OpConstant, self.ty("f64"), i,
                     bits & 0xFFFFFFFF, bits >> 32)
        elif kind == "true":
            self.ins(self.sec_types, OpConstantTrue, self.ty("bool"), i)
        elif kind == "false":
            self.ins(self.sec_types, OpConstantFalse, self.ty("bool"), i)
        else:
            raise UnsupportedError(f"spirv: unknown const {key!r}")
        self.consts[key] = i
        return i

    # --------------------------- module frame ---------------------------
    def generate(self):
        header_caps = ([CAP_INT64, CAP_FLOAT64]
                       + ([CAP_SHADER] if self.flavor == "vulkan"
                          else [CAP_ADDRESSES, CAP_KERNEL, CAP_LINKAGE]))
        self.ext_id = self.nid()
        ext_name = ("GLSL.std.450" if self.flavor == "vulkan"
                    else "OpenCL.std")
        self.ins(self.sec_ext, OpExtInstImport, self.ext_id,
                 *_words_for_string(ext_name))
        self.math_table = (GL_MATH if self.flavor == "vulkan" else CL_MATH)

        fn_id = self._build_function()

        model = EM_GLCOMPUTE if self.flavor == "vulkan" else EM_KERNEL
        self.sec_entry.insert(0, (
            (len([model, fn_id] + _words_for_string(self.f.name)
                 + self.interface) + 1) << 16 | OpEntryPoint,
            model, fn_id, *_words_for_string(self.f.name), *self.interface))
        if self.flavor == "vulkan":
            lx, ly, lz = self.local_size
            self.ins(self.sec_modes, OpExecutionMode, fn_id,
                     MODE_LOCALSIZE, lx, ly, lz)

        words = [MAGIC,
                 0x00010300 if self.flavor == "vulkan" else 0x00010000,
                 0,                     # generator tool id (unregistered)
                 self._id + 1, 0]
        for cap in header_caps + sorted(self.extra_caps):
            words += [(2 << 16) | OpCapability, cap]
        for ext in sorted(self.extensions):
            ws = _words_for_string(ext)
            words += [((len(ws) + 1) << 16) | OpExtension, *ws]
        for sec in (self.sec_ext,):
            for inst in sec:
                words += list(inst)
        words += [(3 << 16) | OpMemoryModel,
                  AM_LOGICAL if self.flavor == "vulkan" else AM_PHYSICAL64,
                  MM_GLSL450 if self.flavor == "vulkan" else MM_OPENCL]
        for sec in (self.sec_entry, self.sec_modes, self.sec_dec,
                    self.sec_types, self.body):
            for inst in sec:
                words += list(inst)
        return struct.pack(f"<{len(words)}I", *words)

    # ----------------------- kernel ABI plumbing -----------------------
    def _builtin(self, builtin, comp_key):
        """Module-level Input builtin variable (vec3), created on demand."""
        v = self.builtin_vars.get(builtin)
        if v is None:
            pty = self.ty(("ptr", SC_INPUT, comp_key))
            v = self.nid()
            self.ins(self.sec_types, OpVariable, pty, v, SC_INPUT)
            self.ins(self.sec_dec, OpDecorate, v, DEC_BUILTIN, builtin)
            self.builtin_vars[builtin] = v
            self.interface.append(v)
        return v

    def _thread_query(self, builtin, axis=0):
        """Load builtin vec3, extract one component, return as i64."""
        if self.flavor == "vulkan":
            var = self._builtin(builtin, "v3u32")
            vec = self.nid()
            self.ins(self.body, OpLoad, self.ty("v3u32"), vec, var)
            x = self.nid()
            self.ins(self.body, OpCompositeExtract, self.ty("u32"), x,
                     vec, axis)
            w = self.nid()
            self.ins(self.body, OpUConvert, self.ty("i64"), w, x)
            return w
        var = self._builtin(builtin, "v3i64")
        vec = self.nid()
        self.ins(self.body, OpLoad, self.ty("v3i64"), vec, var)
        x = self.nid()
        self.ins(self.body, OpCompositeExtract, self.ty("i64"), x, vec,
                 axis)
        return x

    def _build_function(self):
        void = self.ty("void")
        scalars = [(a.arg, self.arg_types[a.arg]) for a in self.f.args.args
                   if self.arg_types[a.arg] in (I64, F64)]
        pointers = [(a.arg, self.arg_types[a.arg]) for a in self.f.args.args
                    if self.arg_types[a.arg] in (PF64, PI64)]
        bad = [a.arg for a in self.f.args.args
               if self.arg_types[a.arg] not in (I64, F64, PF64, PI64)]
        if bad:
            raise UnsupportedError(
                f"spirv: unsupported kernel argument types for {bad} — "
                "GPU kernels take i64/f64 scalars and f64*/i64* pointers")

        param_ids = []
        if self.flavor == "opencl":
            # kernel signature carries every argument directly
            param_keys = []
            for name, t in [(a.arg, self.arg_types[a.arg])
                            for a in self.f.args.args]:
                if t in (PF64, PI64):
                    param_keys.append(("ptr", SC_CROSSWORKGROUP,
                                       "f64" if t == PF64 else "i64"))
                else:
                    param_keys.append(self.ABSTY[t])
            fnty = self.nid()
            self.ins(self.sec_types, OpTypeFunction, fnty, void,
                     *[self.ty(k) for k in param_keys])
        else:
            # vulkan: void(); buffers + push constants carry the arguments
            fnty = self.nid()
            self.ins(self.sec_types, OpTypeFunction, fnty, void)
            for binding, (name, t) in enumerate(pointers):
                ety = "f64" if t == PF64 else "i64"
                rt = self.ty(("rtarr", ety))
                st = self.nid()
                self.ins(self.sec_types, OpTypeStruct, st, rt)
                self.ins(self.sec_dec, OpDecorate, st, DEC_BLOCK)
                self.ins(self.sec_dec, OpMemberDecorate, st, 0,
                         DEC_OFFSET, 0)
                pty = self.nid()
                self.ins(self.sec_types, OpTypePointer, pty,
                         SC_STORAGEBUFFER, st)
                var = self.nid()
                self.ins(self.sec_types, OpVariable, pty, var,
                         SC_STORAGEBUFFER)
                self.ins(self.sec_dec, OpDecorate, var,
                         DEC_DESCRIPTORSET, 0)
                self.ins(self.sec_dec, OpDecorate, var, DEC_BINDING,
                         binding)
                self.bufs[name] = (var, ety, "arg")
            if scalars:
                st = self.nid()
                self.ins(self.sec_types, OpTypeStruct, st,
                         *[self.ty(self.ABSTY[t]) for _, t in scalars])
                self.ins(self.sec_dec, OpDecorate, st, DEC_BLOCK)
                for idx in range(len(scalars)):
                    self.ins(self.sec_dec, OpMemberDecorate, st, idx,
                             DEC_OFFSET, 8 * idx)
                pty = self.nid()
                self.ins(self.sec_types, OpTypePointer, pty,
                         SC_PUSHCONSTANT, st)
                self.push_var = self.nid()
                self.ins(self.sec_types, OpVariable, pty, self.push_var,
                         SC_PUSHCONSTANT)

        fn_id = self.nid()
        self.ins(self.body, OpFunction, void, fn_id, 0, fnty)
        if self.flavor == "opencl":
            for a in self.f.args.args:
                t = self.arg_types[a.arg]
                pid = self.nid()
                if t in (PF64, PI64):
                    key = ("ptr", SC_CROSSWORKGROUP,
                           "f64" if t == PF64 else "i64")
                    self.ins(self.body, OpFunctionParameter,
                             self.ty(key), pid)
                    self.bufs[a.arg] = (pid, "f64" if t == PF64 else "i64",
                                        "arg")
                else:
                    self.ins(self.body, OpFunctionParameter,
                             self.ty(self.ABSTY[t]), pid)
                param_ids.append((a.arg, t, pid))
        entry = self.nid()
        self.ins(self.body, OpLabel, entry)

        # Function-storage variables first (spec: they open the block):
        # one per scalar local AND per scalar parameter (mutable, like
        # the CPU backend's allocas).
        names = dict(self.var_types)
        for name, t in [(n, t) for n, t, _ in param_ids] \
                if self.flavor == "opencl" else []:
            names.setdefault(name, t)
        if self.flavor == "vulkan":
            for name, t in scalars:
                names.setdefault(name, t)
        for name in sorted(names):
            t = names[name]
            if t in (PF64, PI64):
                continue
            if t not in self.ABSTY:
                raise UnsupportedError(
                    f"spirv: unsupported local type {t} for {name!r}")
            pty = self.ty(("ptr", SC_FUNCTION, self.ABSTY[t]))
            vid = self.nid()
            self.ins(self.body, OpVariable, pty, vid, SC_FUNCTION)
            self.vars[name] = (vid, t)
        # late Function variables (atomic CAS scratch) insert here — the
        # spec requires all OpVariables to open the entry block
        self._var_pos = len(self.body)

        # initialize parameter variables
        if self.flavor == "opencl":
            for name, t, pid in param_ids:
                if t in (I64, F64):
                    self.ins(self.body, OpStore, self.vars[name][0], pid)
        else:
            for idx, (name, t) in enumerate(scalars):
                pty = self.ty(("ptr", SC_PUSHCONSTANT, self.ABSTY[t]))
                ac = self.nid()
                self.ins(self.body, OpAccessChain, pty, ac, self.push_var,
                         self.const(("i32", idx)))
                val = self.nid()
                self.ins(self.body, OpLoad, self.ty(self.ABSTY[t]),
                         val, ac)
                self.ins(self.body, OpStore, self.vars[name][0], val)

        for stmt in self.f.body:
            self.stmt(stmt)
        if not self.terminated:
            self.ins(self.body, OpReturn)
        self.ins(self.body, OpFunctionEnd)
        return fn_id

    # ---------------------------- statements ----------------------------
    def stmt(self, node):
        if self.terminated:
            return  # unreachable Python code after return/break/continue
        m = getattr(self, "s_" + type(node).__name__, None)
        if m is None:
            raise UnsupportedError(
                f"spirv: unsupported statement {type(node).__name__}")
        m(node)

    def _label(self, lid):
        self.ins(self.body, OpLabel, lid)
        self.terminated = False

    def _elem_ptr(self, name, index_id):
        """Pointer to element `index` of pointer-arg or shared array."""
        base, ety, kind = self.bufs[name]
        p = self.nid()
        if kind == "wg":            # workgroup array: direct index
            pty = self.ty(("ptr", SC_WORKGROUP, ety))
            self.ins(self.body, OpAccessChain, pty, p, base, index_id)
        elif self.flavor == "opencl":
            pty = self.ty(("ptr", SC_CROSSWORKGROUP, ety))
            self.ins(self.body, OpInBoundsPtrAccessChain, pty, p,
                     base, index_id)
        else:
            pty = self.ty(("ptr", SC_STORAGEBUFFER, ety))
            self.ins(self.body, OpAccessChain, pty, p, base,
                     self.const(("i32", 0)), index_id)
        return p, ety

    def s_Assign(self, node):
        t = node.targets[0]
        v = node.value
        if (isinstance(t, ast.Name) and isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id in ("shared_f64", "shared_i64")):
            # workgroup-shared array: module-scope Workgroup variable
            n = v.args[0].value
            ety = "f64" if v.func.id == "shared_f64" else "i64"
            pty = self.ty(("ptr", SC_WORKGROUP, ("arr", ety, n)))
            var = self.nid()
            self.ins(self.sec_types, OpVariable, pty, var, SC_WORKGROUP)
            self.bufs[t.id] = (var, ety, "wg")
            return
        if isinstance(t, ast.Subscript):
            idx, ity = self.expr(t.slice)
            if ity != I64:
                raise UnsupportedError("spirv: index must be integer")
            p, ety = self._elem_ptr(t.value.id, idx)
            v, vt = self.expr(node.value)
            v = self.cast(v, vt, F64 if ety == "f64" else I64)
            self.ins(self.body, OpStore, p, v)
        else:
            vid, ty = self.vars[t.id]
            v, vt = self.expr(node.value)
            self.ins(self.body, OpStore, vid, self.cast(v, vt, ty))

    def s_AugAssign(self, node):
        cur = ast.Name(id=node.target.id, ctx=ast.Load())
        binop = ast.BinOp(left=cur, op=node.op, right=node.value)
        self.s_Assign(ast.Assign(targets=[node.target], value=binop))

    def s_Return(self, node):
        if node.value is not None:
            self.expr(node.value)  # evaluated, then dropped: kernels are void
        self.ins(self.body, OpReturn)
        self.terminated = True

    def s_Expr(self, node):
        # side-effect intrinsics must emit; docstrings etc. are dropped
        v = node.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id in ("barrier", "atomic_add")):
            self.expr(v)

    def s_Pass(self, node):
        pass

    def s_Break(self, node):
        self.ins(self.body, OpBranch, self.loop_stack[-1][1])
        self.terminated = True

    def s_Continue(self, node):
        self.ins(self.body, OpBranch, self.loop_stack[-1][0])
        self.terminated = True

    def s_If(self, node):
        cond, ct = self.expr(node.test)
        cond = self.tobool(cond, ct)
        then_l, merge_l = self.nid(), self.nid()
        else_l = self.nid() if node.orelse else merge_l
        self.ins(self.body, OpSelectionMerge, merge_l, 0)
        self.ins(self.body, OpBranchConditional, cond, then_l, else_l)
        self._label(then_l)
        for s in node.body:
            self.stmt(s)
        if not self.terminated:
            self.ins(self.body, OpBranch, merge_l)
        if node.orelse:
            self._label(else_l)
            for s in node.orelse:
                self.stmt(s)
            if not self.terminated:
                self.ins(self.body, OpBranch, merge_l)
        self._label(merge_l)

    def _loop(self, cond_gen, body_stmts, continue_gen):
        header, check, body_l, cont, merge = (self.nid() for _ in range(5))
        self.ins(self.body, OpBranch, header)
        self._label(header)
        self.ins(self.body, OpLoopMerge, merge, cont, 0)
        self.ins(self.body, OpBranch, check)
        self._label(check)
        cond = cond_gen()
        self.ins(self.body, OpBranchConditional, cond, body_l, merge)
        self._label(body_l)
        self.loop_stack.append((cont, merge))
        for s in body_stmts:
            self.stmt(s)
        self.loop_stack.pop()
        if not self.terminated:
            self.ins(self.body, OpBranch, cont)
        self._label(cont)
        continue_gen()
        self.ins(self.body, OpBranch, header)
        self._label(merge)

    def s_While(self, node):
        def cond():
            c, ct = self.expr(node.test)
            return self.tobool(c, ct)
        self._loop(cond, node.body, lambda: None)

    def s_For(self, node):
        if not (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id in ("range", "prange")):
            raise UnsupportedError("spirv: for supports range(...) only")
        a = node.iter.args
        zero = self.const(("i64", 0))
        one = self.const(("i64", 1))
        if len(a) == 1:
            start, stop_node, step = zero, a[0], one
        elif len(a) == 2:
            start = self.casti(self.expr(a[0]))
            stop_node, step = a[1], one
        else:
            start = self.casti(self.expr(a[0]))
            stop_node = a[1]
            step = self.casti(self.expr(a[2]))
        stop = self.casti(self.expr(stop_node))
        var = self.vars[node.target.id][0]
        self.ins(self.body, OpStore, var, start)

        def cond():
            # (step > 0) ? (i < stop) : (i > stop) — matches the CPU
            # backend's direction-aware range condition
            i = self.nid()
            self.ins(self.body, OpLoad, self.ty("i64"), i, var)
            pos = self.nid()
            self.ins(self.body, OpSGreaterThan, self.ty("bool"), pos,
                     step, zero)
            up = self.nid()
            self.ins(self.body, OpSLessThan, self.ty("bool"), up, i, stop)
            dn = self.nid()
            self.ins(self.body, OpSGreaterThan, self.ty("bool"), dn,
                     i, stop)
            sel = self.nid()
            self.ins(self.body, OpSelect, self.ty("bool"), sel, pos, up, dn)
            return sel

        def advance():
            i = self.nid()
            self.ins(self.body, OpLoad, self.ty("i64"), i, var)
            nxt = self.nid()
            self.ins(self.body, OpIAdd, self.ty("i64"), nxt, i, step)
            self.ins(self.body, OpStore, var, nxt)

        self._loop(cond, node.body, advance)

    # ---------------------------- expressions ----------------------------
    def casti(self, pair):
        v, t = pair
        if t != I64:
            raise UnsupportedError("spirv: integer expression required")
        return v

    def tobool(self, v, t):
        if t == BOOL:
            return v
        if t == I64:
            r = self.nid()
            self.ins(self.body, OpINotEqual, self.ty("bool"), r, v,
                     self.const(("i64", 0)))
            return r
        r = self.nid()
        self.ins(self.body, OpFOrdNotEqual, self.ty("bool"), r, v,
                 self.const(("f64", 0.0)))
        return r

    def cast(self, v, frm, to):
        if frm == to:
            return v
        r = self.nid()
        if frm == I64 and to == F64:
            self.ins(self.body, OpConvertSToF, self.ty("f64"), r, v)
        elif frm == F64 and to == I64:
            self.ins(self.body, OpConvertFToS, self.ty("i64"), r, v)
        elif frm == BOOL and to == I64:
            self.ins(self.body, OpSelect, self.ty("i64"), r, v,
                     self.const(("i64", 1)), self.const(("i64", 0)))
        elif frm == BOOL and to == F64:
            self.ins(self.body, OpSelect, self.ty("f64"), r, v,
                     self.const(("f64", 1.0)), self.const(("f64", 0.0)))
        else:
            raise UnsupportedError(f"spirv: no cast {frm} -> {to}")
        return r

    def _scratch_var(self, type_key):
        """Function-storage variable created after entry-block emission
        started; inserted at the recorded OpVariable position."""
        pty = self.ty(("ptr", SC_FUNCTION, type_key))
        vid = self.nid()
        inst = ((4 << 16) | OpVariable, pty, vid, SC_FUNCTION)
        self.body.insert(self._var_pos, inst)
        self._var_pos += 1
        return vid

    def _atomic_add(self, node):
        """atomic_add(buf, i, v): OpAtomicIAdd for i64; for f64, native
        OpAtomicFAddEXT under Vulkan (SPV_EXT_shader_atomic_float_add),
        or a compare-exchange loop over the bit pattern under OpenCL
        (logical addressing forbids the pointer bitcast on Vulkan).
        Returns the old value."""
        if not isinstance(node.args[0], ast.Name):
            raise UnsupportedError("atomic_add: buffer name expected")
        name = node.args[0].id
        idx = self.casti(self.expr(node.args[1]))
        ptr, ety = self._elem_ptr(name, idx)
        want = F64 if ety == "f64" else I64
        v, vt = self.expr(node.args[2])
        val = self.cast(v, vt, want)
        scope = self.const(("i32", SCOPE_DEVICE))
        relaxed = self.const(("i32", 0))
        if ety == "i64":
            self.extra_caps.add(CAP_INT64_ATOMICS)
            old = self.nid()
            self.ins(self.body, OpAtomicIAdd, self.ty("i64"), old,
                     ptr, scope, relaxed, val)
            return old, I64
        if self.flavor == "vulkan":
            self.extra_caps.add(CAP_ATOMIC_FLOAT64_ADD_EXT)
            self.extensions.add(EXT_ATOMIC_FLOAT_ADD)
            old = self.nid()
            self.ins(self.body, OpAtomicFAddEXT, self.ty("f64"), old,
                     ptr, scope, relaxed, val)
            return old, F64
        # OpenCL flavor: CAS loop on the i64 bit pattern
        self.extra_caps.add(CAP_INT64_ATOMICS)
        _, _, kind = self.bufs[name]
        sc = SC_WORKGROUP if kind == "wg" else SC_CROSSWORKGROUP
        ipty = self.ty(("ptr", sc, "i64"))
        iptr = self.nid()
        self.ins(self.body, OpBitcast, ipty, iptr, ptr)
        exp_var = self._scratch_var("i64")     # expected bit pattern
        old_var = self._scratch_var("i64")     # winning old bits
        seed_f = self.nid()
        self.ins(self.body, OpLoad, self.ty("f64"), seed_f, ptr)
        seed = self.nid()
        self.ins(self.body, OpBitcast, self.ty("i64"), seed, seed_f)
        self.ins(self.body, OpStore, exp_var, seed)

        header, check, body_l, cont, merge = (self.nid() for _ in range(5))
        self.ins(self.body, OpBranch, header)
        self._label(header)
        self.ins(self.body, OpLoopMerge, merge, cont, 0)
        self.ins(self.body, OpBranch, check)
        self._label(check)
        e = self.nid()
        self.ins(self.body, OpLoad, self.ty("i64"), e, exp_var)
        ef = self.nid()
        self.ins(self.body, OpBitcast, self.ty("f64"), ef, e)
        nf = self.nid()
        self.ins(self.body, OpFAdd, self.ty("f64"), nf, ef, val)
        nb = self.nid()
        self.ins(self.body, OpBitcast, self.ty("i64"), nb, nf)
        got = self.nid()
        self.ins(self.body, OpAtomicCompareExchange, self.ty("i64"), got,
                 iptr, scope, relaxed, relaxed, nb, e)
        self.ins(self.body, OpStore, old_var, got)
        ok = self.nid()
        self.ins(self.body, OpIEqual, self.ty("bool"), ok, got, e)
        # success -> merge; failure -> continue with the freshly-read bits
        self.ins(self.body, OpBranchConditional, ok, merge, body_l)
        self._label(body_l)
        self.ins(self.body, OpStore, exp_var, got)
        self.ins(self.body, OpBranch, cont)
        self._label(cont)
        self.ins(self.body, OpBranch, header)
        self._label(merge)
        oldb = self.nid()
        self.ins(self.body, OpLoad, self.ty("i64"), oldb, old_var)
        oldf = self.nid()
        self.ins(self.body, OpBitcast, self.ty("f64"), oldf, oldb)
        return oldf, F64

    # GLSL.std.450 defines these only for 16/32-bit floats; f64 operands
    # are undefined behavior under Vulkan. Compute them at f32 and widen
    # back — a documented precision trade-off, like the metal backend.
    GL_F32_ONLY = {"sin", "cos", "tan", "exp", "log", "pow", "atan2"}

    def _mathcall(self, fname, args):
        ext = self.math_table.get(fname)
        if ext is None:
            raise UnsupportedError(f"spirv: math.{fname} not supported")
        ids = [self.cast(v, t, F64) for v, t in args]
        if self.flavor == "vulkan" and fname in self.GL_F32_ONLY:
            narrow = []
            for v in ids:
                w = self.nid()
                self.ins(self.body, OpFConvert, self.ty("f32"), w, v)
                narrow.append(w)
            r32 = self.nid()
            self.ins(self.body, OpExtInst, self.ty("f32"), r32,
                     self.ext_id, ext, *narrow)
            r = self.nid()
            self.ins(self.body, OpFConvert, self.ty("f64"), r, r32)
            return r, F64
        r = self.nid()
        self.ins(self.body, OpExtInst, self.ty("f64"), r, self.ext_id,
                 ext, *ids)
        return r, F64

    def _srem(self, a, b):
        """Truncated remainder a - (a/b)*b, built from OpSDiv.

        Deliberately avoids OpSRem: NVIDIA's Vulkan driver evaluates
        64-bit OpSRem as an UNSIGNED remainder (observed on 572.x:
        -7 srem 3 -> 0, matching (a mod 2^64) mod b), while OpSDiv
        truncates correctly on every driver tested.
        """
        q = self.nid()
        self.ins(self.body, OpSDiv, self.ty("i64"), q, a, b)
        qb = self.nid()
        self.ins(self.body, OpIMul, self.ty("i64"), qb, q, b)
        r = self.nid()
        self.ins(self.body, OpISub, self.ty("i64"), r, a, qb)
        return q, r

    def _floordiv(self, a, b):
        """Python floor division on i64 (sign-correct)."""
        q, r = self._srem(a, b)
        x = self.nid()
        self.ins(self.body, OpBitwiseXor, self.ty("i64"), x, a, b)
        neg = self.nid()
        self.ins(self.body, OpSLessThan, self.ty("bool"), neg, x,
                 self.const(("i64", 0)))
        nz = self.nid()
        self.ins(self.body, OpINotEqual, self.ty("bool"), nz, r,
                 self.const(("i64", 0)))
        adj = self.nid()
        self.ins(self.body, OpLogicalAnd, self.ty("bool"), adj, neg, nz)
        qm1 = self.nid()
        self.ins(self.body, OpISub, self.ty("i64"), qm1, q,
                 self.const(("i64", 1)))
        out = self.nid()
        self.ins(self.body, OpSelect, self.ty("i64"), out, adj, qm1, q)
        return out

    def _imod(self, a, b):
        _, r = self._srem(a, b)
        x = self.nid()
        self.ins(self.body, OpBitwiseXor, self.ty("i64"), x, a, b)
        neg = self.nid()
        self.ins(self.body, OpSLessThan, self.ty("bool"), neg, x,
                 self.const(("i64", 0)))
        nz = self.nid()
        self.ins(self.body, OpINotEqual, self.ty("bool"), nz, r,
                 self.const(("i64", 0)))
        adj = self.nid()
        self.ins(self.body, OpLogicalAnd, self.ty("bool"), adj, neg, nz)
        rb = self.nid()
        self.ins(self.body, OpIAdd, self.ty("i64"), rb, r, b)
        out = self.nid()
        self.ins(self.body, OpSelect, self.ty("i64"), out, adj, rb, r)
        return out

    def _fmod(self, a, b):
        r = self.nid()
        self.ins(self.body, OpFRem, self.ty("f64"), r, a, b)
        nz = self.nid()
        self.ins(self.body, OpFOrdNotEqual, self.ty("bool"), nz, r,
                 self.const(("f64", 0.0)))
        rneg = self.nid()
        self.ins(self.body, OpFOrdLessThan, self.ty("bool"), rneg, r,
                 self.const(("f64", 0.0)))
        bneg = self.nid()
        self.ins(self.body, OpFOrdLessThan, self.ty("bool"), bneg, b,
                 self.const(("f64", 0.0)))
        # sign(r) != sign(b), as (rneg && !bneg) || (!rneg && bneg)
        nb = self.nid()
        self.ins(self.body, OpLogicalNot, self.ty("bool"), nb, bneg)
        nr = self.nid()
        self.ins(self.body, OpLogicalNot, self.ty("bool"), nr, rneg)
        a1 = self.nid()
        self.ins(self.body, OpLogicalAnd, self.ty("bool"), a1, rneg, nb)
        a2 = self.nid()
        self.ins(self.body, OpLogicalAnd, self.ty("bool"), a2, nr, bneg)
        sd = self.nid()
        self.ins(self.body, OpLogicalOr, self.ty("bool"), sd, a1, a2)
        adj = self.nid()
        self.ins(self.body, OpLogicalAnd, self.ty("bool"), adj, nz, sd)
        rb = self.nid()
        self.ins(self.body, OpFAdd, self.ty("f64"), rb, r, b)
        out = self.nid()
        self.ins(self.body, OpSelect, self.ty("f64"), out, adj, rb, r)
        return out

    IOPS = {ast.Add: OpIAdd, ast.Sub: OpISub, ast.Mult: OpIMul,
            ast.BitAnd: OpBitwiseAnd, ast.BitOr: OpBitwiseOr,
            ast.BitXor: OpBitwiseXor, ast.LShift: OpShiftLeftLogical,
            ast.RShift: OpShiftRightArithmetic}
    FOPS = {ast.Add: OpFAdd, ast.Sub: OpFSub, ast.Mult: OpFMul}
    ICMP = {ast.Eq: OpIEqual, ast.NotEq: OpINotEqual,
            ast.Lt: OpSLessThan, ast.LtE: OpSLessThanEqual,
            ast.Gt: OpSGreaterThan, ast.GtE: OpSGreaterThanEqual}
    FCMP = {ast.Eq: OpFOrdEqual, ast.NotEq: OpFOrdNotEqual,
            ast.Lt: OpFOrdLessThan, ast.LtE: OpFOrdLessThanEqual,
            ast.Gt: OpFOrdGreaterThan, ast.GtE: OpFOrdGreaterThanEqual}

    def expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                key = ("true", None) if node.value else ("false", None)
                return self.const(key), BOOL
            if isinstance(node.value, float):
                return self.const(("f64", node.value)), F64
            if isinstance(node.value, int):
                return self.const(("i64", node.value)), I64
            raise UnsupportedError("spirv: unsupported constant")
        if isinstance(node, ast.Name):
            ent = self.vars.get(node.id)
            if ent is None:
                raise UnsupportedError(
                    f"spirv: unknown name {node.id!r} (pointer args may "
                    "only be indexed)")
            vid, t = ent
            r = self.nid()
            self.ins(self.body, OpLoad, self.ty(self.ABSTY[t]), r, vid)
            return r, t
        if isinstance(node, ast.Subscript):
            idx = self.casti(self.expr(node.slice))
            p, ety = self._elem_ptr(node.value.id, idx)
            r = self.nid()
            self.ins(self.body, OpLoad, self.ty(ety), r, p)
            return r, F64 if ety == "f64" else I64
        if isinstance(node, ast.UnaryOp):
            v, t = self.expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return v, t
            r = self.nid()
            if isinstance(node.op, ast.USub):
                self.ins(self.body,
                         OpFNegate if t == F64 else OpSNegate,
                         self.ty(self.ABSTY[t]), r, v)
                return r, t
            if isinstance(node.op, ast.Not):
                self.ins(self.body, OpLogicalNot, self.ty("bool"), r,
                         self.tobool(v, t))
                return r, BOOL
            if isinstance(node.op, ast.Invert) and t == I64:
                self.ins(self.body, OpNot, self.ty("i64"), r, v)
                return r, I64
            raise UnsupportedError("spirv: unsupported unary op")
        if isinstance(node, ast.BinOp):
            lv, lt = self.expr(node.left)
            rv, rt = self.expr(node.right)
            op = type(node.op)
            if op is ast.Div or op is ast.Pow:
                lv = self.cast(lv, lt, F64)
                rv = self.cast(rv, rt, F64)
                if op is ast.Pow:
                    return self._mathcall("pow", [(lv, F64), (rv, F64)])
                r = self.nid()
                self.ins(self.body, OpFDiv, self.ty("f64"), r, lv, rv)
                return r, F64
            if lt == F64 or rt == F64:
                lv = self.cast(lv, lt, F64)
                rv = self.cast(rv, rt, F64)
                if op is ast.Mod:
                    return self._fmod(lv, rv), F64
                if op is ast.FloorDiv:
                    d = self.nid()
                    self.ins(self.body, OpFDiv, self.ty("f64"), d, lv, rv)
                    return self._mathcall("floor", [(d, F64)])
                o = self.FOPS.get(op)
                if o is None:
                    raise UnsupportedError("spirv: unsupported float op")
                r = self.nid()
                self.ins(self.body, o, self.ty("f64"), r, lv, rv)
                return r, F64
            lv = self.cast(lv, lt, I64)
            rv = self.cast(rv, rt, I64)
            if op is ast.FloorDiv:
                return self._floordiv(lv, rv), I64
            if op is ast.Mod:
                return self._imod(lv, rv), I64
            o = self.IOPS.get(op)
            if o is None:
                raise UnsupportedError("spirv: unsupported int op")
            r = self.nid()
            self.ins(self.body, o, self.ty("i64"), r, lv, rv)
            return r, I64
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise UnsupportedError("spirv: chained comparisons")
            lv, lt = self.expr(node.left)
            rv, rt = self.expr(node.comparators[0])
            if lt == F64 or rt == F64:
                lv = self.cast(lv, lt, F64)
                rv = self.cast(rv, rt, F64)
                o = self.FCMP[type(node.ops[0])]
            else:
                lv = self.cast(lv, lt, I64)
                rv = self.cast(rv, rt, I64)
                o = self.ICMP[type(node.ops[0])]
            r = self.nid()
            self.ins(self.body, o, self.ty("bool"), r, lv, rv)
            return r, BOOL
        if isinstance(node, ast.BoolOp):
            op = OpLogicalAnd if isinstance(node.op, ast.And) \
                else OpLogicalOr
            acc = None
            for v in node.values:
                x, xt = self.expr(v)
                x = self.tobool(x, xt)
                if acc is None:
                    acc = x
                else:
                    r = self.nid()
                    self.ins(self.body, op, self.ty("bool"), r, acc, x)
                    acc = r
            return acc, BOOL
        if isinstance(node, ast.IfExp):
            c, ct = self.expr(node.test)
            c = self.tobool(c, ct)
            tv, tt = self.expr(node.body)
            fv, ft = self.expr(node.orelse)
            ty = F64 if F64 in (tt, ft) else tt
            tv = self.cast(tv, tt, ty)
            fv = self.cast(fv, ft, ty)
            r = self.nid()
            self.ins(self.body, OpSelect, self.ty(self.ABSTY[ty]), r,
                     c, tv, fv)
            return r, ty
        if isinstance(node, ast.Call):
            return self._call(node)
        raise UnsupportedError(
            f"spirv: unsupported expression {type(node).__name__}")

    def _call(self, node):
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in MATH_MODULES
                and node.func.attr in MATH_FNS):
            return self._mathcall(node.func.attr,
                                  [self.expr(a) for a in node.args])
        if not isinstance(node.func, ast.Name):
            raise UnsupportedError("spirv: unsupported call")
        fn = node.func.id
        if fn == "barrier":
            self.ins(self.body, OpControlBarrier,
                     self.const(("i32", SCOPE_WORKGROUP)),
                     self.const(("i32", SCOPE_WORKGROUP)),
                     self.const(("i32", SEM_WG_ACQREL)))
            return self.const(("i64", 0)), I64
        if fn == "atomic_add":
            return self._atomic_add(node)
        if fn in GPU_INTRINSICS:
            from ..typeinfer import gpu_intrinsic_axis
            stem, axis = gpu_intrinsic_axis(fn)
            if stem == "thread_id":
                return self._thread_query(BI_LOCALINVOCATIONID, axis), I64
            if stem == "block_id":
                return self._thread_query(BI_WORKGROUPID, axis), I64
            if self.flavor == "vulkan":   # block_dim: baked LocalSize
                return self.const(("i64", self.local_size[axis])), I64
            return self._thread_query(BI_WORKGROUPSIZE, axis), I64
        if fn == "abs":
            v, t = self.expr(node.args[0])
            if t == F64:
                return self._mathcall("fabs", [(v, t)])
            neg = self.nid()
            self.ins(self.body, OpSNegate, self.ty("i64"), neg, v)
            lt = self.nid()
            self.ins(self.body, OpSLessThan, self.ty("bool"), lt, v,
                     self.const(("i64", 0)))
            r = self.nid()
            self.ins(self.body, OpSelect, self.ty("i64"), r, lt, neg, v)
            return r, I64
        if fn == "float":
            v, t = self.expr(node.args[0])
            return self.cast(v, t, F64), F64
        if fn == "int":
            v, t = self.expr(node.args[0])
            return self.cast(v, t, I64), I64
        raise UnsupportedError(f"spirv: unsupported call {fn!r}")


def generate(func_ast, arg_types, var_types, ret_type, flavor,
             local_size=(64, 1, 1)):
    """Return SPIR-V binary (bytes) for the kernel."""
    return SpirvGen(func_ast, arg_types, var_types, ret_type,
                    flavor, local_size).generate()
