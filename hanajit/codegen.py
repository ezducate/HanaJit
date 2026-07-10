"""Lower a type-annotated Python AST to LLVM IR using llvmlite.ir.

Supported subset (v0.1): i64/f64/bool scalars, arithmetic, comparisons,
if/elif/else, while, for-in-range, break/continue, self-recursion,
abs()/int()/float(), ternary expressions.
"""
import ast
from llvmlite import ir
from .errors import UnsupportedError
from .typeinfer import (I64, F64, F32, BOOL, PF64, PI64, AF64, AI64,
                        POINTER_ELEM, ARRAY_ELEM, ELEM, LAZY, lazy_of,
                        seq_elem, arr_ty, arr_base, arr_nd, arr_contig,
                        GPU_INTRINSICS, MATH_FNS, MATH_MODULES,
                        NP_REDUCTIONS, NP_ELEMWISE2)

LLTY = {I64: ir.IntType(64), F64: ir.DoubleType(), F32: ir.FloatType(),
        BOOL: ir.IntType(1),
        PF64: ir.DoubleType().as_pointer(), PI64: ir.IntType(64).as_pointer()}
for _t, _e in ARRAY_ELEM.items():
    LLTY[_t] = LLTY[_e].as_pointer()

# ABI components per array kind (after the data pointer, all i64):
#   1c: n | 1s: n, stride | 2c: s0, s1 | 2s: s0, s1, st0, st1
ARR_META = {"1c": 1, "1s": 2, "2c": 2, "2s": 4}


def arr_meta_count(t):
    return ARR_META[t[-3:-1]]



def _common_num(a, b):
    """Common numeric type for a binary op, float-aware (F32 included).
    F64 dominates F32 dominates I64 dominates BOOL."""
    if F64 in (a, b):
        return F64
    if F32 in (a, b):
        return F32
    return I64

class Lazy:
    """A fused array expression: element type, runtime length, and a
    generator emitting IR for element i at consumption time. Composing
    Lazies fuses everything into the single consumer loop — no
    temporaries, no allocation."""
    __slots__ = ("ety", "n", "gen")

    def __init__(self, ety, n, gen):
        self.ety, self.n, self.gen = ety, n, gen

# per-vendor thread intrinsics (all return i32)
GPU_LOWER = {
    "cuda": {"thread_id": "llvm.nvvm.read.ptx.sreg.tid.x",
             "block_id": "llvm.nvvm.read.ptx.sreg.ctaid.x",
             "block_dim": "llvm.nvvm.read.ptx.sreg.ntid.x"},
    "amd":  {"thread_id": "llvm.amdgcn.workitem.id.x",
             "block_id": "llvm.amdgcn.workgroup.id.x"},
}


class CodeGen:
    def __init__(self, func_ast, arg_types, var_types, ret_type,
                 module_name="hanajit", fastmath=False, gpu=None,
                 reduce_reassoc=False):
        self.ff = ["fast"] if fastmath else []
        # reassoc ONLY on reduction accumulators: lets the loop vectorizer
        # use parallel SIMD accumulators (like numpy's pairwise sum) without
        # enabling nnan/ninf/nsz. Safe on normal data; documented.
        self.rff = ["fast"] if fastmath else (
            ["reassoc"] if reduce_reassoc else [])
        self.gpu = gpu
        self.func_ast = func_ast
        self.arg_types = arg_types
        self.var_types = var_types
        self.ret_type = ret_type
        self.module = ir.Module(name=module_name)
        self.vars = {}          # name -> (alloca ptr, abstract type)
        self.arrays = {}        # name -> (data ptr, length value, abstract ty)
        self.loop_stack = []    # (continue_block, break_block)

    # ---------- casting ----------
    def cast(self, val, from_ty, to_ty):
        if from_ty == to_ty:
            return val
        b = self.builder
        if from_ty == BOOL and to_ty == I64:
            return b.zext(val, LLTY[I64])
        if from_ty == BOOL and to_ty in (F64, F32):
            return b.uitofp(val, LLTY[to_ty])
        if from_ty == I64 and to_ty in (F64, F32):
            return b.sitofp(val, LLTY[to_ty])
        if from_ty in (F64, F32) and to_ty == I64:
            return b.fptosi(val, LLTY[I64])
        if from_ty == F32 and to_ty == F64:
            return b.fpext(val, LLTY[F64])
        if from_ty == F64 and to_ty == F32:
            return b.fptrunc(val, LLTY[F32])
        if to_ty == BOOL:
            return self.truthy(val, from_ty)
        raise UnsupportedError(f"cannot cast {from_ty} -> {to_ty}")

    def truthy(self, val, ty):
        b = self.builder
        if ty == BOOL:
            return val
        if ty == I64:
            return b.icmp_signed("!=", val, ir.Constant(LLTY[I64], 0))
        if ty == F32:
            return b.fcmp_ordered("!=", val, ir.Constant(LLTY[F32], 0.0))
        return b.fcmp_ordered("!=", val, ir.Constant(LLTY[F64], 0.0))

    # ---------- entry ----------
    def generate(self):
        ll_args = []
        for a in self.func_ast.args.args:
            ty = self.arg_types[a.arg]
            if ty in ARRAY_ELEM:
                ll_args += [LLTY[ty]] + [LLTY[I64]] * arr_meta_count(ty)
            else:
                ll_args.append(LLTY[ty])
        ret_ll = (ir.VoidType() if self.gpu
                  else LLTY[self.ret_type])
        fnty = ir.FunctionType(ret_ll, ll_args)
        self.fn = ir.Function(self.module, fnty, name=self.func_ast.name)
        entry = self.fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)

        # allocas for every non-array variable, typed by inference
        for name, ty in self.var_types.items():
            if ty in ARRAY_ELEM or ty in LAZY:
                continue  # arrays/fused exprs live as values, not allocas
            self.vars[name] = (self.builder.alloca(LLTY[ty], name=name), ty)
        k = 0
        for a in self.func_ast.args.args:
            ty = self.arg_types[a.arg]
            if ty in ARRAY_ELEM:
                m = arr_meta_count(ty)
                comps = tuple(self.fn.args[k:k + 1 + m])
                comps[0].name = a.arg
                # distinct array args assumed non-overlapping (numba's
                # contract too): unlocks vectorization across stores
                comps[0].add_attribute("noalias")
                self.arrays[a.arg] = (ty, comps)
                k += 1 + m
            else:
                llarg = self.fn.args[k]
                llarg.name = a.arg
                if ty in POINTER_ELEM:
                    llarg.add_attribute("noalias")
                ptr, vty = self.vars[a.arg]
                self.builder.store(self.cast(llarg, ty, vty), ptr)
                k += 1

        for stmt in self.func_ast.body:
            self.stmt(stmt)

        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(LLTY[self.ret_type],
                                         0 if self.ret_type != F64 else 0.0))
        return self.module

    # ---------- statements ----------
    def stmt(self, node):
        if self.builder.block.is_terminated:
            return  # dead code after return/break
        m = getattr(self, "stmt_" + type(node).__name__, None)
        if m is None:
            raise UnsupportedError(f"unsupported statement: {type(node).__name__}")
        m(node)

    def stmt_Assign(self, node):
        import ast as _ast
        t = node.targets[0]
        if (isinstance(t, _ast.Name)
                and self.var_types.get(t.id) in ARRAY_ELEM):
            if t.id in self.arrays:
                raise UnsupportedError(
                    "array views may be bound once (no reassignment)")
            self.arrays[t.id] = self._arr_of(node.value, f"{t.id} = ...")
            return
        val, ty = self.expr(node.value)
        if isinstance(t, _ast.Subscript) and isinstance(t.slice, _ast.Slice):
            base, bty = self.expr(t.value)
            if bty not in ARRAY_ELEM or arr_nd(bty) != 1:
                raise UnsupportedError("slice-store target must be 1-D array")
            sl = t.slice
            vty, vcomps = self._slice_view(bty, base, sl.lower, sl.upper,
                                           sl.step)
            elem = ELEM[vty]
            lz = self.to_lazy(val, ty)
            b_ = self.builder
            n = vcomps[1] if lz is None else lz.n

            def stp(acc, i):
                v = (self.cast(val, ty, elem) if lz is None
                     else self.cast(lz.gen(i), lz.ety, elem))
                p = b_.gep(vcomps[0], [b_.mul(i, vcomps[2])])
                b_.store(v, p)
                return acc
            self._reduce_loop(n, ir.Constant(LLTY[I64], 0), stp, I64)
            return
        if isinstance(t, _ast.Subscript):
            base, bty = self.expr(t.value)
            elem = ELEM[bty]
            if bty in ARRAY_ELEM:
                if isinstance(t.slice, _ast.Tuple):
                    i = self.cast(*self.expr(t.slice.elts[0]), I64)
                    j = self.cast(*self.expr(t.slice.elts[1]), I64)
                    p = self._elem_ptr(bty, base, i, j)
                else:
                    p = self._elem_ptr(bty, base,
                                       self.cast(*self.expr(t.slice), I64))
            else:
                p = self.builder.gep(base,
                                     [self.cast(*self.expr(t.slice), I64)])
            self.builder.store(self.cast(val, ty, elem), p)
            return
        if ty in ARRAY_ELEM:  # array view binding: y = x[2:9], r = m[i], ...
            if t.id in self.arrays:
                raise UnsupportedError(
                    "array views are single-assignment (rebind a new name)")
            self.arrays[t.id] = (ty, val)
            return
        if ty in LAZY:        # named fused expression: t = a * b
            if not hasattr(self, "lazies"):
                self.lazies = {}
            if t.id in self.lazies:
                raise UnsupportedError(
                    "fused expressions are single-assignment")
            self.lazies[t.id] = (val, ty)
            return
        ptr, vty = self.vars[t.id]
        self.builder.store(self.cast(val, ty, vty), ptr)

    def stmt_AugAssign(self, node):
        name = node.target.id
        ptr, vty = self.vars[name]
        cur = (self.builder.load(ptr), vty)
        # float accumulator with += / -= is a reduction: apply reassoc flags
        # (if enabled) so LLVM can vectorize with parallel accumulators.
        if (vty == F64 and self.rff
                and isinstance(node.op, (ast.Add, ast.Sub))):
            rhs, rty = self.expr(node.value)
            rhs = self.cast(rhs, rty, F64)
            b = self.builder
            val = (b.fadd(cur[0], rhs, flags=self.rff)
                   if isinstance(node.op, ast.Add)
                   else b.fsub(cur[0], rhs, flags=self.rff))
            b.store(val, ptr)
            return
        val, ty = self.binop(node.op, cur, self.expr(node.value))
        self.builder.store(self.cast(val, ty, vty), ptr)

    def stmt_Return(self, node):
        if self.gpu:
            if node.value is not None:
                self.expr(node.value)   # evaluate for side effects only
            self.builder.ret_void()
            return
        val, ty = self.expr(node.value)
        self.builder.ret(self.cast(val, ty, self.ret_type))

    def stmt_Expr(self, node):
        pass

    def stmt_Pass(self, node):
        pass

    def stmt_If(self, node):
        cond = self.truthy(*self.expr(node.test))
        then_bb = self.fn.append_basic_block("then")
        else_bb = self.fn.append_basic_block("else")
        end_bb = self.fn.append_basic_block("endif")
        self.builder.cbranch(cond, then_bb, else_bb)
        self.builder.position_at_end(then_bb)
        for s in node.body:
            self.stmt(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(end_bb)
        self.builder.position_at_end(else_bb)
        for s in node.orelse:
            self.stmt(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(end_bb)
        self.builder.position_at_end(end_bb)

    def stmt_While(self, node):
        cond_bb = self.fn.append_basic_block("while.cond")
        body_bb = self.fn.append_basic_block("while.body")
        end_bb = self.fn.append_basic_block("while.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        self.builder.cbranch(self.truthy(*self.expr(node.test)), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        self.loop_stack.append((cond_bb, end_bb))
        for s in node.body:
            self.stmt(s)
        self.loop_stack.pop()
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)

    def stmt_For(self, node):
        args = [self.cast(*self.expr(a), I64) for a in node.iter.args]
        zero, one = ir.Constant(LLTY[I64], 0), ir.Constant(LLTY[I64], 1)
        if len(args) == 1:
            start, stop, step = zero, args[0], one
        elif len(args) == 2:
            start, stop, step = args[0], args[1], one
        else:
            start, stop, step = args

        ptr, _ = self.vars[node.target.id]
        self.builder.store(start, ptr)

        cond_bb = self.fn.append_basic_block("for.cond")
        body_bb = self.fn.append_basic_block("for.body")
        inc_bb = self.fn.append_basic_block("for.inc")
        end_bb = self.fn.append_basic_block("for.end")
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        i = self.builder.load(ptr)
        pos = self.builder.icmp_signed(">", step, zero)
        lt = self.builder.icmp_signed("<", i, stop)
        gt = self.builder.icmp_signed(">", i, stop)
        self.builder.cbranch(self.builder.select(pos, lt, gt), body_bb, end_bb)

        self.builder.position_at_end(body_bb)
        self.loop_stack.append((inc_bb, end_bb))
        for s in node.body:
            self.stmt(s)
        self.loop_stack.pop()
        if not self.builder.block.is_terminated:
            self.builder.branch(inc_bb)

        self.builder.position_at_end(inc_bb)
        self.builder.store(self.builder.add(self.builder.load(ptr), step), ptr)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)

    def stmt_Break(self, node):
        if not self.loop_stack:
            raise UnsupportedError("break outside loop")
        self.builder.branch(self.loop_stack[-1][1])

    def stmt_Continue(self, node):
        if not self.loop_stack:
            raise UnsupportedError("continue outside loop")
        self.builder.branch(self.loop_stack[-1][0])

    # ---------- expressions ----------
    def expr(self, node):
        b = self.builder
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return ir.Constant(LLTY[BOOL], int(node.value)), BOOL
            if isinstance(node.value, int):
                return ir.Constant(LLTY[I64], node.value), I64
            if isinstance(node.value, float):
                return ir.Constant(LLTY[F64], node.value), F64
            raise UnsupportedError(f"constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in self.arrays:
                ty, comps = self.arrays[node.id]
                return comps, ty
            if hasattr(self, "lazies") and node.id in self.lazies:
                return self.lazies[node.id]
            ptr, ty = self.vars[node.id]
            return b.load(ptr), ty
        if isinstance(node, ast.BinOp):
            l, r = self.expr(node.left), self.expr(node.right)
            if l[1] in LAZY or r[1] in LAZY or l[1] in ARRAY_ELEM \
                    or r[1] in ARRAY_ELEM:
                probe = self._elem_binop_ty(node.op, l[1], r[1])
                lz = self._combine(probe, [l, r],
                                   lambda e: self.binop(node.op, e[0],
                                                        e[1])[0])
                return lz, lazy_of(probe)
            return self.binop(node.op, l, r)
        if isinstance(node, ast.UnaryOp):
            v, ty = self.expr(node.operand)
            if ty in LAZY or ty in ARRAY_ELEM:
                ety = seq_elem(ty)
                if isinstance(node.op, ast.Not):
                    lz = self._combine(BOOL, [(v, ty)], lambda e: b.not_(
                        self.truthy(e[0][0], e[0][1])))
                    return lz, "~bool"
                def neg(e):
                    x, t = e[0]
                    return b.fneg(x) if t == F64 else b.neg(x)
                return self._combine(ety, [(v, ty)],
                                     neg if isinstance(node.op, ast.USub)
                                     else lambda e: e[0][0]), lazy_of(ety)
            if isinstance(node.op, ast.USub):
                return (b.fneg(v) if ty == F64 else b.neg(v)), ty
            if isinstance(node.op, ast.UAdd):
                return v, ty
            if isinstance(node.op, ast.Not):
                return b.not_(self.truthy(v, ty)), BOOL
            raise UnsupportedError("unsupported unary op")
        if isinstance(node, ast.Compare):
            # peephole: `a % b == 0` needs no floor-mod adjustment —
            # the adjusted remainder is zero iff the raw remainder is zero.
            if type(node.ops[0]) in (ast.Eq, ast.NotEq):
                for modnode, zeronode in ((node.left, node.comparators[0]),
                                          (node.comparators[0], node.left)):
                    if (isinstance(modnode, ast.BinOp)
                            and isinstance(modnode.op, ast.Mod)
                            and isinstance(zeronode, ast.Constant)
                            and type(zeronode.value) is int
                            and zeronode.value == 0
                            and not isinstance(modnode.left, ast.Name)
                            or isinstance(modnode, ast.BinOp)
                            and isinstance(modnode.op, ast.Mod)
                            and isinstance(zeronode, ast.Constant)
                            and type(zeronode.value) is int
                            and zeronode.value == 0):
                        (lv, lt) = self.expr(modnode.left)
                        (rv, rt) = self.expr(modnode.right)
                        if lt == I64 and rt == I64:
                            rem = b.srem(lv, rv)
                            cmp = ("==" if isinstance(node.ops[0], ast.Eq)
                                   else "!=")
                            return b.icmp_signed(
                                cmp, rem, ir.Constant(LLTY[I64], 0)), BOOL
                        break
            l = self.expr(node.left)
            r = self.expr(node.comparators[0])
            if l[1] in LAZY or r[1] in LAZY or l[1] in ARRAY_ELEM \
                    or r[1] in ARRAY_ELEM:
                op = node.ops[0]

                def cmp(e):
                    (lv, lt), (rv, rt) = e
                    common = _common_num(lt, rt)
                    lv = self.cast(lv, lt, common)
                    rv = self.cast(rv, rt, common)
                    o = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
                         ast.LtE: "<=", ast.Gt: ">",
                         ast.GtE: ">="}[type(op)]
                    return (b.fcmp_ordered(o, lv, rv) if common == F64
                            else b.icmp_signed(o, lv, rv))
                return self._combine(BOOL, [l, r], cmp), "~bool"
            (lv, lt), (rv, rt) = l, r
            common = _common_num(lt, rt)
            lv, rv = self.cast(lv, lt, common), self.cast(rv, rt, common)
            op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
                  ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}.get(
                      type(node.ops[0]))
            if op is None:
                raise UnsupportedError("unsupported comparison")
            if common == F64:
                return b.fcmp_ordered(op, lv, rv), BOOL
            return b.icmp_signed(op, lv, rv), BOOL
        if isinstance(node, ast.BoolOp):
            vals = [self.truthy(*self.expr(v)) for v in node.values]
            acc = vals[0]
            for v in vals[1:]:
                acc = b.and_(acc, v) if isinstance(node.op, ast.And) else b.or_(acc, v)
            return acc, BOOL
        if isinstance(node, ast.IfExp):
            # real branches (not select): only the taken arm is evaluated,
            # matching Python semantics and skipping dead work (e.g. an
            # unneeded floor-division in a collatz ternary)
            cond = self.truthy(*self.expr(node.test))
            then_bb = self.fn.append_basic_block("ternary.then")
            else_bb = self.fn.append_basic_block("ternary.else")
            end_bb = self.fn.append_basic_block("ternary.end")
            b.cbranch(cond, then_bb, else_bb)
            b.position_at_end(then_bb)
            tv, tt = self.expr(node.body)
            then_out = b.block
            b.position_at_end(else_bb)
            fv, ft = self.expr(node.orelse)
            else_out = b.block
            common = F64 if F64 in (tt, ft) else tt
            b.position_at_end(then_out)
            tvc = self.cast(tv, tt, common)
            b.branch(end_bb)
            b.position_at_end(else_out)
            fvc = self.cast(fv, ft, common)
            b.branch(end_bb)
            b.position_at_end(end_bb)
            phi = b.phi(LLTY[common])
            phi.add_incoming(tvc, then_out)
            phi.add_incoming(fvc, else_out)
            return phi, common
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"):
                aty, comps = self._arr_of(node.value.value, ".shape")
                if not isinstance(node.slice, ast.Constant):
                    raise UnsupportedError(".shape index must be a literal")
                k = node.slice.value
                if arr_nd(aty) == 1:
                    if k != 0:
                        raise UnsupportedError("1-D shape has index 0 only")
                    return comps[1], I64
                return comps[1 + k], I64
            base, bty = self.expr(node.value)
            if bty in ARRAY_ELEM:
                if isinstance(node.slice, ast.Slice):
                    sl = node.slice
                    nty, ncomps = self._slice_view(bty, base, sl.lower,
                                                   sl.upper, sl.step)
                    return ncomps, nty
                if isinstance(node.slice, ast.Tuple):
                    i = self.cast(*self.expr(node.slice.elts[0]), I64)
                    j = self.cast(*self.expr(node.slice.elts[1]), I64)
                    return b.load(self._elem_ptr(bty, base, i, j)), ELEM[bty]
                idx = self.cast(*self.expr(node.slice), I64)
                if arr_nd(bty) == 2:
                    nty, ncomps = self._row_view(bty, base, idx)
                    return ncomps, nty
                return b.load(self._elem_ptr(bty, base, idx)), ELEM[bty]
            idx = self.cast(*self.expr(node.slice), I64)
            return b.load(b.gep(base, [idx])), ELEM[bty]
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in MATH_MODULES
                and node.attr in ("pi", "e")):
            import math as _m
            return ir.Constant(LLTY[F64],
                               _m.pi if node.attr == "pi" else _m.e), F64
        if isinstance(node, ast.Attribute) and node.attr == "T":
            aty, comps = self._arr_of(node.value, ".T")
            one = ir.Constant(LLTY[I64], 1)
            if aty[-3:-1] == "2c":
                st0, st1 = comps[2], one
            else:
                st0, st1 = comps[3], comps[4]
            return ((comps[0], comps[2], comps[1], st1, st0),
                    arr_ty(arr_base(aty), 2, False))
        if isinstance(node, ast.Call):
            return self.call(node)
        raise UnsupportedError(f"unsupported expression: {type(node).__name__}")

    MATH_INTRIN = {"sqrt": "llvm.sqrt.f64", "exp": "llvm.exp.f64",
                   "log": "llvm.log.f64", "sin": "llvm.sin.f64",
                   "cos": "llvm.cos.f64", "floor": "llvm.floor.f64",
                   "ceil": "llvm.ceil.f64", "fabs": "llvm.fabs.f64"}

    def _arr_of(self, node, what):
        v, ty = self.expr(node)
        if ty not in ARRAY_ELEM:
            raise UnsupportedError(f"{what} requires an array argument")
        return ty, v

    # ---- array geometry -----------------------------------------------
    def _wrap(self, i, n):
        """numpy-style negative index: i<0 -> i+n (no bounds check)."""
        b = self.builder
        return b.select(b.icmp_signed("<", i, ir.Constant(LLTY[I64], 0)),
                        b.add(i, n), i)

    def _arr_len(self, ty, comps):
        return comps[1]  # n for 1-D, s0 for 2-D

    def _arr_total(self, ty, comps):
        b = self.builder
        return comps[1] if arr_nd(ty) == 1 else b.mul(comps[1], comps[2])

    def _elem_ptr(self, ty, comps, idx, jdx=None):
        b = self.builder
        one = ir.Constant(LLTY[I64], 1)
        kind = ty[-3:-1]
        if kind == "1c":
            return b.gep(comps[0], [self._wrap(idx, comps[1])])
        if kind == "1s":
            return b.gep(comps[0],
                         [b.mul(self._wrap(idx, comps[1]), comps[2])])
        if kind == "2c":
            st0, st1 = comps[2], one
        else:
            st0, st1 = comps[3], comps[4]
        off = b.mul(self._wrap(idx, comps[1]), st0)
        if jdx is not None:
            off = b.add(off, b.mul(self._wrap(jdx, comps[2]), st1))
        return b.gep(comps[0], [off])

    def _row_view(self, ty, comps, idx):
        base = self._elem_ptr(ty, comps, idx, None)
        if ty[-3:-1] == "2c":
            return arr_ty(arr_base(ty), 1, True), (base, comps[2])
        return arr_ty(arr_base(ty), 1, False), (base, comps[2], comps[4])

    def _slice_view(self, ty, comps, lo, hi, st_node):
        """Full Python slice semantics on a 1-D array (CPython
        slice.indices), branch-free with selects."""
        b = self.builder
        I = lambda v: ir.Constant(LLTY[I64], v)
        n = comps[1]
        stride = comps[2] if ty[-3:-1] == "1s" else I(1)
        step = (self.cast(*self.expr(st_node), I64)
                if st_node is not None else I(1))
        pos = b.icmp_signed(">", step, I(0))

        def norm(node, dflt_pos, dflt_neg, lo_clamp_neg):
            if node is None:
                return b.select(pos, dflt_pos, dflt_neg)
            v = self.cast(*self.expr(node), I64)
            v = b.select(b.icmp_signed("<", v, I(0)), b.add(v, n), v)
            # clamp: pos in [0, n]; neg in [-1, n-1]
            lo_ = b.select(pos, I(0), I(-1)) if lo_clamp_neg else I(0)
            lo_ = b.select(pos, I(0), I(-1))
            hi_ = b.select(pos, n, b.sub(n, I(1)))
            v = b.select(b.icmp_signed("<", v, lo_), lo_, v)
            return b.select(b.icmp_signed(">", v, hi_), hi_, v)

        start = norm(lo, I(0), b.sub(n, I(1)), True)
        stop = norm(hi, n, I(-1), True)
        # length = max(0, ceil((stop-start)/step)) with sign-aware rounding
        diff = b.sub(stop, start)
        adj = b.select(pos, b.add(diff, b.sub(step, I(1))),
                       b.add(diff, b.add(step, I(1))))
        ln = b.sdiv(adj, step)
        ln = b.select(b.icmp_signed("<", ln, I(0)), I(0), ln)
        newptr = b.gep(comps[0], [b.mul(start, stride)])
        newstride = b.mul(step, stride)
        return (arr_ty(arr_base(ty), 1, False), (newptr, ln, newstride))

    def _reduce_loop(self, n, init, step_fn, acc_ty):
        """acc = init; for i in 0..n: acc = step_fn(acc, i).

        Emitted in SSA phi form (no alloca, no per-iteration stack spill).
        The induction variable and accumulator are loop phis, which the
        LLVM loop vectorizer can split into parallel SIMD accumulators."""
        b = self.builder
        pre = b.block
        cond = self.fn.append_basic_block("red.cond")
        body = self.fn.append_basic_block("red.body")
        end = self.fn.append_basic_block("red.end")
        b.branch(cond)

        b.position_at_end(cond)
        i_phi = b.phi(LLTY[I64], "red.i")
        acc_phi = b.phi(LLTY[acc_ty], "red.acc")
        i_phi.add_incoming(ir.Constant(LLTY[I64], 0), pre)
        acc_phi.add_incoming(init, pre)
        b.cbranch(b.icmp_signed("<", i_phi, n), body, end)

        b.position_at_end(body)
        new_acc = step_fn(acc_phi, i_phi)
        next_i = b.add(i_phi, ir.Constant(LLTY[I64], 1))
        body_end = b.block
        i_phi.add_incoming(next_i, body_end)
        acc_phi.add_incoming(new_acc, body_end)
        b.branch(cond)

        b.position_at_end(end)
        return acc_phi

    def _elem_binop_ty(self, op, lt, rt):
        el = seq_elem(lt) or lt
        er = seq_elem(rt) or rt
        if isinstance(op, ast.Div):
            return F64
        if isinstance(op, ast.Pow):
            return _common_num(el, er)
        return _common_num(el, er)

    def to_lazy(self, val, ty):
        """Coerce an array value or existing Lazy into a Lazy; scalars
        return None (broadcast at combine time)."""
        if ty in LAZY:
            return val
        if ty in ARRAY_ELEM:
            if arr_nd(ty) == 2 and not arr_contig(ty):
                raise UnsupportedError(
                    "strided 2-D views not usable in fused expressions")
            n, load = self._flat_load(ty, val)
            return Lazy(ARRAY_ELEM[ty], n, load)
        return None

    def _combine(self, out_ety, parts, emit):
        """parts: list of (val, ty). Returns fused Lazy applying emit to
        per-element scalar (value, elem_ty) pairs."""
        lz = [self.to_lazy(v, t) for v, t in parts]
        n = next(l.n for l in lz if l is not None)

        def gen(i):
            elems = []
            for (v, t), l in zip(parts, lz):
                if l is None:
                    elems.append((v, t))
                else:
                    elems.append((l.gen(i), l.ety))
            return emit(elems)
        return Lazy(out_ety, n, gen)

    def _flat_load(self, ty, comps):
        """Return (total, load(i)) treating the array as flat storage
        (valid for 1c/1s/2c)."""
        b = self.builder
        kind = ty[-3:-1]
        if kind == "1s":
            return comps[1], lambda i: b.load(
                b.gep(comps[0], [b.mul(i, comps[2])]))
        total = self._arr_total(ty, comps)
        return total, lambda i: b.load(b.gep(comps[0], [i]))

    def _seq(self, node, what):
        """Evaluate node to a Lazy (array or fused expression)."""
        v, t = self.expr(node)
        lz = self.to_lazy(v, t)
        if lz is None:
            raise UnsupportedError(f"{what} requires an array/expression")
        return lz

    def _lower_reduction(self, op, argnodes):
        b = self.builder
        I = lambda v: ir.Constant(LLTY[I64], v)
        lz = self._seq(argnodes[0], f"np.{op}")
        ety, n, load = lz.ety, lz.n, lz.gen
        addf = (lambda a, v: b.add(a, v)) if ety != F64 else \
               (lambda a, v: b.fadd(a, v, flags=self.rff))
        zero = ir.Constant(LLTY[ety], 0 if ety != F64 else 0.0)
        if op in ("sum", "mean", "var", "std"):
            sty = I64 if ety in (I64, BOOL) else F64
            def stp(a, i):
                v = self.cast(load(i), ety, sty)
                return b.add(a, v) if sty == I64 else b.fadd(
                    a, v, flags=self.rff)
            s = self._reduce_loop(n, ir.Constant(LLTY[sty],
                                                 0 if sty == I64 else 0.0),
                                  stp, sty)
            if op == "sum":
                return s, sty
            mean = b.fdiv(self.cast(s, sty, F64), b.sitofp(n, LLTY[F64]))
            if op == "mean":
                return mean, F64
            def stp2(a, i):
                d = b.fsub(self.cast(load(i), ety, F64), mean)
                return b.fadd(a, b.fmul(d, d))
            ss = self._reduce_loop(n, ir.Constant(LLTY[F64], 0.0), stp2, F64)
            var = b.fdiv(ss, b.sitofp(n, LLTY[F64]))
            if op == "var":
                return var, F64
            sq = self.intrinsic("llvm.sqrt.f64", LLTY[F64], [LLTY[F64]])
            return b.call(sq, [var]), F64
        if op == "prod":
            one = ir.Constant(LLTY[ety], 1 if ety != F64 else 1.0)
            mul = b.mul if ety != F64 else b.fmul
            return self._reduce_loop(n, one,
                                     lambda a, i: mul(a, load(i)), ety), ety
        if op == "dot":
            l2 = self._seq(argnodes[1], "np.dot")
            common = _common_num(ety, l2.ety)
            mul = b.mul if common == I64 else b.fmul
            add = (b.add if common == I64
                   else (lambda x, y: b.fadd(x, y, flags=self.rff)))
            z = ir.Constant(LLTY[common], 0 if common == I64 else 0.0)
            def stp(a, i):
                return add(a, mul(self.cast(load(i), ety, common),
                                  self.cast(l2.gen(i), l2.ety, common)))
            return self._reduce_loop(n, z, stp, common), common
        if op in ("any", "all"):
            init = ir.Constant(LLTY[BOOL], 0 if op == "any" else 1)
            comb = b.or_ if op == "any" else b.and_
            def stp(a, i):
                return comb(a, self.truthy(load(i), ety))
            return self._reduce_loop(n, init, stp, BOOL), BOOL
        if op == "count_nonzero":
            def stp(a, i):
                return b.add(a, b.zext(self.truthy(load(i), ety),
                                       LLTY[I64]))
            return self._reduce_loop(n, I(0), stp, I64), I64
        if op in ("argmin", "argmax"):
            # packed loop: track best value + index via two allocas
            best = b.alloca(LLTY[ety])
            bidx = b.alloca(LLTY[I64])
            b.store(load(I(0)), best)
            b.store(I(0), bidx)
            cmp = "<" if op == "argmin" else ">"
            def stp(a, i):
                v = load(i)
                cur = b.load(best)
                better = (b.icmp_signed(cmp, v, cur) if ety != F64
                          else b.fcmp_ordered(cmp, v, cur))
                b.store(b.select(better, v, cur), best)
                b.store(b.select(better, i, b.load(bidx)), bidx)
                return a
            self._reduce_loop(n, I(0), stp, I64)
            return b.load(bidx), I64
        first = load(I(0))
        cmp = "<" if op == "min" else ">"
        def stp(a, i):
            v = load(i)
            if ety != F64:
                return b.select(b.icmp_signed(cmp, v, a), v, a)
            # numpy semantics: any NaN poisons min/max. Take v when it is
            # NaN or strictly better; once acc is NaN, nothing displaces it
            # (ordered compares with NaN are false, and isnan(v) is false
            # for real numbers).
            take = b.or_(b.fcmp_unordered("uno", v, v),
                         b.fcmp_ordered(cmp, v, a))
            return b.select(take, v, a)
        return self._reduce_loop(n, first, stp, ety), ety

    def call(self, node):
        b = self.builder
        import ast as _ast
        if (isinstance(node.func, _ast.Attribute)
                and node.func.attr in ("reshape", "ravel")):
            aty, comps = self._arr_of(node.func.value, node.func.attr)
            total = self._arr_total(aty, comps)
            if node.func.attr == "ravel" or len(node.args) == 1:
                return (comps[0], total), arr_ty(arr_base(aty), 1, True)
            d0 = self.cast(*self.expr(node.args[0]), I64)
            d1 = self.cast(*self.expr(node.args[1]), I64)
            neg1 = ir.Constant(LLTY[I64], -1)
            d0 = b.select(b.icmp_signed("==", d0, neg1),
                          b.sdiv(total, d1), d0)
            d1 = b.select(b.icmp_signed("==", d1, neg1),
                          b.sdiv(total, d0), d1)
            return (comps[0], d0, d1), arr_ty(arr_base(aty), 2, True)
        if (isinstance(node.func, _ast.Attribute)
                and node.func.attr in NP_REDUCTIONS
                and (node.func.attr != "T")
                and not (isinstance(node.func.value, _ast.Name)
                         and node.func.value.id in MATH_MODULES)):
            # method form: x.sum(), (a*b).mean(), ...
            return self._lower_reduction(node.func.attr,
                                         [node.func.value]
                                         + list(node.args))
        if (isinstance(node.func, _ast.Attribute)
                and isinstance(node.func.value, _ast.Name)
                and node.func.value.id in MATH_MODULES
                and node.func.attr in NP_REDUCTIONS):
            return self._lower_reduction(node.func.attr, node.args)
        if False:
            op = None
            aty, comps = self._arr_of(node.args[0], f"np.{op}")
            ety = ARRAY_ELEM[aty]
            n, load = self._flat_load(aty, comps)
            zero = ir.Constant(LLTY[ety], 0 if ety == I64 else 0.0)
            addf = (b.add if ety == I64
                    else (lambda a, v: b.fadd(a, v, flags=self.rff)))
            if op in ("sum", "mean"):
                s = self._reduce_loop(n, zero,
                                      lambda a, i: addf(a, load(i)), ety)
                if op == "mean":
                    return b.fdiv(self.cast(s, ety, F64),
                                  b.sitofp(n, LLTY[F64])), F64
                return s, ety
            if op == "dot":
                aty2, comps2 = self._arr_of(node.args[1], "np.dot")
                e2 = ARRAY_ELEM[aty2]
                _, load2 = self._flat_load(aty2, comps2)
                common = _common_num(ety, e2)
                mulf = b.mul if common == I64 else b.fmul
                addc = b.add if common == I64 else b.fadd
                z = ir.Constant(LLTY[common], 0 if common == I64 else 0.0)
                def stp(a, i):
                    return addc(a, mulf(self.cast(load(i), ety, common),
                                        self.cast(load2(i), e2, common)))
                return self._reduce_loop(n, z, stp, common), common
            first = load(ir.Constant(LLTY[I64], 0))
            cmp = "<" if op == "min" else ">"
            def stp(a, i):
                v = load(i)
                c = (b.icmp_signed(cmp, v, a) if ety == I64
                     else b.fcmp_ordered(cmp, v, a))
                return b.select(c, v, a)
            return self._reduce_loop(n, first, stp, ety), ety
        if (isinstance(node.func, _ast.Attribute)
                and isinstance(node.func.value, _ast.Name)
                and node.func.value.id in MATH_MODULES
                and node.func.attr in ("abs",) + NP_ELEMWISE2 + (
                    "where", "clip", "arange", "linspace")):
            fa = node.func.attr
            I = lambda v: ir.Constant(LLTY[I64], v)
            if fa == "arange":
                args = [self.expr(a) for a in node.args]
                ety = F64 if any(t == F64 for _, t in args) else I64
                vals = [self.cast(v, t, ety) for v, t in args]
                one = ir.Constant(LLTY[ety], 1 if ety == I64 else 1.0)
                zero = ir.Constant(LLTY[ety], 0 if ety == I64 else 0.0)
                if len(vals) == 1:
                    start, stop, step = zero, vals[0], one
                elif len(vals) == 2:
                    start, stop, step = vals[0], vals[1], one
                else:
                    start, stop, step = vals
                if ety == I64:
                    diff = b.sub(stop, start)
                    n = b.sdiv(b.add(diff, b.sub(step, I(1))), step)
                else:
                    ceil = self.intrinsic("llvm.ceil.f64", LLTY[F64],
                                          [LLTY[F64]])
                    n = b.fptosi(b.call(ceil, [b.fdiv(b.fsub(stop, start),
                                                      step)]), LLTY[I64])
                n = b.select(b.icmp_signed("<", n, I(0)), I(0), n)
                if ety == I64:
                    gen = lambda i: b.add(start, b.mul(i, step))
                else:
                    gen = lambda i: b.fadd(start, b.fmul(
                        b.sitofp(i, LLTY[F64]), step))
                return Lazy(ety, n, gen), lazy_of(ety)
            if fa == "linspace":
                a0 = self.cast(*self.expr(node.args[0]), F64)
                a1 = self.cast(*self.expr(node.args[1]), F64)
                n = self.cast(*self.expr(node.args[2]), I64)
                d = b.fdiv(b.fsub(a1, a0),
                           b.sitofp(b.sub(n, I(1)), LLTY[F64]))
                gen = lambda i: b.fadd(a0, b.fmul(b.sitofp(i, LLTY[F64]), d))
                return Lazy(F64, n, gen), "~f64"
            parts = [self.expr(a) for a in node.args]
            seqish = any(t in LAZY or t in ARRAY_ELEM for _, t in parts)
            if fa == "abs":
                if not seqish:
                    v, t = parts[0]
                    return self.call_abs(v, t)
                ety = seq_elem(parts[0][1])
                def emit(e):
                    return self.call_abs(e[0][0], e[0][1])[0]
                return self._combine(ety, parts, emit), lazy_of(ety)
            if fa in NP_ELEMWISE2:
                cmp = "<" if fa == "minimum" else ">"
                def emit(e):
                    (lv, lt), (rv, rt) = e
                    common = _common_num(lt, rt)
                    lv = self.cast(lv, lt, common)
                    rv = self.cast(rv, rt, common)
                    c = (b.fcmp_ordered(cmp, lv, rv) if common == F64
                         else b.icmp_signed(cmp, lv, rv))
                    return b.select(c, lv, rv)
                ety = F64 if any(F64 in (seq_elem(t) or t,)
                                 for _, t in parts) else I64
                out = self._combine(ety, parts, emit)
                return (out, lazy_of(ety)) if seqish else (
                    emit([(v, t) for v, t in parts]), ety)
            if fa == "where":
                cnd, a_, b_ = parts
                ety = F64 if F64 in ((seq_elem(a_[1]) or a_[1]),
                                     (seq_elem(b_[1]) or b_[1])) else I64
                def emit(e):
                    (cv, ct), (av, at), (bv, bt) = e
                    return b.select(self.truthy(cv, ct),
                                    self.cast(av, at, ety),
                                    self.cast(bv, bt, ety))
                return self._combine(ety, parts, emit), lazy_of(ety)
            if fa == "clip":
                x, lo, hi = parts
                ety = F64 if F64 in tuple((seq_elem(t) or t)
                                          for _, t in parts) else I64
                def emit(e):
                    (xv, xt), (lv, lt), (hv, ht) = e
                    xv = self.cast(xv, xt, ety)
                    lv = self.cast(lv, lt, ety)
                    hv = self.cast(hv, ht, ety)
                    lo_ = (b.fcmp_ordered("<", xv, lv) if ety == F64
                           else b.icmp_signed("<", xv, lv))
                    xv = b.select(lo_, lv, xv)
                    hi_ = (b.fcmp_ordered(">", xv, hv) if ety == F64
                           else b.icmp_signed(">", xv, hv))
                    return b.select(hi_, hv, xv)
                out = self._combine(ety, parts, emit)
                return (out, lazy_of(ety)) if seqish else (
                    emit(parts), ety)
        if (isinstance(node.func, _ast.Attribute)
                and isinstance(node.func.value, _ast.Name)
                and node.func.value.id in MATH_MODULES
                and node.func.attr in MATH_FNS):
            name = node.func.attr
            parts = [self.expr(a) for a in node.args]
            if any(t in LAZY or t in ARRAY_ELEM for _, t in parts):
                def emit(e):
                    v = self.cast(e[0][0], e[0][1], F64)
                    if name == "pow":
                        w = self.cast(e[1][0], e[1][1], F64)
                        p = self.intrinsic("llvm.pow.f64", LLTY[F64],
                                           [LLTY[F64]] * 2)
                        return b.call(p, [v, w])
                    f = self.intrinsic(self.MATH_INTRIN[name], LLTY[F64],
                                       [LLTY[F64]])
                    return b.call(f, [v])
                return self._combine(F64, parts, emit), "~f64"
            args = [self.cast(v, t, F64) for v, t in parts]
            if name == "pow":
                p = self.intrinsic("llvm.pow.f64", LLTY[F64], [LLTY[F64]] * 2)
                return b.call(p, args[:2]), F64
            f = self.intrinsic(self.MATH_INTRIN[name], LLTY[F64], [LLTY[F64]])
            return b.call(f, [args[0]]), F64
        fname = node.func.id
        if fname == "len":
            aty, comps = self._arr_of(node.args[0], "len()")
            return self._arr_len(aty, comps), I64
        if fname in GPU_INTRINSICS:
            # AMDGPU has no direct block_dim sreg: read workgroup_size.x from
            # the HSA dispatch packet (offset 4, i16). Verified against real
            # llvm-mc/llc for gfx90a.
            if self.gpu == "amd" and fname == "block_dim":
                i32 = ir.IntType(32)
                i16p = ir.IntType(16).as_pointer(4)
                i8p4 = ir.IntType(8).as_pointer(4)
                dp = self.intrinsic("llvm.amdgcn.dispatch.ptr", i8p4, [])
                ptr = b.call(dp, [])
                gep = b.gep(ptr, [ir.Constant(LLTY[I64], 4)])
                p16 = b.bitcast(gep, i16p)
                wgs = b.load(p16)
                wgs.align = 4
                return b.zext(wgs, LLTY[I64]), I64
            table = GPU_LOWER.get(self.gpu)
            if table is None or fname not in table:
                raise UnsupportedError(
                    f"{fname}() requires a GPU target with that intrinsic "
                    f"(cuda: all three; amd: thread_id/block_id/block_dim)")
            f = self.intrinsic(table[fname], ir.IntType(32), [])
            return b.zext(b.call(f, []), LLTY[I64]), I64
        if fname == "abs":
            v, ty = self.expr(node.args[0])
            if ty == F64:
                fabs = self.intrinsic("llvm.fabs.f64", LLTY[F64], [LLTY[F64]])
                return b.call(fabs, [v]), F64
            neg = b.neg(v)
            return b.select(b.icmp_signed("<", v, ir.Constant(LLTY[I64], 0)), neg, v), I64
        if fname == "float":
            v, ty = self.expr(node.args[0])
            return self.cast(v, ty, F64), F64
        if fname == "int":
            v, ty = self.expr(node.args[0])
            return self.cast(v, ty, I64), I64
        if fname == self.func_ast.name:  # self-recursion
            args = []
            for a, farg in zip(node.args, self.func_ast.args.args):
                v, ty = self.expr(a)
                args.append(self.cast(v, ty, self.arg_types[farg.arg]))
            return b.call(self.fn, args), self.ret_type
        raise UnsupportedError(f"unsupported call: {fname}")

    def call_abs(self, v, ty):
        b = self.builder
        if ty == F64:
            fabs = self.intrinsic("llvm.fabs.f64", LLTY[F64], [LLTY[F64]])
            return b.call(fabs, [v]), F64
        neg = b.neg(v)
        return b.select(b.icmp_signed(
            "<", v, ir.Constant(LLTY[I64], 0)), neg, v), I64

    def intrinsic(self, name, ret, argtys):
        for f in self.module.functions:
            if f.name == name:
                return f
        return ir.Function(self.module, ir.FunctionType(ret, argtys), name=name)

    # Python semantics: // floors toward -inf; % takes the divisor's sign.
    # LLVM sdiv/srem/frem truncate toward zero, so adjust when signs differ.
    def _py_floordiv_i64(self, a, d):
        b = self.builder
        zero = ir.Constant(LLTY[I64], 0)
        q = b.sdiv(a, d)
        r = b.srem(a, d)
        diff_sign = b.icmp_signed("<", b.xor(a, d), zero)
        adj = b.and_(diff_sign, b.icmp_signed("!=", r, zero))
        return b.sub(q, b.zext(adj, LLTY[I64]))

    def _py_mod_i64(self, a, d):
        b = self.builder
        zero = ir.Constant(LLTY[I64], 0)
        r = b.srem(a, d)
        diff_sign = b.icmp_signed("<", b.xor(a, d), zero)
        adj = b.and_(diff_sign, b.icmp_signed("!=", r, zero))
        return b.add(r, b.select(adj, d, zero))

    def _py_mod_f64(self, a, d):
        b = self.builder
        zero = ir.Constant(LLTY[F64], 0.0)
        r = b.frem(a, d)
        rneg = b.fcmp_ordered("<", r, zero)
        dneg = b.fcmp_ordered("<", d, zero)
        adj = b.and_(b.xor(rneg, dneg), b.fcmp_ordered("!=", r, zero))
        return b.select(adj, b.fadd(r, d), r)

    def binop(self, op, left, right):
        b = self.builder
        (lv, lt), (rv, rt) = left, right
        if isinstance(op, ast.Div):
            return b.fdiv(self.cast(lv, lt, F64), self.cast(rv, rt, F64), flags=self.ff), F64
        if isinstance(op, ast.Pow):
            if F64 in (lt, rt):
                p = self.intrinsic("llvm.pow.f64", LLTY[F64], [LLTY[F64]] * 2)
                return b.call(p, [self.cast(lv, lt, F64), self.cast(rv, rt, F64)]), F64
            p = self.intrinsic("llvm.powi.f64.i32", LLTY[F64], [LLTY[F64], ir.IntType(32)])
            res = b.call(p, [self.cast(lv, lt, F64), b.trunc(rv, ir.IntType(32))])
            return b.fptosi(res, LLTY[I64]), I64
        common = _common_num(lt, rt)
        lv, rv = self.cast(lv, lt, common), self.cast(rv, rt, common)
        if isinstance(op, ast.FloorDiv) and common == I64:
            return self._py_floordiv_i64(lv, rv), I64
        if isinstance(op, ast.Mod):
            if common == I64:
                return self._py_mod_i64(lv, rv), I64
            return self._py_mod_f64(lv, rv), common
        table_i = {ast.Add: b.add, ast.Sub: b.sub, ast.Mult: b.mul,
                   ast.FloorDiv: b.sdiv, ast.Mod: b.srem,
                   ast.BitAnd: b.and_, ast.BitOr: b.or_, ast.BitXor: b.xor,
                   ast.LShift: b.shl, ast.RShift: b.ashr}
        import functools as _ft
        _f = lambda op: _ft.partial(op, flags=self.ff)
        table_f = {ast.Add: _f(b.fadd), ast.Sub: _f(b.fsub),
                   ast.Mult: _f(b.fmul), ast.FloorDiv: None,
                   ast.Mod: _f(b.frem)}
        t = table_f if common in (F64, F32) else table_i
        fn = t.get(type(op))
        if fn is None:
            if common == F64 and isinstance(op, ast.FloorDiv):
                fl = self.intrinsic("llvm.floor.f64", LLTY[F64], [LLTY[F64]])
                return b.call(fl, [b.fdiv(lv, rv)]), F64
            raise UnsupportedError(f"unsupported operator: {type(op).__name__}")
        return fn(lv, rv), common
