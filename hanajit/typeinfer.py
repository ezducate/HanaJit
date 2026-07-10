"""Type inference over Python AST (uses CPython's own `ast` parser output).

Fixpoint abstract interpretation: variables get types (i64 / f64 / bool),
mixed int/float unifies to f64. Anything outside the supported subset raises
UnsupportedError, which triggers fallback to the CPython interpreter
(this is how full-ecosystem compatibility is preserved).
"""
import ast
from .errors import UnsupportedError

I64, F64, BOOL = "i64", "f64", "bool"
PF64, PI64 = "f64*", "i64*"
# array kinds: base[<ndim><c|s>]  e.g. f64[1c] contiguous, i64[2s] strided
AF64, AI64 = "f64[1c]", "i64[1c]"
F32 = "f32"
POINTER_ELEM = {PF64: F64, PI64: I64}
ARRAY_ELEM = {f"{b}[{n}{c}]": (F64 if b == "f64" else
                               (F32 if b == "f32" else I64))
              for b in ("f64", "i64", "f32") for n in "12" for c in "cs"}
ELEM = {**POINTER_ELEM, **ARRAY_ELEM}
NP_REDUCTIONS = ("sum", "dot", "min", "max", "mean", "prod", "argmin",
                 "argmax", "any", "all", "var", "std", "count_nonzero")
NP_ELEMWISE2 = ("minimum", "maximum")   # np.minimum(a, b)
LAZY = {"~f64": F64, "~i64": I64, "~bool": BOOL, "~f32": F32}


def lazy_of(elem):
    return {F64: "~f64", I64: "~i64", BOOL: "~bool", F32: "~f32"}[elem]


def seq_elem(t):
    """Element type of any 1-D-iterable value: lazy expr or array."""
    if t in LAZY:
        return LAZY[t]
    if t in ARRAY_ELEM and (arr_nd(t) == 1 or arr_contig(t)):
        return ARRAY_ELEM[t]
    return None


def arr_ty(base, nd, contig):
    return f"{base}[{nd}{'c' if contig else 's'}]"


def arr_base(t):
    return t.split("[")[0]


def arr_nd(t):
    return int(t[-3])


def arr_contig(t):
    return t[-2] == "c"
GPU_INTRINSICS = ("thread_id", "block_id", "block_dim")
MATH_FNS = ("sqrt", "exp", "log", "sin", "cos", "floor", "ceil", "pow",
            "fabs")
MATH_MODULES = ("math", "np", "numpy")


def unify(a, b):
    if a == b:
        return a
    floats = {F32, F64}
    if {a, b} <= {I64, F64, F32, BOOL}:
        # highest float wins; f64 > f32 > int > bool
        if F64 in (a, b):
            return F64
        if F32 in (a, b):
            return F32
        return I64
    raise UnsupportedError(f"cannot unify types {a} and {b}")


class TypeInferencer(ast.NodeVisitor):
    def __init__(self, func_ast: ast.FunctionDef, arg_types: dict):
        self.func_ast = func_ast
        self.env = dict(arg_types)   # name -> type
        self.ret_type = None
        self.changed = True

    def run(self):
        iters = 0
        while self.changed and iters < 20:
            self.changed = False
            for stmt in self.func_ast.body:
                self.visit(stmt)
            iters += 1
        if self.ret_type is None:
            raise UnsupportedError("function must have a return statement")
        return dict(self.env), self.ret_type

    def _reduction_type(self, op, atys):
        elems = [seq_elem(t) for t in atys]
        if not elems or any(e is None for e in elems):
            raise UnsupportedError(
                f"np.{op} requires 1-D array/expression arguments "
                "(strided 2-D views not compiled yet)")
        e = elems[0]
        if op in ("argmin", "argmax", "count_nonzero"):
            return I64
        if op in ("any", "all"):
            return BOOL
        if op in ("mean", "var", "std"):
            return F64
        if op == "dot":
            return unify(elems[0], elems[1])
        return I64 if e == BOOL and op == "sum" else e

    # ---- helpers ----
    def set_var(self, name, ty):
        old = self.env.get(name)
        new = ty if old is None else unify(old, ty)
        if new != old:
            self.env[name] = new
            self.changed = True

    # ---- statements ----
    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise UnsupportedError("multiple assignment targets not supported")
        t = node.targets[0]
        if isinstance(t, ast.Subscript):        # x[i] = v / x[i,j] = v
            base = self.expr(t.value)
            if base not in ELEM:
                raise UnsupportedError("subscript store requires an array/pointer")
            if isinstance(t.slice, ast.Slice):   # y[a:b:c] = expr
                if base not in ARRAY_ELEM or arr_nd(base) != 1:
                    raise UnsupportedError("slice-store target must be 1-D")
                for part in (t.slice.lower, t.slice.upper, t.slice.step):
                    if part is not None and self.expr(part) != I64:
                        raise UnsupportedError("slice bounds must be ints")
                v = self.expr(node.value)
                unify(ELEM[base], seq_elem(v) or v)
                return
            if isinstance(t.slice, ast.Tuple):
                if base not in ARRAY_ELEM or arr_nd(base) != 2 or len(
                        t.slice.elts) != 2 or any(
                        self.expr(e) != I64 for e in t.slice.elts):
                    raise UnsupportedError("2-D store index must be two ints")
            elif self.expr(t.slice) != I64:
                raise UnsupportedError("subscript index must be an integer")
            elif base in ARRAY_ELEM and arr_nd(base) == 2:
                raise UnsupportedError("cannot assign a row; use x[i, j]")
            unify(ELEM[base], self.expr(node.value))
            return
        if not isinstance(t, ast.Name):
            raise UnsupportedError("only simple `x = expr` assignment supported")
        self.set_var(t.id, self.expr(node.value))

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise UnsupportedError("only simple augmented assignment supported")
        cur = self.env.get(node.target.id)
        if cur is None:
            raise UnsupportedError(f"variable {node.target.id} used before assignment")
        rhs = self.expr(node.value)
        ty = F64 if isinstance(node.op, ast.Div) else unify(cur, rhs)
        self.set_var(node.target.id, ty)

    def visit_Return(self, node):
        if node.value is None:
            raise UnsupportedError("bare `return` not supported")
        ty = self.expr(node.value)
        self.ret_type = ty if self.ret_type is None else unify(self.ret_type, ty)

    def visit_If(self, node):
        self.expr(node.test)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_While(self, node):
        self.expr(node.test)
        for s in node.body:
            self.visit(s)

    def visit_For(self, node):
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id in ("range", "prange")
                and isinstance(node.target, ast.Name)):
            raise UnsupportedError("only `for i in range(...)` loops supported")
        for a in node.iter.args:
            if self.expr(a) != I64:
                raise UnsupportedError("range() arguments must be integers")
        self.set_var(node.target.id, I64)
        for s in node.body:
            self.visit(s)
        if node.orelse:
            raise UnsupportedError("for/else not supported")

    def visit_Expr(self, node):
        pass  # e.g. docstrings

    def visit_Pass(self, node):
        pass

    def visit_Break(self, node):
        pass

    def visit_Continue(self, node):
        pass

    def generic_visit(self, node):
        raise UnsupportedError(f"unsupported statement: {type(node).__name__}")

    # ---- expressions ----
    def expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BOOL
            if isinstance(node.value, int):
                return I64
            if isinstance(node.value, float):
                return F64
            raise UnsupportedError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Name):
            ty = self.env.get(node.id)
            if ty is None:
                # may be defined later in loop body; assume i64 provisionally
                raise UnsupportedError(f"unknown variable {node.id!r}")
            return ty
        if isinstance(node, ast.BinOp):
            l, r = self.expr(node.left), self.expr(node.right)
            if l in POINTER_ELEM or r in POINTER_ELEM:
                raise UnsupportedError("raw-pointer arithmetic not supported")
            le, re_ = seq_elem(l), seq_elem(r)
            if le is not None or re_ is not None:   # fused lazy expression
                el = le or l
                er = re_ or r
                if el is None or er is None or el not in (I64, F64, BOOL) \
                        or er not in (I64, F64, BOOL):
                    raise UnsupportedError(
                        "array expressions combine 1-D arrays and scalars")
                if isinstance(node.op, ast.Div):
                    return "~f64"
                if isinstance(node.op, ast.Pow):
                    return lazy_of(F64 if F64 in (el, er) else I64)
                e = unify(el, er)
                return lazy_of(I64 if e == BOOL else e)
            if isinstance(node.op, ast.Div):
                return F64
            if isinstance(node.op, ast.Pow):
                return F64 if F64 in (l, r) else I64
            t = unify(l, r)
            return I64 if t == BOOL else t   # bool arithmetic -> int
        if isinstance(node, ast.UnaryOp):
            t = self.expr(node.operand)
            e = seq_elem(t)
            if e is not None and t not in (I64, F64, BOOL):
                return "~bool" if isinstance(node.op, ast.Not) else lazy_of(e)
            if isinstance(node.op, ast.Not):
                return BOOL
            return t
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise UnsupportedError("chained comparisons not supported")
            l, r = self.expr(node.left), self.expr(node.comparators[0])
            if seq_elem(l) not in (None, l) or seq_elem(r) not in (None, r) \
                    or l in LAZY or r in LAZY or l in ARRAY_ELEM \
                    or r in ARRAY_ELEM:
                return "~bool"
            unify(l, r)
            return BOOL
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                self.expr(v)
            return BOOL
        if isinstance(node, ast.Subscript):
            # x.shape[k]
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"):
                if self.expr(node.value.value) not in ARRAY_ELEM:
                    raise UnsupportedError(".shape on non-array")
                return I64
            base = self.expr(node.value)
            if base not in ELEM:
                raise UnsupportedError("subscript load requires an array/pointer")
            if isinstance(node.slice, ast.Slice):        # x[a:b:c] view
                if base not in ARRAY_ELEM or arr_nd(base) != 1:
                    raise UnsupportedError("slicing supported on 1-D arrays")
                for part in (node.slice.lower, node.slice.upper,
                             node.slice.step):
                    if part is not None and self.expr(part) != I64:
                        raise UnsupportedError("slice bounds must be ints")
                st = node.slice.step
                unit = st is None or (isinstance(st, ast.Constant)
                                      and st.value == 1)
                return arr_ty(arr_base(base), 1,
                              arr_contig(base) and unit)
            if isinstance(node.slice, ast.Tuple):        # x[i, j]
                if base not in ARRAY_ELEM or arr_nd(base) != 2:
                    raise UnsupportedError("tuple index needs a 2-D array")
                if len(node.slice.elts) != 2 or any(
                        self.expr(e) != I64 for e in node.slice.elts):
                    raise UnsupportedError("2-D index must be two ints")
                return ELEM[base]
            if self.expr(node.slice) != I64:
                raise UnsupportedError("subscript index must be an integer")
            if base in ARRAY_ELEM and arr_nd(base) == 2:  # row view
                return arr_ty(arr_base(base), 1, arr_contig(base))
            return ELEM[base]
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name)
                    and node.value.id in MATH_MODULES
                    and node.attr in ("pi", "e")):
                return F64
            if node.attr == "T":
                base = self.expr(node.value)
                if base not in ARRAY_ELEM or arr_nd(base) != 2:
                    raise UnsupportedError(".T supported on 2-D arrays")
                return arr_ty(arr_base(base), 2, False)
            raise UnsupportedError(f"unsupported attribute .{node.attr}")
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in NP_REDUCTIONS
                    and not (isinstance(node.func.value, ast.Name)
                             and node.func.value.id in MATH_MODULES)):
                base = self.expr(node.func.value)     # x.sum() method form
                return self._reduction_type(node.func.attr, [base])
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("reshape", "ravel")):
                base = self.expr(node.func.value)
                if base not in ARRAY_ELEM or not arr_contig(base):
                    raise UnsupportedError(
                        f".{node.func.attr} requires a contiguous array")
                for a in node.args:
                    if self.expr(a) != I64:
                        raise UnsupportedError("reshape dims must be ints")
                if node.func.attr == "ravel" or len(node.args) == 1:
                    return arr_ty(arr_base(base), 1, True)
                if len(node.args) == 2:
                    return arr_ty(arr_base(base), 2, True)
                raise UnsupportedError("reshape supports 1 or 2 dims")
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in MATH_MODULES
                    and node.func.attr in NP_REDUCTIONS):
                atys = [self.expr(a) for a in node.args]
                return self._reduction_type(node.func.attr, atys)
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in MATH_MODULES):
                fa = node.func.attr
                if fa in MATH_FNS or fa == "abs":
                    tys = [self.expr(a) for a in node.args]
                    if any(seq_elem(t) is not None and t not in
                           (I64, F64, BOOL) for t in tys):
                        return "~f64" if fa != "abs" else lazy_of(
                            seq_elem(tys[0]))
                    if fa == "abs":
                        return tys[0]
                    return F64
                if fa in NP_ELEMWISE2:
                    a, b = [self.expr(x) for x in node.args]
                    ea = seq_elem(a) or a
                    eb = seq_elem(b) or b
                    e = unify(ea, eb)
                    return lazy_of(e) if (a not in (I64, F64, BOOL)
                                          or b not in (I64, F64, BOOL)) else e
                if fa == "where":
                    c, a, b = [self.expr(x) for x in node.args]
                    e = unify(seq_elem(a) or a, seq_elem(b) or b)
                    return lazy_of(e)
                if fa == "clip":
                    x, lo, hi = [self.expr(t) for t in node.args]
                    e = unify(unify(seq_elem(x) or x, seq_elem(lo) or lo),
                              seq_elem(hi) or hi)
                    return lazy_of(e) if x not in (I64, F64) else e
                if fa == "arange":
                    tys = [self.expr(a) for a in node.args]
                    return "~f64" if F64 in tys else "~i64"
                if fa == "linspace":
                    for a in node.args:
                        self.expr(a)
                    return "~f64"
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in GPU_INTRINSICS:
                    return I64
                if fname == "len":
                    if self.expr(node.args[0]) not in ARRAY_ELEM:
                        raise UnsupportedError(
                            "len() supported only on array arguments")
                    return I64
                if fname in ("abs",):
                    return self.expr(node.args[0])
                if fname in ("float",):
                    self.expr(node.args[0]); return F64
                if fname in ("int",):
                    self.expr(node.args[0]); return I64
                if fname == self.func_ast.name:  # recursion
                    for a in node.args:
                        self.expr(a)
                    return self.ret_type or I64
            raise UnsupportedError("only abs/int/float/self-recursion calls supported")
        if isinstance(node, ast.IfExp):
            self.expr(node.test)
            return unify(self.expr(node.body), self.expr(node.orelse))
        raise UnsupportedError(f"unsupported expression: {type(node).__name__}")
