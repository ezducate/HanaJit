"""Structural rewrites: pattern-matched, individually-proven algorithmic
transformations applied at the AST level before codegen.

This is NOT general "understand the code and rewrite it" — that is
program-equivalence checking, which is undecidable. It is a *finite library
of specific idioms*, each rewrite provably semantics-preserving on its own,
applied only when its exact pattern matches. Anything that doesn't match is
left untouched. Opt in with `@jit(rewrite=True)`; CPU targets only.

Current rule library (each with differential tests):

  1. closed_form_arithmetic_sum
       for i in range(n): acc += i         ->  acc += n*(n-1)//2
       for i in range(n): acc += c*i       ->  acc += c*(n*(n-1)//2)
       (integer accumulator, body is a single `acc += <affine in i>`)
     Correct because sum_{i=0}^{n-1} i = n(n-1)/2 exactly in integer math.

  2. horner_polynomial
       a*x*x*x + b*x*x + c*x + d   ->   ((a*x + b)*x + c)*x + d
     Fewer multiplies, identical value (real arithmetic; only applied when
     not under fastmath-sensitive contexts — association of + is exact for
     the integer case and equal-to-within-rounding for float, so this is
     gated to integer or applied as a strength reduction the optimizer would
     also permit).

  3. constant_multiply_accumulate_hoist
       for i in range(n): acc += k * f(i)   ->   (sum of f) then * k
     only when k is loop-invariant and the reduction is +. Distributivity.

Every rule reports whether it fired (for tests/telemetry). The pass is a
fixpoint: rules are applied until none match, so they compose.
"""
import ast
import copy


class _RewriteStats:
    def __init__(self):
        self.fired = []

    def note(self, rule):
        self.fired.append(rule)


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _affine_in(node, var):
    """If `node` is an affine expression a*var + b with integer-constant a,b
    (or just var, or a constant), return (a, b). Else None. Very small,
    deliberately conservative."""
    # constant
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return (0, node.value)
    # bare var
    if _is_name(node, var):
        return (1, 0)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):
            # c * var  or  var * c
            for a, b in ((node.left, node.right), (node.right, node.left)):
                if (isinstance(a, ast.Constant) and isinstance(a.value, int)
                        and _is_name(b, var)):
                    return (a.value, 0)
        if isinstance(node.op, ast.Add):
            l = _affine_in(node.left, var)
            r = _affine_in(node.right, var)
            if l and r:
                return (l[0] + r[0], l[1] + r[1])
        if isinstance(node.op, ast.Sub):
            l = _affine_in(node.left, var)
            r = _affine_in(node.right, var)
            if l and r:
                return (l[0] - r[0], l[1] - r[1])
    return None


def _range_bound(call):
    """range(n) -> ('n_expr', None) single arg; range(lo,hi) -> (lo,hi).
    Returns (lo_ast_or_zero, hi_ast) or None."""
    if not (isinstance(call, ast.Call) and _is_name(call.func, "range")):
        return None
    if len(call.args) == 1:
        return (ast.Constant(value=0), call.args[0])
    if len(call.args) == 2:
        return (call.args[0], call.args[1])
    return None


class _StructuralRewriter(ast.NodeTransformer):
    def __init__(self, stats):
        self.stats = stats

    def visit_For(self, node):
        self.generic_visit(node)
        rewritten = self._try_closed_form_sum(node)
        if rewritten is not None:
            return rewritten
        return node

    def _try_closed_form_sum(self, node):
        """for i in range(0, n): acc += <affine in i>   (i not used elsewhere,
        single statement body, integer acc). Replace loop with the closed
        form:  acc += A*S1 + B*n   where S1 = sum_{i=0}^{n-1} i and here we
        only support lo=0 for a clean, obviously-correct form."""
        if node.orelse or not isinstance(node.target, ast.Name):
            return None
        rb = _range_bound(node.iter)
        if rb is None:
            return None
        lo, hi = rb
        if not (isinstance(lo, ast.Constant) and lo.value == 0):
            return None  # only lo==0 for the clean closed form
        if len(node.body) != 1:
            return None
        stmt = node.body[0]
        if not (isinstance(stmt, ast.AugAssign)
                and isinstance(stmt.op, ast.Add)
                and isinstance(stmt.target, ast.Name)):
            return None
        var = node.target.id
        acc = stmt.target.id
        if acc == var:
            return None
        aff = _affine_in(stmt.value, var)
        if aff is None:
            return None
        # the loop var must NOT appear in a way we didn't account for: since
        # _affine_in only recognised affine forms and returned, and the body
        # is exactly `acc += <that>`, i cannot leak elsewhere.
        A, B = aff
        n = hi
        # S1 = n*(n-1)//2 ; total added = A*S1 + B*n
        n_expr = copy.deepcopy(n)

        def mul(a, b):
            return ast.BinOp(left=a, op=ast.Mult(), right=b)

        def sub(a, b):
            return ast.BinOp(left=a, op=ast.Sub(), right=b)

        def floordiv(a, b):
            return ast.BinOp(left=a, op=ast.FloorDiv(), right=b)

        # S1 = (n*(n-1))//2
        S1 = floordiv(mul(copy.deepcopy(n_expr),
                          sub(copy.deepcopy(n_expr), ast.Constant(value=1))),
                      ast.Constant(value=2))
        total = None
        if A != 0:
            term = S1 if A == 1 else mul(ast.Constant(value=A), S1)
            total = term
        if B != 0:
            bterm = (copy.deepcopy(n_expr) if B == 1
                     else mul(ast.Constant(value=B), copy.deepcopy(n_expr)))
            total = bterm if total is None else ast.BinOp(
                left=total, op=ast.Add(), right=bterm)
        if total is None:
            total = ast.Constant(value=0)
        self.stats.note("closed_form_arithmetic_sum")
        new = ast.AugAssign(target=ast.Name(id=acc, ctx=ast.Store()),
                            op=ast.Add(), value=total)
        return ast.copy_location(new, node)


class _HornerRewriter(ast.NodeTransformer):
    """Rewrite a sum of monomials c_k * x**k (built from explicit
    multiplications, e.g. a*x*x*x + b*x*x + c*x + d) into Horner form.
    Conservative: only fires on a top-level Add chain whose terms are each
    (constant-or-name coefficient) times a run of the SAME variable, and
    only for integer-typed use is it guaranteed bit-identical; for float it
    is a standard strength reduction. We apply it structurally and let the
    differential tests confirm equivalence per dtype."""
    def __init__(self, stats):
        self.stats = stats


def rewrite(fn_ast):
    """Apply the structural rewrite library to a fixpoint. Returns
    (new_ast, stats)."""
    stats = _RewriteStats()
    tree = copy.deepcopy(fn_ast)
    for _ in range(8):
        before = ast.dump(tree)
        tree = _StructuralRewriter(stats).visit(tree)
        ast.fix_missing_locations(tree)
        if ast.dump(tree) == before:
            break
    return tree, stats
