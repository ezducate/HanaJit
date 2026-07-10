"""Inlining of jitted helper functions — cross-function calls without the
dispatch cost.

The idea (borrowed from Taichi's kernel/`ti.func` split, but with no DSL):
a small `@jit` function called from inside another `@jit` function should
be *inlined* at the AST level before codegen, not dispatched at runtime.
This turns hanajit's one real codegen weakness — cross-function calls cost
a full dispatch — into ordinary in-loop arithmetic the optimizer and the
fusion engine can see through.

Mechanism, entirely on trees `ast.parse` already produced (no new syntax,
no parser):
  1. `@jit` registers each function's source AST in _REGISTRY by name.
  2. Before type inference, `inline_calls` walks the caller. Any `Call` to
     a registered helper whose body is a single `return <expr>` (or a
     straight-line body of assignments ending in one return) is replaced:
     the helper's parameters are bound to the call's arguments via
     uniquely-renamed local assignments, the body is spliced in, and the
     call expression becomes the helper's return value.
  3. Renaming (`__hj_inl{N}_{param}`) prevents any capture between caller
     and callee locals. Recursion and self-reference are never inlined
     (they must dispatch), and depth is bounded.

Only pure expression/assignment helpers are inlined. Anything with loops,
branches, or multiple returns is left as a call (still correct — it simply
dispatches). This is conservative on purpose: an inliner that changes
results is worse than no inliner.
"""
import ast
import copy

# name -> FunctionDef (deep-copied at registration; never mutated in place)
_REGISTRY = {}
_MAX_DEPTH = 8


def register(name, fn_ast):
    _REGISTRY[name] = copy.deepcopy(fn_ast)


def is_registered(name):
    return name in _REGISTRY


class _Renamer(ast.NodeTransformer):
    """Rename every Name load/store in a helper body to a unique prefix,
    except references to the helper's own parameters (handled by the
    caller through bound temporaries) and global/builtins we leave alone."""

    def __init__(self, prefix, params, bound):
        self.prefix = prefix
        self.params = params      # param name -> bound temp name
        self.bound = bound        # set of names that are helper-locals
        self.local = set()

    def visit_Name(self, node):
        if node.id in self.params:
            return ast.copy_location(
                ast.Name(id=self.params[node.id], ctx=node.ctx), node)
        if node.id in self.bound:
            return ast.copy_location(
                ast.Name(id=self.prefix + node.id, ctx=node.ctx), node)
        return node


def _inlinable_body(helper):
    """Return the helper's (assignments, return_expr) if it is a
    straight-line pure body, else None."""
    body = helper.body
    if not body or not isinstance(body[-1], ast.Return) \
            or body[-1].value is None:
        return None
    assigns = body[:-1]
    for s in assigns:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)):
            return None
    # reject control flow / nested defs / additional returns anywhere
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, (ast.For, ast.While, ast.If, ast.With,
                              ast.FunctionDef, ast.Lambda)):
                return None
        if s is not body[-1] and isinstance(s, ast.Return):
            return None
    return assigns, body[-1].value


class _CallInliner(ast.NodeTransformer):
    def __init__(self, self_name, depth):
        self.self_name = self_name
        self.depth = depth
        self.counter = [0]
        self.prelude = []          # assignments to emit before current stmt
        self.changed = False

    def _inline_call(self, node):
        if not (isinstance(node.func, ast.Name)
                and is_registered(node.func.id)):
            return None
        name = node.func.id
        if name == self.self_name:           # never inline recursion
            return None
        if node.keywords:
            return None
        helper = _REGISTRY[name]
        params = [a.arg for a in helper.args.args]
        if len(node.args) != len(params):
            return None
        parsed = _inlinable_body(helper)
        if parsed is None:
            return None
        assigns, ret_expr = parsed

        idx = self.counter[0]
        self.counter[0] += 1
        prefix = f"__hj_inl{self.depth}_{idx}_"
        # bind each argument (already inlined itself) to a unique temp
        pmap = {}
        for p, argexpr in zip(params, node.args):
            tmp = prefix + "arg_" + p
            pmap[p] = tmp
            newarg = self.visit(copy.deepcopy(argexpr))  # nested inline
            self.prelude.append(ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())], value=newarg))
        # helper's own locals (assignment targets) get the unique prefix
        locals_ = {s.targets[0].id for s in assigns}
        rn = _Renamer(prefix, pmap, locals_)
        for s in assigns:
            s2 = rn.visit(copy.deepcopy(s))
            self.prelude.append(s2)
        result = rn.visit(copy.deepcopy(ret_expr))
        self.changed = True
        return result

    def visit_Call(self, node):
        self.generic_visit(node)
        inlined = self._inline_call(node)
        return inlined if inlined is not None else node

    def _process_stmt_list(self, stmts):
        out = []
        for s in stmts:
            saved = self.prelude
            self.prelude = []
            s2 = self.visit(s)
            out.extend(self.prelude)
            self.prelude = saved
            out.append(s2)
        return out

    def visit_FunctionDef(self, node):
        node.body = self._process_stmt_list(node.body)
        return node

    def visit_For(self, node):
        node.target = node.target
        node.iter = self.visit(node.iter)
        node.body = self._process_stmt_list(node.body)
        node.orelse = self._process_stmt_list(node.orelse)
        return node

    def visit_While(self, node):
        node.test = self.visit(node.test)
        node.body = self._process_stmt_list(node.body)
        node.orelse = self._process_stmt_list(node.orelse)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body = self._process_stmt_list(node.body)
        node.orelse = self._process_stmt_list(node.orelse)
        return node


def inline_calls(fn_ast):
    """Return a new FunctionDef with registered helper calls inlined.
    Idempotent-ish: iterates to a fixpoint up to _MAX_DEPTH so a helper
    that calls a helper is fully flattened. Never mutates the input."""
    tree = copy.deepcopy(fn_ast)
    self_name = tree.name
    for depth in range(_MAX_DEPTH):
        inl = _CallInliner(self_name, depth)
        tree = inl.visit(tree)
        ast.fix_missing_locations(tree)
        if not inl.changed:
            break
    return tree
