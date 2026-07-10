"""Automatic parallelization: `@jit(parallel=True)` promotes the
outermost eligible `for i in range(...)` loop to `prange`, so the user
writes an ordinary Python loop and gets multithreaded execution — the
ergonomic that Taichi gets from auto-parallel top-level `for`, but with
no DSL and no new syntax.

This is a pure AST rewrite: find the single top-level `for i in range(n)`
loop, replace its iterator `range` with `prange`, and hand the result to
the existing `parallel.make_parallel` machinery (chunked, GIL-released
thread pool, reassociated reductions). If the function doesn't match the
parallelizable shape, we raise UnsupportedError and the caller compiles it
serially — auto-parallel never changes results, only scheduling.

Eligibility is intentionally conservative and identical to what
`parallel.analyze` already accepts: exactly one top-level range loop,
simple pre-assignments, a body of loop-local assigns / array writes / at
most one `acc +=` reduction, and a single return. Anything else stays
serial.
"""
import ast
import copy

from .errors import UnsupportedError


def rewrite_range_to_prange(fn_ast):
    """Return a copy of fn_ast with the single top-level `range` loop's
    iterator replaced by `prange`. Raise UnsupportedError if there isn't
    exactly one eligible top-level range loop."""
    tree = copy.deepcopy(fn_ast)
    body = tree.body
    range_loops = [
        i for i, s in enumerate(body)
        if isinstance(s, ast.For) and isinstance(s.iter, ast.Call)
        and isinstance(s.iter.func, ast.Name)
        and s.iter.func.id in ("range", "prange")
    ]
    if len(range_loops) != 1:
        raise UnsupportedError(
            "parallel=True needs exactly one top-level range loop")
    loop = body[range_loops[0]]
    if loop.iter.func.id == "range":
        if len(loop.iter.args) not in (1, 2):
            raise UnsupportedError(
                "parallel loop must be range(n) or range(lo, hi) (no step)")
        loop.iter.func = ast.copy_location(ast.Name(id="prange",
                                                    ctx=ast.Load()),
                                           loop.iter.func)
    ast.fix_missing_locations(tree)
    return tree
