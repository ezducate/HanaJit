"""Parallel loops: `for i in prange(...)` partitions across threads.

How it works: the function is rewritten at the AST level into a *chunk
kernel* `__chunk(lo, hi, <original args>)` whose loop runs `range(lo, hi)`,
compiled with nogil=True. The driver splits the full iteration space into
one chunk per worker and launches them on a thread pool — the GIL is
released inside each chunk, so chunks execute truly in parallel across
cores. A single `acc += ...` reduction is supported: each chunk starts from
the neutral element and returns its partial; the driver sums partials and
adds the original initial value once. Array writes to `y[i]` are safe
because chunks own disjoint index ranges (overlapping writes across
iterations are the user's responsibility, as with numba's prange).

Supported pattern (v1):
    <simple scalar assignments>          # replicated into each chunk
    for i in prange(n) | prange(lo, hi):
        <body: loop-local assigns, array reads/writes, one `acc +=` var>
    return <expression of acc and scalar parameters>

Anything else degrades to the serial compiled loop (prange == range), with
a note via verbose=True. Floating-point reductions are reassociated across
chunks (same contract as numba.prange): results match serial to ~1e-12
relative, not bit-exactly.
"""
import ast
import copy
import os
from concurrent.futures import ThreadPoolExecutor

from .errors import UnsupportedError


class _ParallelInfo:
    pass


def analyze(fn_ast):
    """Match the supported prange pattern or raise UnsupportedError."""
    info = _ParallelInfo()
    body = fn_ast.body
    # locate the single top-level prange loop
    loops = [i for i, s in enumerate(body)
             if isinstance(s, ast.For) and isinstance(s.iter, ast.Call)
             and isinstance(s.iter.func, ast.Name)
             and s.iter.func.id == "prange"]
    if len(loops) != 1:
        raise UnsupportedError("exactly one top-level prange loop required")
    li = loops[0]
    loop = body[li]
    if not isinstance(loop.target, ast.Name):
        raise UnsupportedError("prange target must be a name")
    if len(loop.iter.args) not in (1, 2):
        raise UnsupportedError("prange takes 1 or 2 bounds (no step)")

    pre, post = body[:li], body[li + 1:]
    for s in pre:
        if not (isinstance(s, ast.Assign) and isinstance(s.targets[0],
                                                         ast.Name)):
            raise UnsupportedError("only simple assignments before prange")
    if len(post) != 1 or not isinstance(post[0], ast.Return) \
            or post[0].value is None:
        raise UnsupportedError("prange must be followed by a single return")

    pre_names = {s.targets[0].id for s in pre}
    # reduction detection: AugAssign(+=/-=) on a pre-loop variable
    reds, writes_pre = set(), set()
    for n in ast.walk(loop):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                and n.target.id in pre_names:
            if not isinstance(n.op, (ast.Add, ast.Sub)):
                raise UnsupportedError(
                    "only +=/-= reductions are parallelized")
            reds.add(n.target.id)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in pre_names:
            writes_pre.add(n.targets[0].id)
    if writes_pre:
        raise UnsupportedError(
            f"cross-iteration assignment to {sorted(writes_pre)}: "
            "not parallelizable")
    if len(reds) > 1:
        raise UnsupportedError("at most one reduction variable (v1)")

    info.loop_index = loop.target.id
    info.bounds = loop.iter.args
    info.pre = pre
    info.loop = loop
    info.ret_expr = post[0].value
    info.red = next(iter(reds)) if reds else None
    # the reduction's initial value must be a literal constant
    info.red_init = None
    if info.red is not None:
        for s in pre:
            if s.targets[0].id == info.red:
                if not isinstance(s.value, ast.Constant) or isinstance(
                        s.value.value, bool):
                    raise UnsupportedError(
                        "reduction initial value must be a numeric literal")
                info.red_init = s.value.value
        if info.red_init is None:
            raise UnsupportedError("reduction variable must be initialized "
                                   "before the prange loop")
    return info


def build_chunk_ast(fn_ast, info):
    """def __chunk(__lo, __hi, <orig args>): pre'; loop(range(lo,hi));
    return red (or 0)."""
    chunk = copy.deepcopy(fn_ast)
    chunk.name = f"__hj_chunk_{fn_ast.name}"
    lo = ast.arg(arg="__lo"); hi = ast.arg(arg="__hi")
    chunk.args.args = [lo, hi] + chunk.args.args
    body = chunk.body
    li = next(i for i, s in enumerate(body) if isinstance(s, ast.For)
              and isinstance(s.iter, ast.Call)
              and s.iter.func.id == "prange")
    loop = body[li]
    loop.iter = ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                         args=[ast.Name(id="__lo", ctx=ast.Load()),
                               ast.Name(id="__hi", ctx=ast.Load())],
                         keywords=[])
    pre = body[:li]
    if info.red is not None:  # neutral element per chunk
        for s in pre:
            if s.targets[0].id == info.red:
                s.value = ast.Constant(
                    value=0.0 if isinstance(info.red_init, float) else 0)
        ret = ast.Return(value=ast.Name(id=info.red, ctx=ast.Load()))
    else:
        ret = ast.Return(value=ast.Constant(value=0))
    chunk.body = pre + [loop, ret]
    return ast.fix_missing_locations(ast.Module(body=[chunk],
                                                type_ignores=[]))


def make_parallel(pyfunc, fn_ast, jit_kwargs, workers=None):
    """Return a parallel driver callable, or raise UnsupportedError."""
    from .decorator import jit, _get_func_ast  # noqa: circular-safe

    info = analyze(fn_ast)
    module = build_chunk_ast(fn_ast, info)
    src = ast.unparse(module)
    glb = dict(pyfunc.__globals__)
    exec(compile(module, f"<hanajit-parallel:{pyfunc.__name__}>", "exec"),
         glb)
    chunk_py = glb[f"__hj_chunk_{pyfunc.__name__}"]
    chunk_py.__hanajit_source__ = src  # inspect.getsource can't see exec'd
    kw = dict(jit_kwargs)
    kw.pop("nogil", None)
    chunk = jit(nogil=True, native_dispatch=False, **kw)(chunk_py)

    argnames = [a.arg for a in fn_ast.args.args]
    bound_code = [compile(ast.Expression(
        body=ast.fix_missing_locations(copy.deepcopy(b))), "<bound>", "eval")
        for b in info.bounds]
    ret_code = compile(ast.Expression(body=ast.fix_missing_locations(
        copy.deepcopy(info.ret_expr))), "<ret>", "eval")
    nworkers = workers or os.cpu_count() or 1
    pool = ThreadPoolExecutor(nworkers)

    def driver(*args):
        env = dict(zip(argnames, args))
        bounds = [eval(c, pyfunc.__globals__, env) for c in bound_code]
        lo, hi = (0, bounds[0]) if len(bounds) == 1 else tuple(bounds)
        n = max(hi - lo, 0)
        t = max(1, min(nworkers, n))
        step = (n + t - 1) // t if t else 0
        spans = [(lo + k * step, min(lo + (k + 1) * step, hi))
                 for k in range(t)] or [(lo, hi)]
        if t == 1:
            partials = [chunk(spans[0][0], spans[0][1], *args)]
        else:
            futs = [pool.submit(chunk, a, b, *args) for a, b in spans]
            partials = [f.result() for f in futs]
        if info.red is not None:
            env[info.red] = info.red_init + sum(partials)
        return eval(ret_code, pyfunc.__globals__, env)

    driver.__name__ = pyfunc.__name__
    driver.__wrapped__ = pyfunc
    driver.chunk = chunk           # the underlying nogil chunk dispatcher
    driver.parallel = True
    driver.workers = nworkers
    return driver
