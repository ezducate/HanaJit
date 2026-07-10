"""Evolutionary post-optimization: squeeze the already-optimized kernel.

`f.evolve(*example_args)` runs a genetic algorithm over the space of
*semantics-preserving* compilation strategies for an already-compiled
kernel: optimization pipeline speed level, loop vectorization, SLP
vectorization, loop unrolling, interleaving, and target-machine codegen
level — and, only if explicitly allowed, fastmath IR regeneration.

Why this is "logically equivalent by construction": every gene toggles a
transformation LLVM guarantees preserves semantics. We do NOT mutate
instructions (STOKE-style stochastic superoptimization needs an SMT prover
to re-establish equivalence per candidate). The one semantics-affecting
gene, fastmath, is opt-in via allow_fastmath=True and changes only FP
rounding/association, never logic. On top of the by-construction argument,
every candidate is differentially validated against the baseline kernel on
the example inputs plus generated probes before it may win.

Fitness = measured wall time (best-of-reps) on the example arguments.
The winner is installed into the live dispatcher only if it beats the
baseline; otherwise the baseline stays and the report says so honestly.
"""
import ctypes
import random
import re
import time

from llvmlite import binding as llvm

from .errors import UnsupportedError
from .typeinfer import ARRAY_ELEM, F64
from .backends import cpu as cpu_backend

GENES = {
    "speed": (1, 2, 3),        # pass-pipeline speed level
    "tm": (1, 2, 3),           # target-machine codegen opt level
    "lv": (0, 1),              # loop vectorization
    "slp": (0, 1),             # SLP vectorization
    "unroll": (0, 1),          # loop unrolling pass
    "inter": (0, 1),           # loop interleaving
    "uc": (0, 2, 4, 8),        # explicit llvm.loop.unroll.count (0 = off)
    "vw": (0, 4, 8),           # explicit llvm.loop.vectorize.width (0 = off)
}
BASELINE = {"speed": 3, "tm": 3, "lv": 1, "slp": 1, "unroll": 1, "inter": 1,
            "uc": 0, "vw": 0, "fm": 0}

_LOOP_BR = re.compile(r'(br label %"(?:for|while|red)\.cond[^"]*")')


def _apply_loop_md(ir_text, uc, vw):
    """Attach !llvm.loop unroll-count / vectorize-width metadata to every
    loop back-edge. Both hints are semantics-preserving directives to
    LLVM's own transforms."""
    if not uc and not vw:
        return ir_text
    refs, defs, k = ["!hj0"], [], 1
    if uc:
        defs.append(f'!hj{k} = !{{!"llvm.loop.unroll.count", i32 {uc}}}')
        refs.append(f"!hj{k}"); k += 1
    if vw:
        defs.append(f'!hj{k} = !{{!"llvm.loop.vectorize.width", i32 {vw}}}')
        refs.append(f"!hj{k}"); k += 1
    header = f'!hj0 = distinct !{{{", ".join(refs)}}}'
    body = _LOOP_BR.sub(r"\1, !llvm.loop !hj0", ir_text)
    return body + "\n" + header + "\n" + "\n".join(defs) + "\n"


def _random_genome(rng, allow_fm):
    g = {k: rng.choice(v) for k, v in GENES.items()}
    g["fm"] = rng.choice((0, 1)) if allow_fm else 0
    return g


def _mutate(g, rng, allow_fm):
    g = dict(g)
    k = rng.choice(list(GENES) + (["fm"] if allow_fm else []))
    opts = GENES.get(k, (0, 1))
    g[k] = rng.choice([v for v in opts if v != g[k]] or opts)
    return g


def _crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in a}


_HYPER_ATTRS = {
    "reassoc": '"reassoc-fp-math"="true"',
    "contract": '"contract-fp-math"="true"',
    "nsz": '"no-signed-zeros-fp-math"="true"',
    "arcp": '"reciprocal-fp-math"="true"',
    "afn": '"approx-func-fp-math"="true" "unsafe-fp-math"="true"',
}


def _apply_hyper_attrs(ir_text, genome):
    """Attach aggressive fp function attributes for hyper-aggressive mode.
    Structured risk: these change fp rounding/association/precision, never
    control flow — so random-probe validation is meaningful evidence."""
    attrs = [v for k, v in _HYPER_ATTRS.items() if genome.get(k)]
    if not attrs:
        return ir_text
    import re as _re
    # tag every define with attribute group #hj_h and append the group
    ir_text = _re.sub(r"(define [^{]*?)\{",
                      r"\1#hjh {", ir_text, count=0)
    grp = 'attributes #hjh = { ' + " ".join(attrs) + ' }'
    return ir_text + "\n" + grp + "\n"


def _compile_candidate(ir_text, func_name, sig, ret_type, genome):
    """Compile one genome; return a plain ctypes callable (kept alive by
    the engines list in the cpu backend)."""
    tm = cpu_backend._host_target_machine(genome["tm"])
    mod = llvm.parse_assembly(ir_text)  # loop-md applied upstream
    mod.verify()
    cpu_backend._optimize(mod, tm, genome["speed"], tuning={
        "loop_vectorization": bool(genome["lv"]),
        "slp_vectorization": bool(genome["slp"]),
        "loop_unrolling": bool(genome["unroll"]),
        "loop_interleaving": bool(genome["inter"])})
    engine = llvm.create_mcjit_compiler(mod, tm)
    engine.finalize_object()
    cpu_backend._engines.append(engine)
    addr = engine.get_function_address(func_name)
    return cpu_backend._proto(ret_type, sig)(addr), addr


def _time_call(fn, args, reps):
    best = float("inf")
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return r, best


# Hyper-aggressive transforms: each is a *structured* risk (not instruction
# mutation), so "passed N random probes" is meaningful evidence — but none
# is guaranteed for all inputs. Gated behind hyper_aggressive=True + a large
# random differential suite. Results are NEVER cached or persisted.
HYPER_GENES = {
    "reassoc": (0, 1),     # reassociate fp (changes rounding/overflow order)
    "contract": (0, 1),    # fuse mul+add to FMA
    "nsz": (0, 1),         # assume no signed zeros
    "arcp": (0, 1),        # allow reciprocal (a/b -> a * (1/b))
    "afn": (0, 1),         # approximate transcendental functions
}


def evolve(dispatcher, example_args, generations=6, population=10,
           reps=5, seed=0, allow_fastmath=False, verbose=False,
           hyper_aggressive=False, hyper_probes=256, hyper_tol=1e-4,
           _hyper_confirmed=False):
    if hyper_aggressive and not _hyper_confirmed:
        raise UnsupportedError(
            "hyper_aggressive mode requires _hyper_confirmed=True. This mode "
            "applies unsafe optimizations validated ONLY on random probes; "
            "the result may be WRONG on untested inputs and is never cached. "
            "Use f.evolve_hyper(...) which shows the disclaimer.")
    from .decorator import (_abstract_types, _is_ndarray, _hybrid_key,
                            _make_array_caller)
    rng = random.Random(seed)

    sig = (dispatcher.signature if dispatcher.signature is not None
           else _abstract_types(example_args))
    has_arrays = any(t in ARRAY_ELEM for t in sig)

    # ensure a baseline exists and grab its IR (pre-optimization form)
    if sig not in dispatcher.cache:
        dispatcher._slow_call(*example_args)
    if sig not in dispatcher.modules:
        raise UnsupportedError(
            "evolve() needs the kernel IR; disk-cache hits skip codegen — "
            "call once with cache=False first")
    ir_plain = str(dispatcher.modules[sig])
    fn_name = dispatcher.pyfunc.__name__
    ir_fast = dispatcher._build_ir(sig, fastmath=True) if \
        (allow_fastmath or hyper_aggressive) else None
    ret_type = dispatcher._sig_ret[sig]
    # hyper mode: fastmath IR is the base, and we also grow the genome with
    # aggressive fp genes applied as function attributes on the IR.
    if hyper_aggressive:
        allow_fastmath = True  # fastmath IR is the substrate

    def wrap(callable_):
        return (_make_array_caller(callable_, sig) if has_arrays
                else callable_)

    def fresh_args():
        return tuple(a.copy() if _is_ndarray(a) else a
                     for a in example_args)

    # differential probes: the example plus perturbed variants. Hyper mode
    # uses a much larger random suite (the ONLY correctness evidence there).
    import numpy as _np
    probes = [fresh_args()]
    n_extra = (hyper_probes if hyper_aggressive else 3)
    for _ in range(n_extra):
        newp = []
        for a in example_args:
            if _is_ndarray(a):
                # fresh random array, same shape/dtype, spanning a range
                r = _np.random.default_rng(rng.randint(0, 2**31)).uniform(
                    -10, 10, a.shape).astype(a.dtype)
                newp.append(r)
            elif isinstance(a, int) and a > 8:
                newp.append(a + rng.randint(-min(a - 1, 50), 50))
            elif isinstance(a, float):
                newp.append(a * rng.uniform(0.1, 3.0) - rng.uniform(0, 2))
            else:
                newp.append(a)
        probes.append(tuple(newp))

    baseline_fn = wrap(dispatcher.cache[sig])
    expected = [baseline_fn(*tuple(x.copy() if _is_ndarray(x) else x
                                   for x in p)) for p in probes]
    _, t_base = _time_call(baseline_fn, fresh_args(), reps)
    tol = (hyper_tol if hyper_aggressive
           else (1e-9 if allow_fastmath else 0.0))

    def fitness(genome):
        ir_text = ir_fast if genome.get("fm") else ir_plain
        ir_text = _apply_loop_md(ir_text, genome.get("uc", 0),
                                 genome.get("vw", 0))
        ir_text = _apply_hyper_attrs(ir_text, genome)
        try:
            cand, addr = _compile_candidate(ir_text, fn_name, sig,
                                            ret_type, genome)
        except Exception:
            return None
        w = wrap(cand)
        for p, exp in zip(probes, expected):  # equivalence gate
            got = w(*tuple(x.copy() if _is_ndarray(x) else x for x in p))
            if tol == 0.0:
                if got != exp:
                    return None
            elif abs(got - exp) > tol * max(1.0, abs(exp)):
                return None
        _, t = _time_call(w, fresh_args(), reps)
        return (t, genome)

    memo = {}
    evals = [0]

    def fitness_memo(g):
        key = tuple(sorted(g.items()))
        if key in memo:
            return memo[key]
        evals[0] += 1
        memo[key] = fitness(g)
        return memo[key]

    def rand_genome():
        g = _random_genome(rng, allow_fastmath)
        if hyper_aggressive:
            for k, opts in HYPER_GENES.items():
                g[k] = rng.choice(opts)
        return g
    base = dict(BASELINE)
    if hyper_aggressive:
        for k in HYPER_GENES:
            base[k] = 0
    pop = [base] + [rand_genome() for _ in range(population - 1)]
    scored = [f for f in (fitness_memo(g) for g in pop) if f]
    stagnant, prev_best = 0, float("inf")
    for gen in range(generations):
        scored.sort(key=lambda x: x[0])
        if verbose:
            print(f"[evolve] gen {gen}: best {scored[0][0]*1e3:.3f} ms "
                  f"{scored[0][1]}")
        if scored[0][0] >= prev_best * 0.995:
            stagnant += 1
            if stagnant >= 2:
                break                    # early stop: converged
        else:
            stagnant = 0
        prev_best = scored[0][0]
        elite = [g for _, g in scored[:max(2, population // 4)]]
        children = []
        while len(children) < population - len(elite):
            a, b = rng.sample(elite, 2) if len(elite) > 1 else (elite[0],) * 2
            c = _crossover(a, b, rng)
            if rng.random() < 0.5:
                c = _mutate(c, rng, allow_fastmath)
                if hyper_aggressive:
                    k = rng.choice(list(HYPER_GENES))
                    c[k] = rng.choice(HYPER_GENES[k])
            children.append(c)
        scored = ([(t, g) for t, g in scored[:len(elite)]]
                  + [f for f in (fitness_memo(g) for g in children) if f])

    # re-measure finalists to de-noise the podium before declaring a winner
    scored.sort(key=lambda x: x[0])
    finalists = scored[:3]
    rescored = []
    for _, g in finalists:
        f = fitness(g)
        if f:
            rescored.append(f)
    rescored.sort(key=lambda x: x[0])
    t_best, g_best = rescored[0] if rescored else scored[0]
    improved = t_best < t_base * 0.995
    if improved:
        win_ir = _apply_loop_md(ir_fast if g_best.get("fm") else ir_plain,
                                g_best.get("uc", 0), g_best.get("vw", 0))
        dispatcher._install_evolved(sig, g_best, win_ir)
    return {"baseline_ms": t_base * 1e3, "best_ms": t_best * 1e3,
            "speedup": t_base / t_best if improved else 1.0,
            "genome": g_best if improved else BASELINE,
            "installed": improved, "evaluations": evals[0],
            "persisted": (improved and dispatcher.disk_cache
                          and not hyper_aggressive),
            "hyper_aggressive": hyper_aggressive,
            "probes_validated": len(probes)}
