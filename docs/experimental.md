# Experimental modes

Two opt-in, CPU-only, experimental features. Both are honest about their
limits — read these notes before relying on either.

## `@jit(rewrite=True)` — structural rewrites

A finite library of pattern-matched algebraic rewrites applied at the AST
level before codegen. Each rule is individually proven semantics-preserving
and fires only on its exact pattern; non-matching code is untouched.

Current rules:
- **closed_form_arithmetic_sum**: `for i in range(n): acc += a*i + b`
  becomes `acc += a*(n*(n-1)//2) + b*n` — an O(1) closed form, bit-exact in
  integer arithmetic.

**Honest scope.** This is *not* general "understand and rewrite the code" —
that is program equivalence, which is undecidable. It is a pattern library.
And our own LLVM -O3 backend *already* closes simple affine integer sums,
so for those the speed benefit over plain `@jit` is ~1x. The rewrite pass
earns its keep by (1) guaranteeing the O(1) form regardless of whether the
optimizer spots it, (2) providing a place to add rewrites LLVM does *not*
do (closed forms of more complex reductions, algebraic identities), and (3)
being portable across LLVM versions/opt levels. It is most dramatic vs
CPython (millions-x on a large closed-form sum), which is the honest
baseline to quote.

## `evolve_hyper(...)` — hyper-aggressive optimization

```python
report = f.evolve_hyper(example_args, confirmed=True, hyper_tol=1e-3)
```

A genetic search like `evolve()`, but the genome adds **unsafe** floating-
point transforms (reassociation, FMA contraction, reciprocal division,
approximate transcendental functions). Fitness emphasizes speed; a
candidate is kept if it is faster **and** matches the baseline within
`hyper_tol` on a **large random differential suite** (256 probes by
default).

### The safety contract — read this

- **The result is validated ONLY on random inputs.** It may be WRONG on
  untested inputs, including edge cases the probes never hit. This is a
  different and stronger risk than fastmath: not "slightly off" but
  "possibly incorrect on inputs you didn't sample."
- **Requires `confirmed=True`.** There is no way to enable it by accident;
  the low-level path additionally requires an internal token.
- **Never cached, never persisted.** A hyper winner lives only in the
  current process. It is never written to the disk cache, even with
  `cache=True`.
- **CPU only.** Not applied to GPU targets.
- **Never installs a regression.** If nothing beats the baseline, the
  baseline is kept.
- **Do not use in safety-critical, financial, or correctness-sensitive
  code.** A superoptimizer that does not guarantee its output has no place
  in pipelines where a wrong answer causes harm.

### Honest performance note

The benefit is hardware- and workload-dependent. On kernels where the
aggressive flags unlock SIMD reduction the safe path couldn't take, hyper
wins (measured up to ~1.1–1.3x beyond safe evolve on some reductions). On
kernels where software `exp`/`sin` have no faster approximate form on your
CPU, or where the safe fastmath path already captured the win, hyper does
nothing (1.0x) — and correctly keeps the baseline. It is a tool to *try*,
not a guaranteed speedup, and its numbers should never be quoted without
the tolerance they were validated against.
