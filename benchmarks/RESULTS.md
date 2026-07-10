# hanajit benchmark results

- Python 3.12.3, x86_64, 1 core(s)
- numba 0.66.0
- hanajit 0.8.0 (--quick)

## 1. Compute workloads

| workload | CPython | hanajit | numba | fj vs py | fj vs numba |
|---|---|---|---|---|---|
| fib(28) recursion | 42.3 ms | 1.0 ms | 1.7 ms | 41.3x | 1.66x |
| int loop 2M (mod/branch) | 138.3 ms | 3.1 ms | 3.0 ms | 44.9x | 0.98x |
| logistic map 2M (float) | 100.8 ms | 5.5 ms | 5.5 ms | 18.3x | 1.00x |
| mandelbrot point x2000 | 63.4 ms | 2.6 ms | 3.0 ms | 24.6x | 1.16x |
| collatz total to 3000 | 15.4 ms | 0.2 ms | 0.2 ms | 68.4x | 1.01x |
| prime count to 3000 | 1.2 ms | 0.0 ms | 0.0 ms | 26.3x | 1.01x |

## 2. Dispatch overhead (200k tiny calls)

| call path | 200k calls | per call |
|---|---|---|
| plain Python function | 13.1 ms | 65 ns |
| hanajit generic (native vectorcall) | 8.6 ms | 43 ns |
| hanajit .specialize() | 8.0 ms | 40 ns |
| numba dispatcher | 36.3 ms | 182 ns |

## 3. Time to first result (fresh process: import + compile + run)

| mode | time-to-first-result |
|---|---|
| python | 0.1 ms |
| hanajit | 133.5 ms |
| hanajit-cache (cold) | 95.0 ms |
| hanajit-cache (warm) | 75.4 ms |
| numba | 952.2 ms |
| numba-cache (cold) | 563.0 ms |
| numba-cache (warm) | 489.8 ms |

## 4. GIL release (main-thread iterations while a background kernel computes)

| kernel | main-thread ticks during kernel |
|---|---|
| @jit (GIL held) | 25,320 |
| @jit(nogil=True) | 212,474 |

## 5. cProfile: where dispatch time goes

### native vectorcall dispatch
```
         1 function calls in 0.000 seconds
   Ordered by: cumulative time
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

### python Dispatcher fallback
```
         300001 function calls in 0.094 seconds
   Ordered by: cumulative time
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   100000    0.068    0.000    0.094    0.000 /home/claude/hanajit/hanajit/decorator.py:166(__call__)
   100000    0.017    0.000    0.017    0.000 {method 'get' of 'dict' objects}
   100000    0.009    0.000    0.009    0.000 {tiny}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

Native dispatch shows no Python-level frames per call — the interpreter never runs between the call site and the kernel.
