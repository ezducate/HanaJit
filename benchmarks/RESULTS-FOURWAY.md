# Four-way benchmark

| workload | CPython | numba | hanajit | hanajit+GA | GA gain | vs numba (GA) |
|---|---|---|---|---|---|---|
| fib(30) recursion | 77.9 ms | 5.0 ms | 2.7 ms | 2.1 ms | 1.28x | 2.38x |
| int loop 20M (mod/branch) | 1,076.5 ms | 17.2 ms | 15.9 ms | 15.9 ms | 1.00x* | 1.08x |
| logistic map 20M (serial fp) | 872.5 ms | 44.8 ms (numba fastmath: 44.0 ms) | 44.6 ms | 44.8 ms | 1.00x* | 1.00x |
| fp reduction 2M | 544.3 ms | 1.6 ms (numba fastmath: 0.7 ms) | 1.6 ms | 0.7 ms | 2.13x | 2.13x |
| fused numpy expr 2M (5 ops) | 58.6 ms | 49.6 ms (numba fastmath: 30.7 ms) | 14.4 ms | 14.3 ms | 1.01x | 3.47x |
| mandelbrot x4000 | 91.7 ms | 3.2 ms (numba fastmath: 3.3 ms) | 3.3 ms | 3.1 ms | 1.06x | 1.03x |

`*` = GA found no improvement; baseline kept (never a regression).
fastmath allowed for GA on: logistic, fp reduction, fused expr, mandelbrot — numba fastmath datapoints shown for fairness.
