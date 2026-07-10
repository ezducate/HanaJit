# hanajit micro-benchmarks (52 workloads)

- Python 3.12.3, x86_64, 1 core(s); numba 0.66.0; hanajit 0.5.1


## int

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| sum of range | 118.9 ms | 0.00 ms | 0.00 ms | 724,723.8x | 1.38x |
| alternating sum | 159.9 ms | 0.30 ms | 0.30 ms | 532.2x | 1.00x |
| gcd chain | 5.7 ms | 0.29 ms | 0.35 ms | 19.3x | 1.18x |
| collatz total | 211.2 ms | 3.10 ms | 3.10 ms | 68.1x | 1.00x |
| factorial mod | 243.5 ms | 11.06 ms | 15.01 ms | 22.0x | 1.36x |
| modular exponentiation | 21.2 ms | 0.45 ms | 0.46 ms | 46.8x | 1.02x |
| digit sum sweep | 102.2 ms | 1.65 ms | 1.60 ms | 61.8x | 0.97x |
| pell recurrence mod | 265.8 ms | 15.04 ms | 14.96 ms | 17.7x | 0.99x |
| fibonacci iterative mod | 204.5 ms | 13.45 ms | 13.27 ms | 15.2x | 0.99x |
| fizzbuzz count | 170.2 ms | 4.63 ms | 4.68 ms | 36.7x | 1.01x |

## float

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| logistic map | 49.1 ms | 2.72 ms | 2.71 ms | 18.1x | 1.00x |
| newton sqrt sweep | 25.1 ms | 0.65 ms | 0.81 ms | 38.4x | 1.24x |
| leibniz pi | 62.5 ms | 1.22 ms | 1.20 ms | 51.2x | 0.98x |
| harmonic series | 39.1 ms | 1.20 ms | 1.20 ms | 32.5x | 1.00x |
| exp taylor sweep | 59.9 ms | 1.45 ms | 1.44 ms | 41.2x | 0.99x |
| sin taylor sweep | 49.3 ms | 0.66 ms | 0.64 ms | 74.7x | 0.97x |
| horner polynomial | 116.2 ms | 2.56 ms | 2.56 ms | 45.4x | 1.00x |
| lcg mean | 201.3 ms | 1.39 ms | 1.37 ms | 144.3x | 0.98x |
| euler pendulum | 78.1 ms | 14.02 ms | 14.12 ms | 5.6x | 1.01x |
| verlet spring | 59.9 ms | 3.70 ms | 3.73 ms | 16.2x | 1.01x |
| softsign sum | 100.5 ms | 1.84 ms | 2.11 ms | 54.7x | 1.15x |
| mandelbrot escape batch | 63.0 ms | 2.43 ms | 2.42 ms | 26.0x | 1.00x |

## control

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| prime count | 27.1 ms | 1.04 ms | 1.04 ms | 25.9x | 0.99x |
| perfect numbers | 36.1 ms | 1.18 ms | 1.18 ms | 30.5x | 1.00x |
| happy numbers | 182.9 ms | 7.52 ms | 7.37 ms | 24.3x | 0.98x |
| palindromic numbers | 53.2 ms | 0.92 ms | 0.88 ms | 58.1x | 0.96x |
| generalized syracuse | 110.7 ms | 4.15 ms | 4.34 ms | 26.7x | 1.05x |
| binary gcd | 46.6 ms | 2.33 ms | 2.31 ms | 20.0x | 0.99x |
| zeller weekday sum | 23.5 ms | 0.70 ms | 0.65 ms | 33.6x | 0.92x |
| loop break/continue mix | 277.2 ms | 7.97 ms | 6.45 ms | 34.8x | 0.81x |

## recursion

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| fibonacci | 26.8 ms | 0.64 ms | 1.07 ms | 42.2x | 1.68x |
| ackermann(3,6) | 19.2 ms | 0.70 ms | 1.09 ms | 27.6x | 1.56x |
| binary tree fold | 51.6 ms | 1.11 ms | 1.45 ms | 46.5x | 1.31x |
| takeuchi tak(18,12,6) | 2.6 ms | 0.09 ms | 0.12 ms | 30.0x | 1.35x |
| cross-function call sweep (known hanajit gap) | 0.5 ms | 0.18 ms | 0.03 ms | 2.8x | 0.14x |

## bits

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| lcg xor cascade | 597.1 ms | 3.91 ms | 3.99 ms | 152.9x | 1.02x |
| popcount sweep | 289.3 ms | 2.33 ms | 2.28 ms | 124.0x | 0.98x |
| reverse 16-bit | 3,615.9 ms | 0.81 ms | 0.82 ms | 4,445.0x | 1.00x |
| parity sum | 3,678.4 ms | 29.80 ms | 28.83 ms | 123.4x | 0.97x |
| gray code sum | 241.9 ms | 0.33 ms | 0.33 ms | 727.7x | 0.99x |

## pointer

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| saxpy | 81.0 ms | 0.35 ms | 0.36 ms | 233.6x | 1.03x |
| dot product | 41.7 ms | 0.25 ms | 0.25 ms | 165.7x | 0.98x |
| abs sum | 27.6 ms | 0.24 ms | 0.24 ms | 114.2x | 1.00x |
| max element | 18.7 ms | 0.28 ms | 0.24 ms | 66.9x | 0.86x |
| count above threshold | 23.2 ms | 0.07 ms | 0.06 ms | 342.9x | 0.93x |
| prefix sum (in place) | 47.6 ms | 0.24 ms | 0.24 ms | 195.0x | 0.98x |
| 3-point stencil | 132.1 ms | 0.37 ms | 0.37 ms | 355.4x | 0.98x |
| polynomial eval array | 78.3 ms | 0.25 ms | 0.24 ms | 315.5x | 0.97x |

## dispatch

| workload | CPython | hanajit | numba | vs py | vs numba |
|---|---|---|---|---|---|
| 200k tiny calls (int,int) | 15.3 ms | 10.48 ms | 44.34 ms | 1.5x | 4.23x |
| 100k medium calls | 142.2 ms | 6.04 ms | 21.85 ms | 23.5x | 3.62x |
| 100k polymorphic calls (int/float alternating) | 8.3 ms | 4.74 ms | 21.16 ms | 1.7x | 4.47x |
| 200k .specialize() calls | 15.1 ms | 9.10 ms | 45.80 ms | 1.7x | 5.04x |

## Summary (geometric means)

| category | workloads | hanajit vs CPython | hanajit vs numba |
|---|---|---|---|
| int | 10 | 112.3x | 1.08x |
| float | 12 | 34.6x | 1.02x |
| control | 8 | 30.2x | 0.96x |
| recursion | 5 | 21.5x | 0.92x |
| bits | 5 | 376.5x | 0.99x |
| pointer | 8 | 196.4x | 0.97x |
| dispatch | 4 | 3.2x | 4.31x |
| **overall** | **52** | **55.5x** | **1.12x** |
