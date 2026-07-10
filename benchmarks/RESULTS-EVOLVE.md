# Four-way: CPython vs numba vs hanajit vs hanajit+GA

| kernel | CPython | numba | hanajit -O3 | hanajit +GA | GA gain | GA vs numba |
|---|---|---|---|---|---|---|
| fp reduction 2M * | 678.4 | 1.06 | 1.48 | 0.60 | 2.48x | 1.78x |
| np.dot 2M * | 1.5 | 1.38 | 1.44 | 1.44 | 1.00x | 0.95x |
| saxpy+sum 2M (writes) * | 488.5 | 2.18 | 2.74 | 2.59 | 1.06x | 0.84x |
| 3-pt stencil 2M * | 941.9 | 1.45 | 1.55 | 0.66 | 2.33x | 2.18x |
| branchy int 3M | 215.2 | 2.77 | 2.76 | 2.63 | 1.05x | 1.06x |
| math-heavy 1M * | 559.2 | 12.91 | 13.08 | 12.61 | 1.04x | 1.02x |
| 2-D checkerboard 1000x1000 * | 257.2 | 0.29 | 0.75 | 0.28 | 2.68x | 1.04x |
| prime count 30k | 22.6 | 1.21 | 1.20 | 1.20 | 1.00x | 1.01x |
| collatz to 30k | 199.8 | 4.04 | 4.02 | 4.00 | 1.01x | 1.01x |
| logistic 2M (chaotic: bit-exact) | 98.3 | 6.56 | 6.44 | 6.40 | 1.01x | 1.03x |
| popcount 1M | 993.6 | 7.93 | 8.75 | 7.88 | 1.11x | 1.01x |
| jacobi 20k x 100 (in-place) * | 808.1 | 0.65 | 0.48 | 0.42 | 1.15x | 1.54x |
| slice window energy 2M * | 409.6 | 0.83 | 1.49 | 0.58 | 2.57x | 1.44x |
| strided np.sum 2M * | 1.3 | 1.13 | 1.32 | 1.29 | 1.02x | 0.88x |
| fib(27) recursion | 21.0 | 1.20 | 0.71 | 0.61 | 1.16x | 1.97x |

(all times ms) **Geomeans:** GA gain over hanajit -O3 = **1.33x**; hanajit+GA vs numba = **1.19x**. `*` = fastmath allowed for BOTH numba (`njit(fastmath=True)`) and the GA; other rows bit-exact. Every result cross-verified before timing.
