# NumPy / SciPy acceleration

| workload | numpy/scipy | hanajit | numba | vs numpy | vs numba |
|---|---|---|---|---|---|
| fused elementwise+reduce, 2M (4 numpy temporaries) | 12.3 ms | 1.7 ms | 1.6 ms | 7.3x | 0.97x |
| dot product 2M (numpy = BLAS: expected numpy win) | 1.25 ms | 1.57 ms | 1.61 ms | 0.80x | 1.03x |
| Jacobi smoothing, 20k x 200 sweeps (in-place, seq. dependency) | 7.4 ms | 1.0 ms | 1.4 ms | 7.4x | 1.44x |
| scipy.quad gaussian (LowLevelCallable vs Python callback) | 0.02 ms | 0.01 ms | 0.01 ms | 4.0x | 1.04x |
