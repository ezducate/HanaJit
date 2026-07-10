"""Validate hanajit's Metal backend on a Mac (Apple Silicon or Intel).

Run on macOS:  python examples/metal_check.py
Generates MSL from a sample kernel and compiles it with Apple's real
Metal compiler (xcrun metal). Requires Xcode command-line tools.
"""
import subprocess
import sys
import tempfile
import os
import warnings

warnings.filterwarnings("ignore")
from hanajit import jit  # noqa: E402


@jit(target="metal", signature="f64*, f64*, f64, i64")
def saxpy(x, y, a, n):
    i = block_id() * block_dim() + thread_id()
    if i < n:
        y[i] = a * x[i] + y[i]
    return 0


_, msl, _ = saxpy.inspect_gpu()
print(msl)

if sys.platform != "darwin":
    print("[not macOS: MSL generated above; compile check skipped]")
    sys.exit(0)

with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "kernel.metal")
    air = os.path.join(td, "kernel.air")
    with open(src, "w") as f:
        f.write(msl)
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", src,
                        "-o", air], capture_output=True, text=True)
    if r.returncode == 0:
        lib = os.path.join(td, "kernel.metallib")
        subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air,
                        "-o", lib], check=True)
        print(f"OK: compiled to AIR + metallib "
              f"({os.path.getsize(lib)} bytes) — Metal backend valid "
              f"on this Mac")
    else:
        print("Metal compilation FAILED:\n" + r.stderr)
        sys.exit(1)
