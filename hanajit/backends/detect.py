"""Hardware auto-detection for @jit(target="auto").

Probes for vendor runtimes by attempting to load their driver libraries
(cheap, no initialization) plus platform checks. Results are cached for
the process. Override everything with HANAJIT_TARGET=<cpu|cuda|amd|intel|metal>.
"""
import ctypes
import functools
import os
import sys

_PROBES = {
    "cuda": {"linux": ["libcuda.so.1", "libcuda.so"],
             "win32": ["nvcuda.dll"],
             "darwin": []},          # NVIDIA is dead on modern macOS
    "amd": {"linux": ["libamdhip64.so", "libhsa-runtime64.so.1"],
            "win32": ["amdhip64.dll"],
            "darwin": []},
    "intel": {"linux": ["libze_loader.so.1", "libze_loader.so"],
              "win32": ["ze_loader.dll"],
              "darwin": []},
}
# preference order per platform ("best" = most mature backend first)
_ORDER = {"darwin": ["metal"], "win32": ["cuda", "intel", "amd"],
          "linux": ["cuda", "amd", "intel"]}


def _platform():
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "win32"
    return "linux"


def _loadable(names):
    for n in names:
        try:
            ctypes.CDLL(n)
            return n
        except OSError:
            continue
    return None


@functools.lru_cache(maxsize=1)
def detect():
    """Return ordered list of (target, evidence) for this machine.
    Always ends with ("cpu", ...) — the only target hanajit executes on
    directly today; GPU entries indicate device-code emission targets."""
    plat = _platform()
    found = []
    forced = os.environ.get("HANAJIT_TARGET")
    if forced:
        return [(forced, "forced via HANAJIT_TARGET")]
    if plat == "darwin":
        found.append(("metal", "macOS (Metal is always present)"))
    for vendor in _ORDER[plat]:
        if vendor == "metal":
            continue
        lib = _loadable(_PROBES[vendor][plat])
        if lib:
            found.append((vendor, f"driver library {lib}"))
    found.append(("cpu", "always available"))
    return found


def best_gpu():
    """Best detected GPU target, or None if the machine has no GPU runtime."""
    for target, _ in detect():
        if target != "cpu":
            return target
    return None
