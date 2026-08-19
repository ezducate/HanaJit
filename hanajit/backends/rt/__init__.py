"""GPU runtime bridges: execute emitted device code from Python.

Each bridge drives the vendor's driver/runtime API directly through
ctypes — no SDK and no build step, only the library the GPU driver
already installs (nvcuda / amdhip64 / ze_loader / vulkan-1).

`get_runtime(vendor)` returns a lazily-created singleton runtime, or
None when the vendor cannot execute on this machine; `why_unavailable`
then holds the human-readable reason (no driver library, no device,
missing toolchain, ...). A missing runtime is never an import error —
emit-only operation must keep working on machines with no GPU.
"""
from ...errors import UnsupportedError


class RuntimeUnavailable(Exception):
    """Raised by a bridge constructor when this machine cannot launch
    kernels for its vendor (no driver, no device, missing tool)."""


_MODULES = {
    "cuda": ".cuda",
    "amd": ".hip",
    "intel": ".level_zero",
    "vulkan": ".vulkan",
    "metal": ".metal",
}

_cache = {}      # vendor -> runtime instance or None
_reasons = {}    # vendor -> reason string when unavailable


def device_index(vendor):
    """User-selected device ordinal for a vendor (default 0), from
    HANAJIT_CUDA_DEVICE / HANAJIT_INTEL_DEVICE / HANAJIT_VULKAN_DEVICE /
    HANAJIT_AMD_DEVICE."""
    import os
    v = os.environ.get(f"HANAJIT_{vendor.upper()}_DEVICE", "0")
    try:
        return max(0, int(v))
    except ValueError:
        return 0


def reset():
    """Forget cached runtimes (used after changing HANAJIT_*_DEVICE)."""
    _cache.clear()
    _reasons.clear()


def get_runtime(vendor):
    """Runtime bridge for `vendor`, or None if unavailable here."""
    if vendor in _cache:
        return _cache[vendor]
    if vendor not in _MODULES:
        _cache[vendor] = None
        _reasons[vendor] = f"no runtime bridge for target {vendor!r}"
        return None
    try:
        import importlib
        mod = importlib.import_module(_MODULES[vendor], __name__)
        rt = mod.Runtime()
    except RuntimeUnavailable as e:
        _cache[vendor] = None
        _reasons[vendor] = str(e)
        return None
    except Exception as e:  # defensive: a broken driver must not crash us
        _cache[vendor] = None
        _reasons[vendor] = f"{type(e).__name__}: {e}"
        return None
    _cache[vendor] = rt
    _reasons[vendor] = None
    return rt


def why_unavailable(vendor):
    get_runtime(vendor)
    return _reasons.get(vendor)


class DeviceArray:
    """A device-resident buffer bound to one runtime bridge.

    Created with `f.to_device(arr)`; pass it to `f.launch()` in place of
    the numpy array to skip the per-launch host<->device copies. Data
    stays on the device across launches; read it back explicitly with
    `to_host()`. Freed on `free()` or garbage collection.
    """

    def __init__(self, runtime, impl, shape, dtype_str, nbytes):
        self.runtime = runtime
        self._impl = impl
        self.shape = shape
        self.dtype = dtype_str        # "float64" / "int64"
        self.nbytes = nbytes

    def __len__(self):
        return self.shape[0]

    def copy_from_host(self, arr):
        """Overwrite device contents from a matching numpy array."""
        self._check_alive()
        if (arr.shape != self.shape or arr.nbytes != self.nbytes
                or str(arr.dtype) != self.dtype):
            raise UnsupportedError(
                "copy_from_host: array shape/dtype mismatch")
        self.runtime.sync()   # order after any queued async launches
        self.runtime.buf_write(self._impl, arr)
        return self

    def to_host(self, out=None):
        """Copy device contents into `out` (or a new numpy array).
        Synchronizes first, so queued sync=False launches are visible."""
        self._check_alive()
        import numpy as np
        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
        elif (out.shape != self.shape or out.nbytes != self.nbytes
              or str(out.dtype) != self.dtype):
            raise UnsupportedError("to_host: array shape/dtype mismatch")
        self.runtime.sync()
        self.runtime.buf_read(self._impl, out)
        return out

    def free(self):
        if self._impl is not None:
            impl, self._impl = self._impl, None
            try:
                self.runtime.buf_free(impl)
            except Exception:
                pass  # interpreter teardown / device already gone

    def _check_alive(self):
        if self._impl is None:
            raise UnsupportedError("DeviceArray already freed")

    def __del__(self):
        self.free()


def to_device(runtime, arr):
    """Allocate a DeviceArray on `runtime` and upload `arr`."""
    impl = runtime.buf_alloc(arr)
    try:
        runtime.buf_write(impl, arr)
    except Exception:
        runtime.buf_free(impl)
        raise
    return DeviceArray(runtime, impl, arr.shape, str(arr.dtype),
                       arr.nbytes)


def normalize_dims(v, default):
    """Accept int or 1-3 tuple; return a 3-tuple padded with 1s."""
    if v is None:
        v = default
    if isinstance(v, int):
        v = (v,)
    t = tuple(int(x) for x in v) + (1, 1, 1)
    if len(t) > 6 or any(x < 1 for x in t[:3]):
        raise UnsupportedError(f"bad launch dimensions: {v!r}")
    return t[:3]
