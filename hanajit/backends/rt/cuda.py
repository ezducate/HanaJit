"""CUDA launch bridge over the driver API (nvcuda), via ctypes.

Loads the PTX the cuda backend emits with cuModuleLoadDataEx (the driver
JIT-compiles PTX for the installed GPU, which is why the portable sm_75
default runs on newer cards too), then cuLaunchKernel. Only the driver
library is needed — no CUDA toolkit, no cuda-python.

Array arguments are copied host->device before launch and device->host
after (all arrays are copied back; the bridge does not track writes).
"""
import ctypes
import threading

from . import RuntimeUnavailable
from ...errors import UnsupportedError

CUDA_SUCCESS = 0


def _load_driver():
    import platform
    names = (["nvcuda"] if platform.system() == "Windows"
             else ["libcuda.so.1", "libcuda.so"])
    for n in names:
        try:
            return (ctypes.WinDLL(n) if platform.system() == "Windows"
                    else ctypes.CDLL(n))
        except OSError:
            continue
    raise RuntimeUnavailable("NVIDIA driver library not found "
                             "(nvcuda.dll / libcuda.so)")


class Runtime:
    vendor = "cuda"

    def __init__(self):
        self._cu = _load_driver()
        self._lock = threading.Lock()
        self._modules = {}   # ptx-text id -> (CUmodule, {name: CUfunction})
        err = self._cu.cuInit(0)
        if err != CUDA_SUCCESS:
            raise RuntimeUnavailable(f"cuInit failed ({self._errname(err)})")
        from . import device_index
        dev = ctypes.c_int()
        if self._cu.cuDeviceGet(ctypes.byref(dev),
                                device_index("cuda")) != CUDA_SUCCESS:
            raise RuntimeUnavailable(
                f"no CUDA device at index {device_index('cuda')} "
                "(HANAJIT_CUDA_DEVICE)")
        self._dev = dev
        ctx = ctypes.c_void_p()
        err = self._cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        if err != CUDA_SUCCESS:
            raise RuntimeUnavailable(
                f"cuDevicePrimaryCtxRetain failed ({self._errname(err)})")
        self._ctx = ctx
        name = ctypes.create_string_buffer(256)
        self._cu.cuDeviceGetName(name, 255, dev)
        self.device_name = name.value.decode("utf-8", "replace")
        self._cu.cuCtxSetCurrent(self._ctx)
        stream = ctypes.c_void_p()
        if self._cu.cuStreamCreate(ctypes.byref(stream), 0) == CUDA_SUCCESS:
            self._stream = stream          # all launches go on this stream
        else:
            self._stream = None            # default stream (still correct)

    def sync(self):
        """Wait for queued (sync=False) launches to finish."""
        self._cu.cuCtxSetCurrent(self._ctx)
        if self._stream is not None:
            self._check(self._cu.cuStreamSynchronize(self._stream),
                        "cuStreamSynchronize")
        else:
            self._check(self._cu.cuCtxSynchronize(), "cuCtxSynchronize")

    def _errname(self, code):
        s = ctypes.c_char_p()
        try:
            if self._cu.cuGetErrorString(code, ctypes.byref(s)) == 0 and s.value:
                return f"{code}: {s.value.decode()}"
        except Exception:
            pass
        return str(code)

    def _check(self, err, what):
        if err != CUDA_SUCCESS:
            raise UnsupportedError(f"cuda: {what} failed "
                                   f"({self._errname(err)})")

    def _module(self, code_text, kernel_name):
        key = code_text
        with self._lock:
            entry = self._modules.get(key)
            if entry is None:
                mod = ctypes.c_void_p()
                data = code_text.encode("utf-8") + b"\0"
                self._check(self._cu.cuModuleLoadDataEx(
                    ctypes.byref(mod), data, 0, None, None),
                    "cuModuleLoadDataEx (PTX load)")
                entry = self._modules[key] = (mod, {})
            mod, funcs = entry
            fn = funcs.get(kernel_name)
            if fn is None:
                fn = ctypes.c_void_p()
                self._check(self._cu.cuModuleGetFunction(
                    ctypes.byref(fn), mod, kernel_name.encode()),
                    f"cuModuleGetFunction({kernel_name})")
                funcs[kernel_name] = fn
            return fn

    # ---- resident device buffers (DeviceArray protocol) ----
    def buf_alloc(self, arr):
        self._check(self._cu.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        dptr = ctypes.c_uint64()
        self._check(self._cu.cuMemAlloc_v2(
            ctypes.byref(dptr), ctypes.c_size_t(arr.nbytes)), "cuMemAlloc")
        return dptr

    def buf_write(self, dptr, arr):
        self._check(self._cu.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        self._check(self._cu.cuMemcpyHtoD_v2(
            dptr, ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes)), "cuMemcpyHtoD")

    def buf_read(self, dptr, arr):
        self._check(self._cu.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        self._check(self._cu.cuMemcpyDtoH_v2(
            ctypes.c_void_p(arr.ctypes.data), dptr,
            ctypes.c_size_t(arr.nbytes)), "cuMemcpyDtoH")

    def buf_free(self, dptr):
        self._cu.cuCtxSetCurrent(self._ctx)
        self._cu.cuMemFree_v2(dptr)

    def launch(self, code_text, kernel_name, grid, block, args, sync=True):
        """args: ("arr", ndarray) | ("dev", DeviceArray) | ("i64", int) |
        ("f64", float). sync=False queues on the stream and returns."""
        self._check(self._cu.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        fn = self._module(code_text, kernel_name)

        storage, dbufs = [], []   # (device_ptr, ndarray) for copy-back
        try:
            for kind, v in args:
                if kind == "dev":
                    storage.append(v._impl)   # resident: no copies
                elif kind == "arr":
                    dptr = ctypes.c_uint64()
                    nbytes = v.nbytes
                    self._check(self._cu.cuMemAlloc_v2(
                        ctypes.byref(dptr), ctypes.c_size_t(nbytes)),
                        "cuMemAlloc")
                    dbufs.append((dptr, v))
                    self._check(self._cu.cuMemcpyHtoD_v2(
                        dptr, ctypes.c_void_p(v.ctypes.data),
                        ctypes.c_size_t(nbytes)), "cuMemcpyHtoD")
                    storage.append(dptr)
                elif kind == "i64":
                    storage.append(ctypes.c_int64(v))
                elif kind == "f64":
                    storage.append(ctypes.c_double(v))
                else:
                    raise UnsupportedError(
                        f"cuda launch: unsupported argument kind {kind!r}")
            params = (ctypes.c_void_p * len(storage))(
                *[ctypes.cast(ctypes.byref(s), ctypes.c_void_p)
                  for s in storage])
            gx, gy, gz = grid
            bx, by, bz = block
            self._check(self._cu.cuLaunchKernel(
                fn, gx, gy, gz, bx, by, bz, 0, self._stream, params,
                None), "cuLaunchKernel")
            if not sync:
                return   # DeviceArray-only (enforced upstream): no
                         # copy-back, nothing transient to free
            self.sync()
            for dptr, arr in dbufs:
                self._check(self._cu.cuMemcpyDtoH_v2(
                    ctypes.c_void_p(arr.ctypes.data), dptr,
                    ctypes.c_size_t(arr.nbytes)), "cuMemcpyDtoH")
        finally:
            for dptr, _ in dbufs:
                self._cu.cuMemFree_v2(dptr)
