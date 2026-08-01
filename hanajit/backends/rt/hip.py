"""AMD GPU launch bridge over HIP (amdhip64), via ctypes.

The amd backend emits GCN ISA as *text*; HIP loads HSA code objects
(ELF), so this bridge assembles the text with clang's AMDGPU backend
(any standard clang; ROCm ships one) before hipModuleLoadData. Both the
HIP runtime library and a clang are therefore required — each absence
is reported as its own unavailability reason.

The launch API mirrors the CUDA driver bridge: hipModuleLaunchKernel
takes the same kernelParams array of pointers-to-values.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading

from . import RuntimeUnavailable
from ...errors import UnsupportedError

HIP_OK = 0


def _load():
    import platform
    names = (["amdhip64"] if platform.system() == "Windows"
             else ["libamdhip64.so.6", "libamdhip64.so.5",
                   "libamdhip64.so"])
    for n in names:
        try:
            return (ctypes.WinDLL(n) if platform.system() == "Windows"
                    else ctypes.CDLL(n))
        except OSError:
            continue
    raise RuntimeUnavailable("HIP runtime not found (amdhip64.dll / "
                             "libamdhip64.so — ships with ROCm or the "
                             "Adrenalin driver)")


def _clang():
    c = os.environ.get("HANAJIT_HIP_CLANG") or shutil.which("clang")
    if c is None:
        raise RuntimeUnavailable(
            "clang not found — needed to assemble GCN text into an HSA "
            "code object (any standard clang; set HANAJIT_HIP_CLANG)")
    return c


class Runtime:
    vendor = "amd"
    # code_kind "asm": the dispatcher hands over native GCN assembly text

    def __init__(self):
        self._hip = _load()
        self._clang = _clang()
        self._lock = threading.Lock()
        self._modules = {}   # asm-text sha1 -> (module, {name: function})
        if self._hip.hipInit(0) != HIP_OK:
            raise RuntimeUnavailable("hipInit failed")
        from . import device_index
        want = device_index("amd")
        n = ctypes.c_int(0)
        if self._hip.hipGetDeviceCount(ctypes.byref(n)) != HIP_OK \
                or n.value == 0:
            raise RuntimeUnavailable("no HIP device")
        if want >= n.value:
            raise RuntimeUnavailable(
                f"HANAJIT_AMD_DEVICE={want} but only {n.value} HIP "
                "device(s) present")
        self._hip.hipSetDevice(want)
        name = ctypes.create_string_buffer(256)
        try:
            self._hip.hipDeviceGetName(name, 255, want)
            self.device_name = name.value.decode("utf-8", "replace")
        except Exception:
            self.device_name = "AMD GPU"

    def _check(self, err, what):
        if err != HIP_OK:
            raise UnsupportedError(f"amd (hip): {what} failed ({err})")

    def _assemble(self, asm_text):
        """GCN assembly text -> HSA code object bytes, via clang."""
        from ..gpu import resolve_arch
        arch = resolve_arch("amd")
        with tempfile.TemporaryDirectory(prefix="hanajit_hip_") as d:
            s = os.path.join(d, "k.s")
            o = os.path.join(d, "k.hsaco")
            with open(s, "w", encoding="utf-8") as f:
                f.write(asm_text)
            cmd = [self._clang, "--target=amdgcn-amd-amdhsa",
                   f"-mcpu={arch}", "-nogpulib", s, "-o", o]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode != 0:
                raise UnsupportedError(
                    "amd: clang failed to assemble GCN text: "
                    + r.stderr.decode("utf-8", "replace")[:500])
            with open(o, "rb") as f:
                return f.read()

    def _function(self, asm_text, kernel_name):
        key = hashlib.sha1(asm_text.encode()).hexdigest()
        with self._lock:
            entry = self._modules.get(key)
            if entry is None:
                hsaco = self._assemble(asm_text)
                mod = ctypes.c_void_p()
                self._check(self._hip.hipModuleLoadData(
                    ctypes.byref(mod), hsaco), "hipModuleLoadData")
                entry = self._modules[key] = (mod, {})
            mod, funcs = entry
            fn = funcs.get(kernel_name)
            if fn is None:
                fn = ctypes.c_void_p()
                self._check(self._hip.hipModuleGetFunction(
                    ctypes.byref(fn), mod, kernel_name.encode()),
                    f"hipModuleGetFunction({kernel_name})")
                funcs[kernel_name] = fn
            return fn

    # ---- resident device buffers (DeviceArray protocol) ----
    def buf_alloc(self, arr):
        dptr = ctypes.c_void_p()
        self._check(self._hip.hipMalloc(
            ctypes.byref(dptr), ctypes.c_size_t(arr.nbytes)), "hipMalloc")
        return dptr

    def buf_write(self, dptr, arr):
        self._check(self._hip.hipMemcpyHtoD(
            dptr, ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes)), "hipMemcpyHtoD")

    def buf_read(self, dptr, arr):
        self._check(self._hip.hipMemcpyDtoH(
            ctypes.c_void_p(arr.ctypes.data), dptr,
            ctypes.c_size_t(arr.nbytes)), "hipMemcpyDtoH")

    def buf_free(self, dptr):
        self._hip.hipFree(dptr)

    def sync(self):
        self._check(self._hip.hipDeviceSynchronize(),
                    "hipDeviceSynchronize")

    def launch(self, code_text, kernel_name, grid, block, args, sync=True):
        hip = self._hip
        fn = self._function(code_text, kernel_name)
        storage, dbufs = [], []
        try:
            for kind, v in args:
                if kind == "dev":
                    storage.append(ctypes.c_uint64(v._impl.value))
                elif kind == "arr":
                    dptr = ctypes.c_void_p()
                    self._check(hip.hipMalloc(
                        ctypes.byref(dptr), ctypes.c_size_t(v.nbytes)),
                        "hipMalloc")
                    dbufs.append((dptr, v))
                    self._check(hip.hipMemcpyHtoD(
                        dptr, ctypes.c_void_p(v.ctypes.data),
                        ctypes.c_size_t(v.nbytes)), "hipMemcpyHtoD")
                    storage.append(ctypes.c_uint64(dptr.value))
                elif kind == "i64":
                    storage.append(ctypes.c_int64(v))
                elif kind == "f64":
                    storage.append(ctypes.c_double(v))
                else:
                    raise UnsupportedError(
                        f"amd launch: unsupported argument kind {kind!r}")
            params = (ctypes.c_void_p * len(storage))(
                *[ctypes.cast(ctypes.byref(s), ctypes.c_void_p)
                  for s in storage])
            self._check(hip.hipModuleLaunchKernel(
                fn, grid[0], grid[1], grid[2],
                block[0], block[1], block[2], 0, None, params, None),
                "hipModuleLaunchKernel")
            if not sync:
                return   # DeviceArray-only: nothing transient below
            self._check(hip.hipDeviceSynchronize(),
                        "hipDeviceSynchronize")
            for dptr, arr in dbufs:
                self._check(hip.hipMemcpyDtoH(
                    ctypes.c_void_p(arr.ctypes.data), dptr,
                    ctypes.c_size_t(arr.nbytes)), "hipMemcpyDtoH")
        finally:
            for dptr, _ in dbufs:
                hip.hipFree(dptr)
