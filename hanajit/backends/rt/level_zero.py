"""Intel GPU launch bridge over Level Zero (ze_loader), via ctypes.

Kernels are generated as OpenCL-flavor SPIR-V by backends/spirv.py
(LLVM's SPIR-V backend has no binary writer in llvmlite) and loaded
with zeModuleCreate(ZE_MODULE_FORMAT_IL_SPIRV). Execution uses one
synchronous immediate command list — every append completes before it
returns, so no events or fences are needed for this bridge's
copy-launch-copy pattern. Enum values are from ze_api.h (v1.x ABI).
"""
import ctypes
import threading

from . import RuntimeUnavailable
from ...errors import UnsupportedError

ZE_OK = 0
ST_DEVICE_PROPERTIES = 0x3
ST_CONTEXT_DESC = 0xD
ST_COMMAND_QUEUE_DESC = 0xE
ST_DEVICE_MEM_ALLOC_DESC = 0x15
ST_MODULE_DESC = 0x1B
ST_KERNEL_DESC = 0x1D
DEVICE_TYPE_GPU = 1
QUEUE_MODE_SYNCHRONOUS = 1
MODULE_FORMAT_IL_SPIRV = 0


class _DeviceProperties(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("type", ctypes.c_int), ("vendorId", ctypes.c_uint32),
                ("deviceId", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("subdeviceId", ctypes.c_uint32),
                ("coreClockRate", ctypes.c_uint32),
                ("maxMemAllocSize", ctypes.c_uint64),
                ("maxHardwareContexts", ctypes.c_uint32),
                ("maxCommandQueuePriority", ctypes.c_uint32),
                ("numThreadsPerEU", ctypes.c_uint32),
                ("physicalEUSimdWidth", ctypes.c_uint32),
                ("numEUsPerSubslice", ctypes.c_uint32),
                ("numSubslicesPerSlice", ctypes.c_uint32),
                ("numSlices", ctypes.c_uint32),
                ("timerResolution", ctypes.c_uint64),
                ("timestampValidBits", ctypes.c_uint32),
                ("kernelTimestampValidBits", ctypes.c_uint32),
                ("uuid", ctypes.c_uint8 * 16),
                ("name", ctypes.c_char * 256)]


class _ContextDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32)]


class _QueueDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("ordinal", ctypes.c_uint32), ("index", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("mode", ctypes.c_int),
                ("priority", ctypes.c_int)]


class _ModuleDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("format", ctypes.c_int),
                ("inputSize", ctypes.c_size_t),
                ("pInputModule", ctypes.c_char_p),
                ("pBuildFlags", ctypes.c_char_p),
                ("pConstants", ctypes.c_void_p)]


class _KernelDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("pKernelName", ctypes.c_char_p)]


class _MemAllocDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32), ("ordinal", ctypes.c_uint32)]


class _GroupCount(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32),
                ("z", ctypes.c_uint32)]


def _load():
    import platform
    names = (["ze_loader"] if platform.system() == "Windows"
             else ["libze_loader.so.1", "libze_loader.so"])
    for n in names:
        try:
            return (ctypes.WinDLL(n) if platform.system() == "Windows"
                    else ctypes.CDLL(n))
        except OSError:
            continue
    raise RuntimeUnavailable("Level Zero loader not found "
                             "(ze_loader.dll / libze_loader.so)")


class Runtime:
    vendor = "intel"
    code_kind = "ast"   # dispatcher hands over the typed AST
    # sync() defined below; other bridges define theirs likewise

    def __init__(self):
        ze = self._ze = _load()
        self._lock = threading.Lock()
        self._modules = {}   # spirv bytes -> (module, {name: kernel})
        if ze.zeInit(0) != ZE_OK:
            raise RuntimeUnavailable("zeInit failed")
        n = ctypes.c_uint32(0)
        if ze.zeDriverGet(ctypes.byref(n), None) != ZE_OK or n.value == 0:
            raise RuntimeUnavailable("no Level Zero driver")
        drivers = (ctypes.c_void_p * n.value)()
        ze.zeDriverGet(ctypes.byref(n), drivers)
        from . import device_index
        want = device_index("intel")
        gpus = []   # (driver, device, name) in enumeration order
        for d in drivers:
            d = ctypes.c_void_p(d)  # array iteration yields raw ints
            dn = ctypes.c_uint32(0)
            if ze.zeDeviceGet(d, ctypes.byref(dn), None) != ZE_OK \
                    or dn.value == 0:
                continue
            devs = (ctypes.c_void_p * dn.value)()
            ze.zeDeviceGet(d, ctypes.byref(dn), devs)
            for dev in devs:
                dev = ctypes.c_void_p(dev)
                props = _DeviceProperties()
                props.stype = ST_DEVICE_PROPERTIES
                if ze.zeDeviceGetProperties(dev, ctypes.byref(props)) \
                        == ZE_OK and props.type == DEVICE_TYPE_GPU:
                    gpus.append((d, dev, props.name.decode(
                        "utf-8", "replace")))
        if not gpus:
            raise RuntimeUnavailable("no Level Zero GPU device")
        if want >= len(gpus):
            raise RuntimeUnavailable(
                f"HANAJIT_INTEL_DEVICE={want} but only {len(gpus)} "
                "Level Zero GPU(s) present")
        self._drv, self._dev, self.device_name = gpus[want]

        cdesc = _ContextDesc(stype=ST_CONTEXT_DESC)
        ctx = ctypes.c_void_p()
        self._check(ze.zeContextCreate(self._drv, ctypes.byref(cdesc),
                                       ctypes.byref(ctx)),
                    "zeContextCreate", RuntimeUnavailable)
        self._ctx = ctx
        qdesc = _QueueDesc(stype=ST_COMMAND_QUEUE_DESC,
                           mode=QUEUE_MODE_SYNCHRONOUS)
        cl = ctypes.c_void_p()
        self._check(ze.zeCommandListCreateImmediate(
            ctx, self._dev, ctypes.byref(qdesc), ctypes.byref(cl)),
            "zeCommandListCreateImmediate", RuntimeUnavailable)
        self._cl = cl
        self._acl = None   # async immediate list, created on first use

    def _async_list(self):
        if self._acl is None:
            qdesc = _QueueDesc(stype=ST_COMMAND_QUEUE_DESC, mode=0)
            acl = ctypes.c_void_p()
            if self._ze.zeCommandListCreateImmediate(
                    self._ctx, self._dev, ctypes.byref(qdesc),
                    ctypes.byref(acl)) == ZE_OK:
                self._acl = acl
            else:
                self._acl = self._cl   # degrade: async becomes sync
        return self._acl

    def sync(self):
        """Wait for queued (sync=False) launches to finish."""
        if self._acl is not None and self._acl is not self._cl:
            try:
                self._check(self._ze.zeCommandListHostSynchronize(
                    self._acl, ctypes.c_uint64(2**64 - 1)),
                    "zeCommandListHostSynchronize")
            except AttributeError:
                pass   # pre-1.6 loader: async list was never used then

    def _check(self, err, what, exc=UnsupportedError):
        if err != ZE_OK:
            raise exc(f"intel (level zero): {what} failed (0x{err:x})")

    def _kernel(self, spirv_bytes, name):
        with self._lock:
            entry = self._modules.get(spirv_bytes)
            if entry is None:
                mdesc = _ModuleDesc(
                    stype=ST_MODULE_DESC, format=MODULE_FORMAT_IL_SPIRV,
                    inputSize=len(spirv_bytes),
                    pInputModule=spirv_bytes, pBuildFlags=b"")
                mod, log = ctypes.c_void_p(), ctypes.c_void_p()
                err = self._ze.zeModuleCreate(
                    self._ctx, self._dev, ctypes.byref(mdesc),
                    ctypes.byref(mod), ctypes.byref(log))
                if err != ZE_OK:
                    msg = ""
                    try:
                        sz = ctypes.c_size_t(0)
                        self._ze.zeModuleBuildLogGetString(
                            log, ctypes.byref(sz), None)
                        buf = ctypes.create_string_buffer(sz.value)
                        self._ze.zeModuleBuildLogGetString(
                            log, ctypes.byref(sz), buf)
                        msg = buf.value.decode("utf-8", "replace")
                    except Exception:
                        pass
                    raise UnsupportedError(
                        f"intel: SPIR-V module build failed (0x{err:x}): "
                        f"{msg[:500]}")
                entry = self._modules[spirv_bytes] = (mod, {})
            mod, kernels = entry
            k = kernels.get(name)
            if k is None:
                kdesc = _KernelDesc(stype=ST_KERNEL_DESC,
                                    pKernelName=name.encode())
                k = ctypes.c_void_p()
                self._check(self._ze.zeKernelCreate(
                    mod, ctypes.byref(kdesc), ctypes.byref(k)),
                    f"zeKernelCreate({name})")
                kernels[name] = k
            return k

    # ---- resident device buffers (DeviceArray protocol) ----
    def buf_alloc(self, arr):
        adesc = _MemAllocDesc(stype=ST_DEVICE_MEM_ALLOC_DESC)
        ptr = ctypes.c_void_p()
        self._check(self._ze.zeMemAllocDevice(
            self._ctx, ctypes.byref(adesc), ctypes.c_size_t(arr.nbytes),
            ctypes.c_size_t(64), self._dev, ctypes.byref(ptr)),
            "zeMemAllocDevice")
        return ptr

    def buf_write(self, ptr, arr):
        self._check(self._ze.zeCommandListAppendMemoryCopy(
            self._cl, ptr, ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes), None, 0, None), "copy to device")

    def buf_read(self, ptr, arr):
        self._check(self._ze.zeCommandListAppendMemoryCopy(
            self._cl, ctypes.c_void_p(arr.ctypes.data), ptr,
            ctypes.c_size_t(arr.nbytes), None, 0, None),
            "copy from device")

    def buf_free(self, ptr):
        self._ze.zeMemFree(self._ctx, ptr)

    def build(self, ast_payload):
        """Typed AST -> OpenCL-flavor SPIR-V binary."""
        from .. import spirv
        fn_ast, arg_types, var_types, ret_type = ast_payload
        return spirv.generate(fn_ast, arg_types, var_types, ret_type,
                              flavor="opencl")

    def launch(self, code, kernel_name, grid, block, args, sync=True):
        if isinstance(code, tuple):          # typed AST payload
            code = self.build(code)
        ze = self._ze
        kernel = self._kernel(code, kernel_name)
        run_list = self._cl if sync else self._async_list()
        adesc = _MemAllocDesc(stype=ST_DEVICE_MEM_ALLOC_DESC)
        dbufs = []
        try:
            for idx, (kind, v) in enumerate(args):
                if kind == "dev":
                    self._check(ze.zeKernelSetArgumentValue(
                        kernel, idx, ctypes.sizeof(v._impl),
                        ctypes.byref(v._impl)), "set resident ptr arg")
                elif kind == "arr":
                    ptr = ctypes.c_void_p()
                    self._check(ze.zeMemAllocDevice(
                        self._ctx, ctypes.byref(adesc),
                        ctypes.c_size_t(v.nbytes), ctypes.c_size_t(64),
                        self._dev, ctypes.byref(ptr)), "zeMemAllocDevice")
                    dbufs.append((ptr, v))
                    self._check(ze.zeCommandListAppendMemoryCopy(
                        self._cl, ptr, ctypes.c_void_p(v.ctypes.data),
                        ctypes.c_size_t(v.nbytes), None, 0, None),
                        "copy to device")
                    self._check(ze.zeKernelSetArgumentValue(
                        kernel, idx, ctypes.sizeof(ptr),
                        ctypes.byref(ptr)), "set pointer arg")
                elif kind == "i64":
                    val = ctypes.c_int64(v)
                    self._check(ze.zeKernelSetArgumentValue(
                        kernel, idx, 8, ctypes.byref(val)), "set i64 arg")
                elif kind == "f64":
                    val = ctypes.c_double(v)
                    self._check(ze.zeKernelSetArgumentValue(
                        kernel, idx, 8, ctypes.byref(val)), "set f64 arg")
                else:
                    raise UnsupportedError(
                        f"intel launch: unsupported arg kind {kind!r}")
            self._check(self._ze.zeKernelSetGroupSize(
                kernel, block[0], block[1], block[2]),
                "zeKernelSetGroupSize")
            gc = _GroupCount(grid[0], grid[1], grid[2])
            self._check(ze.zeCommandListAppendLaunchKernel(
                run_list, kernel, ctypes.byref(gc), None, 0, None),
                "zeCommandListAppendLaunchKernel")
            for ptr, arr in dbufs:
                self._check(ze.zeCommandListAppendMemoryCopy(
                    self._cl, ctypes.c_void_p(arr.ctypes.data), ptr,
                    ctypes.c_size_t(arr.nbytes), None, 0, None),
                    "copy from device")
        finally:
            for ptr, _ in dbufs:
                ze.zeMemFree(self._ctx, ptr)
