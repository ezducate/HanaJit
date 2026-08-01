"""Apple GPU launch bridge over Metal, via ctypes + objc_msgSend.

macOS only. The metal backend transpiles the typed AST to MSL source
(LLVM has no Metal target); this bridge compiles that source at runtime
with newLibraryWithSource: and dispatches it on the system default
device — no Xcode, no xcrun, just Metal.framework.

Precision: Metal has no double. The MSL transpiler lowers f64 to 32-bit
float, so this bridge converts f64 arrays to float32 on the way in and
back on the way out — results carry float32 precision, exactly as the
metal backend documents for its generated source. i64 stays 64-bit
(MSL `long`).

Status: written against the documented Metal/objc ABIs but not yet
validated on Apple hardware (developed on Windows); treat the first
macOS run as the integration test.
"""
import ctypes
import hashlib
import threading

from . import RuntimeUnavailable
from ...errors import UnsupportedError

MTL_RESOURCE_STORAGE_SHARED = 0  # MTLResourceStorageModeShared


class _MTLSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_ulong), ("height", ctypes.c_ulong),
                ("depth", ctypes.c_ulong)]


class Runtime:
    vendor = "metal"
    code_kind = "ast"

    def __init__(self):
        import platform
        if platform.system() != "Darwin":
            raise RuntimeUnavailable("Metal requires macOS")
        try:
            self._objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
            self._metal = ctypes.CDLL(
                "/System/Library/Frameworks/Metal.framework/Metal")
            self._cf = ctypes.CDLL("/System/Library/Frameworks/"
                                   "CoreFoundation.framework/CoreFoundation")
        except OSError as e:
            raise RuntimeUnavailable(f"Metal framework not loadable: {e}")
        self._objc.sel_registerName.restype = ctypes.c_void_p
        self._objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self._objc.objc_msgSend.restype = ctypes.c_void_p
        self._metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self._cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        self._lock = threading.Lock()
        self._pipelines = {}

        self._dev = self._metal.MTLCreateSystemDefaultDevice()
        if not self._dev:
            raise RuntimeUnavailable("no Metal device")
        name_obj = self._msg(self._dev, b"name", ctypes.c_void_p, [])
        utf8 = self._msg(name_obj, b"UTF8String", ctypes.c_char_p, [])
        self.device_name = (utf8 or b"Apple GPU").decode("utf-8", "replace")
        self._queue = self._msg(self._dev, b"newCommandQueue",
                                ctypes.c_void_p, [])
        if not self._queue:
            raise RuntimeUnavailable("newCommandQueue failed")

    # objc_msgSend must be cast per call signature (ABI requirement)
    def _msg(self, receiver, sel, restype, argtypes, *args):
        proto = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p,
                                 *argtypes)
        fn = ctypes.cast(self._objc.objc_msgSend, proto)
        return fn(receiver, self._objc.sel_registerName(sel), *args)

    def _nsstring(self, s):
        return self._cf.CFStringCreateWithCString(
            None, s.encode("utf-8"), 0x08000100)  # kCFStringEncodingUTF8

    def _error_text(self, err_obj):
        try:
            d = self._msg(err_obj, b"localizedDescription",
                          ctypes.c_void_p, [])
            u = self._msg(d, b"UTF8String", ctypes.c_char_p, [])
            return (u or b"").decode("utf-8", "replace")
        except Exception:
            return "unknown Metal error"

    def _pipeline(self, msl_source, kernel_name):
        key = hashlib.sha1(msl_source.encode()).hexdigest()
        with self._lock:
            hit = self._pipelines.get(key)
            if hit is not None:
                return hit
        err = ctypes.c_void_p()
        lib = self._msg(self._dev, b"newLibraryWithSource:options:error:",
                        ctypes.c_void_p,
                        [ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p)],
                        self._nsstring(msl_source), None,
                        ctypes.byref(err))
        if not lib:
            raise UnsupportedError(
                f"metal: MSL compile failed: {self._error_text(err)}")
        fn = self._msg(lib, b"newFunctionWithName:", ctypes.c_void_p,
                       [ctypes.c_void_p], self._nsstring(kernel_name))
        if not fn:
            raise UnsupportedError(
                f"metal: kernel {kernel_name!r} not found in library")
        err = ctypes.c_void_p()
        pso = self._msg(self._dev,
                        b"newComputePipelineStateWithFunction:error:",
                        ctypes.c_void_p,
                        [ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p)],
                        fn, ctypes.byref(err))
        if not pso:
            raise UnsupportedError(
                f"metal: pipeline creation failed: "
                f"{self._error_text(err)}")
        with self._lock:
            self._pipelines[key] = pso
        return pso

    # ---- resident device buffers (DeviceArray protocol) ----
    # Metal computes f64 at float32 (no double in MSL), so an f64
    # DeviceArray is stored as a float32 buffer and converted at the
    # host boundary — same precision contract as transient launches.
    def buf_alloc(self, arr):
        import numpy as np
        f32 = str(arr.dtype) == "float64"
        nbytes = arr.nbytes // 2 if f32 else arr.nbytes
        buf = self._msg(self._dev, b"newBufferWithLength:options:",
                        ctypes.c_void_p,
                        [ctypes.c_ulong, ctypes.c_ulong],
                        nbytes, MTL_RESOURCE_STORAGE_SHARED)
        if not buf:
            raise UnsupportedError("metal: buffer allocation failed")
        return {"buf": buf, "nbytes": nbytes, "f32": f32}

    def buf_write(self, impl, arr):
        import numpy as np
        data = arr.astype(np.float32) if impl["f32"] else arr
        contents = self._msg(impl["buf"], b"contents", ctypes.c_void_p, [])
        ctypes.memmove(contents, data.ctypes.data, impl["nbytes"])

    def buf_read(self, impl, arr):
        import numpy as np
        contents = self._msg(impl["buf"], b"contents", ctypes.c_void_p, [])
        if impl["f32"]:
            tmp = np.empty(arr.shape, dtype=np.float32)
            ctypes.memmove(tmp.ctypes.data, contents, impl["nbytes"])
            arr[:] = tmp
        else:
            ctypes.memmove(arr.ctypes.data, contents, impl["nbytes"])

    def buf_free(self, impl):
        self._msg(impl["buf"], b"release", None, [])

    _pending = ()   # retained sync=False command buffers

    def sync(self):
        for cb in self._pending:
            self._msg(cb, b"waitUntilCompleted", None, [])
            self._msg(cb, b"release", None, [])
        self._pending = ()

    def launch(self, code, kernel_name, grid, block, args, sync=True):
        if not isinstance(code, tuple):
            raise UnsupportedError("metal: bridge expects the typed AST")
        import numpy as np
        from .. import metal as metal_backend
        fn_ast, arg_types, var_types, ret_type = code
        msl = metal_backend.transpile(fn_ast, arg_types, var_types,
                                      ret_type)
        pso = self._pipeline(msl, kernel_name)

        pool = self._objc.objc_autoreleasePoolPush()
        try:
            cb = self._msg(self._queue, b"commandBuffer",
                           ctypes.c_void_p, [])
            enc = self._msg(cb, b"computeCommandEncoder",
                            ctypes.c_void_p, [])
            self._msg(enc, b"setComputePipelineState:", None,
                      [ctypes.c_void_p], pso)

            buffers = []   # (mtl_buffer, f32_view, original_arr)
            for idx, (kind, v) in enumerate(args):
                if kind == "dev":
                    self._msg(enc, b"setBuffer:offset:atIndex:", None,
                              [ctypes.c_void_p, ctypes.c_ulong,
                               ctypes.c_ulong], v._impl["buf"], 0, idx)
                elif kind == "arr":
                    data = (v.astype(np.float32)
                            if v.dtype == np.float64 else v)
                    buf = self._msg(
                        self._dev, b"newBufferWithBytes:length:options:",
                        ctypes.c_void_p,
                        [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong],
                        ctypes.c_void_p(data.ctypes.data),
                        data.nbytes, MTL_RESOURCE_STORAGE_SHARED)
                    buffers.append((buf, data, v))
                    self._msg(enc, b"setBuffer:offset:atIndex:", None,
                              [ctypes.c_void_p, ctypes.c_ulong,
                               ctypes.c_ulong], buf, 0, idx)
                elif kind in ("i64", "f64"):
                    # MSL scalars are constant& parameters: f64 lowered
                    # to float, i64 stays long
                    val = (ctypes.c_float(v) if kind == "f64"
                           else ctypes.c_int64(v))
                    self._msg(enc, b"setBytes:length:atIndex:", None,
                              [ctypes.c_void_p, ctypes.c_ulong,
                               ctypes.c_ulong],
                              ctypes.cast(ctypes.byref(val),
                                          ctypes.c_void_p),
                              ctypes.sizeof(val), idx)
                else:
                    raise UnsupportedError(
                        f"metal launch: unsupported arg kind {kind!r}")

            self._msg(enc, b"dispatchThreadgroups:threadsPerThreadgroup:",
                      None, [_MTLSize, _MTLSize],
                      _MTLSize(*grid), _MTLSize(*block))
            self._msg(enc, b"endEncoding", None, [])
            self._msg(cb, b"commit", None, [])
            if not sync:
                # retain past the autorelease pool; released in sync()
                self._msg(cb, b"retain", ctypes.c_void_p, [])
                self._pending = tuple(self._pending) + (cb,)
                return
            self._msg(cb, b"waitUntilCompleted", None, [])

            for buf, data, orig in buffers:
                contents = self._msg(buf, b"contents", ctypes.c_void_p, [])
                ctypes.memmove(data.ctypes.data, contents, data.nbytes)
                if data is not orig:
                    orig[:] = data   # float32 round-trip back into f64
        finally:
            self._objc.objc_autoreleasePoolPop(pool)
