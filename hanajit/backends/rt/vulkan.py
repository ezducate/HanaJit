"""Vulkan compute launch bridge (vulkan-1), via ctypes.

Kernels are generated as shader-flavor SPIR-V by backends/spirv.py —
pointer arguments become DescriptorSet-0 storage buffers (binding =
argument position), scalar arguments travel in one std430 push-constant
block, and the workgroup size is baked into the module (Vulkan fixes
LocalSize at pipeline creation), so pipelines are cached per (kernel,
block) pair.

Vendor-neutral: any Vulkan 1.1+ device with shaderFloat64 + shaderInt64
can run hanajit kernels (both features are required because kernels use
f64 arithmetic and i64 indexing). Buffers use HOST_VISIBLE|HOST_COHERENT
memory — no staging copies; a compute->host memory barrier makes device
writes visible before readback.

All structure layouts and enum values follow vulkan_core.h.
"""
import ctypes
import hashlib
import struct as _struct
import threading

from . import RuntimeUnavailable
from ...errors import UnsupportedError

VK_OK = 0
API_VERSION_1_1 = (1 << 22) | (1 << 12)
ST_APPLICATION_INFO = 0
ST_INSTANCE_CREATE = 1
ST_DEVICE_QUEUE_CREATE = 2
ST_DEVICE_CREATE = 3
ST_SUBMIT_INFO = 4
ST_MEMORY_ALLOCATE = 5
ST_BUFFER_CREATE = 12
ST_SHADER_MODULE_CREATE = 16
ST_PIPELINE_SHADER_STAGE = 18
ST_COMPUTE_PIPELINE_CREATE = 29
ST_PIPELINE_LAYOUT_CREATE = 30
ST_DESC_SET_LAYOUT_CREATE = 32
ST_DESC_POOL_CREATE = 33
ST_DESC_SET_ALLOCATE = 34
ST_WRITE_DESC_SET = 35
ST_COMMAND_POOL_CREATE = 39
ST_COMMAND_BUFFER_ALLOCATE = 40
ST_COMMAND_BUFFER_BEGIN = 42
ST_MEMORY_BARRIER = 46
QUEUE_COMPUTE_BIT = 0x2
BUFFER_USAGE_STORAGE = 0x20
MEM_HOST_VISIBLE = 0x2
MEM_HOST_COHERENT = 0x4
DESC_TYPE_STORAGE_BUFFER = 7
SHADER_STAGE_COMPUTE = 0x20
BIND_POINT_COMPUTE = 1
CB_LEVEL_PRIMARY = 0
DEVICE_TYPE_DISCRETE = 2
ACCESS_SHADER_WRITE = 0x40
ACCESS_HOST_READ = 0x2000
STAGE_COMPUTE_SHADER = 0x800
STAGE_HOST = 0x4000
FEATURE_COUNT = 55
IDX_SHADER_FLOAT64 = 39
IDX_SHADER_INT64 = 40
ST_FEATURES_2 = 1000059000
ST_VULKAN_1_2_FEATURES = 51
ST_ATOMIC_FLOAT_FEATURES_EXT = 1000260000
V12_FEATURE_COUNT = 47
IDX12_BUFFER_INT64_ATOMICS = 5
AF_FEATURE_COUNT = 12
IDX_AF_BUF_F64_ATOMICS = 2
IDX_AF_BUF_F64_ATOMIC_ADD = 3
EXT_ATOMIC_FLOAT = b"VK_EXT_shader_atomic_float"
API_VERSION_1_2 = (1 << 22) | (2 << 12)

_p = ctypes.c_void_p
_u32 = ctypes.c_uint32
_u64 = ctypes.c_uint64
_i32 = ctypes.c_int32
_sz = ctypes.c_size_t


class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("pApplicationName", ctypes.c_char_p), ("appVer", _u32),
                ("pEngineName", ctypes.c_char_p), ("engVer", _u32),
                ("apiVersion", _u32)]


class _InstanceCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("pApplicationInfo", _p),
                ("layerCount", _u32), ("ppLayers", _p),
                ("extCount", _u32), ("ppExts", _p)]


class _QueueCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("queueFamilyIndex", _u32), ("queueCount", _u32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class _DeviceCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("queueCICount", _u32), ("pQueueCIs", _p),
                ("layerCount", _u32), ("ppLayers", _p),
                ("extCount", _u32), ("ppExts", _p),
                ("pEnabledFeatures", _p)]


class _QueueFamilyProps(ctypes.Structure):
    _fields_ = [("queueFlags", _u32), ("queueCount", _u32),
                ("timestampValidBits", _u32),
                ("granularity", _u32 * 3)]


class _MemType(ctypes.Structure):
    _fields_ = [("propertyFlags", _u32), ("heapIndex", _u32)]


class _MemHeap(ctypes.Structure):
    _fields_ = [("size", _u64), ("flags", _u32)]


class _MemProps(ctypes.Structure):
    _fields_ = [("memoryTypeCount", _u32), ("memoryTypes", _MemType * 32),
                ("memoryHeapCount", _u32), ("memoryHeaps", _MemHeap * 16)]


class _BufferCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("size", _u64), ("usage", _u32), ("sharingMode", _i32),
                ("qfCount", _u32), ("pQFIndices", _p)]


class _MemReq(ctypes.Structure):
    _fields_ = [("size", _u64), ("alignment", _u64),
                ("memoryTypeBits", _u32)]


class _MemAI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("allocationSize", _u64), ("memoryTypeIndex", _u32)]


class _ShaderModuleCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("codeSize", _sz), ("pCode", _p)]


class _DSLBinding(ctypes.Structure):
    _fields_ = [("binding", _u32), ("descriptorType", _i32),
                ("descriptorCount", _u32), ("stageFlags", _u32),
                ("pImmutableSamplers", _p)]


class _DSLCreate(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("bindingCount", _u32), ("pBindings", _p)]


class _PushRange(ctypes.Structure):
    _fields_ = [("stageFlags", _u32), ("offset", _u32), ("size", _u32)]


class _PipelineLayoutCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("setLayoutCount", _u32), ("pSetLayouts", _p),
                ("pushRangeCount", _u32), ("pPushRanges", _p)]


class _StageCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("stage", _i32), ("module", _u64),
                ("pName", ctypes.c_char_p), ("pSpecInfo", _p)]


class _ComputePipelineCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("stage", _StageCI), ("layout", _u64),
                ("basePipeline", _u64), ("baseIndex", _i32)]


class _PoolSize(ctypes.Structure):
    _fields_ = [("type", _i32), ("descriptorCount", _u32)]


class _DescPoolCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("maxSets", _u32), ("poolSizeCount", _u32),
                ("pPoolSizes", _p)]


class _DescSetAI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("descriptorPool", _u64),
                ("descriptorSetCount", _u32), ("pSetLayouts", _p)]


class _DescBufferInfo(ctypes.Structure):
    _fields_ = [("buffer", _u64), ("offset", _u64), ("range", _u64)]


class _WriteDescSet(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("dstSet", _u64),
                ("dstBinding", _u32), ("dstArrayElement", _u32),
                ("descriptorCount", _u32), ("descriptorType", _i32),
                ("pImageInfo", _p), ("pBufferInfo", _p),
                ("pTexelBufferView", _p)]


class _CmdPoolCI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("queueFamilyIndex", _u32)]


class _CmdBufAI(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("commandPool", _u64),
                ("level", _i32), ("commandBufferCount", _u32)]


class _CmdBufBegin(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p), ("flags", _u32),
                ("pInheritanceInfo", _p)]


class _MemBarrier(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("srcAccessMask", _u32), ("dstAccessMask", _u32)]


class _Features2(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("features", _u32 * FEATURE_COUNT)]


class _Vulkan12Features(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("f", _u32 * V12_FEATURE_COUNT)]


class _AtomicFloatFeatures(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("f", _u32 * AF_FEATURE_COUNT)]


class _SubmitInfo(ctypes.Structure):
    _fields_ = [("sType", _i32), ("pNext", _p),
                ("waitCount", _u32), ("pWaitSems", _p),
                ("pWaitStages", _p), ("cbCount", _u32), ("pCBs", _p),
                ("signalCount", _u32), ("pSignalSems", _p)]


def _load():
    import platform
    names = (["vulkan-1"] if platform.system() == "Windows"
             else ["libvulkan.so.1", "libvulkan.so"])
    for n in names:
        try:
            return (ctypes.WinDLL(n) if platform.system() == "Windows"
                    else ctypes.CDLL(n))
        except OSError:
            continue
    raise RuntimeUnavailable("Vulkan loader not found "
                             "(vulkan-1.dll / libvulkan.so)")


class Runtime:
    vendor = "vulkan"
    code_kind = "ast"

    def __init__(self):
        vk = self._vk = _load()
        self._lock = threading.Lock()
        self._pipelines = {}   # (kernel-hash, block) -> pipeline bundle

        app = _AppInfo(sType=ST_APPLICATION_INFO,
                       pApplicationName=b"hanajit",
                       apiVersion=API_VERSION_1_1)
        ici = _InstanceCI(sType=ST_INSTANCE_CREATE,
                          pApplicationInfo=ctypes.cast(
                              ctypes.byref(app), _p))
        inst = _p()
        if vk.vkCreateInstance(ctypes.byref(ici), None,
                               ctypes.byref(inst)) != VK_OK:
            raise RuntimeUnavailable(
                "vkCreateInstance failed (Vulkan 1.1 required for "
                "shader-flavor SPIR-V 1.3)")
        self._inst = inst

        n = _u32(0)
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        if n.value == 0:
            raise RuntimeUnavailable("no Vulkan physical devices")
        devs = (_p * n.value)()
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)

        from . import device_index
        eligible = []   # (is_discrete, dev, family, name) in enum order
        for d in devs:
            d = _p(d)
            feats = (_u32 * FEATURE_COUNT)()
            vk.vkGetPhysicalDeviceFeatures(d, ctypes.byref(feats))
            if not (feats[IDX_SHADER_FLOAT64] and feats[IDX_SHADER_INT64]):
                continue  # kernels use f64 math and i64 indexing
            qn = _u32(0)
            vk.vkGetPhysicalDeviceQueueFamilyProperties(
                d, ctypes.byref(qn), None)
            qprops = (_QueueFamilyProps * qn.value)()
            vk.vkGetPhysicalDeviceQueueFamilyProperties(
                d, ctypes.byref(qn), qprops)
            family = next((i for i in range(qn.value)
                           if qprops[i].queueFlags & QUEUE_COMPUTE_BIT),
                          None)
            if family is None:
                continue
            props = ctypes.create_string_buffer(4096)
            vk.vkGetPhysicalDeviceProperties(d, props)
            dev_type = _struct.unpack_from("<i", props, 16)[0]
            name = props.raw[20:276].split(b"\0")[0].decode(
                "utf-8", "replace")
            eligible.append((dev_type == DEVICE_TYPE_DISCRETE, d,
                             family, name))
        if not eligible:
            raise RuntimeUnavailable(
                "no Vulkan device with shaderFloat64 + shaderInt64 and a "
                "compute queue")
        import os
        if "HANAJIT_VULKAN_DEVICE" in os.environ:
            want = device_index("vulkan")
            if want >= len(eligible):
                raise RuntimeUnavailable(
                    f"HANAJIT_VULKAN_DEVICE={want} but only "
                    f"{len(eligible)} eligible Vulkan device(s)")
            best = eligible[want]
        else:   # default policy: prefer the discrete GPU
            best = max(eligible, key=lambda c: c[0])
        _, self._pdev, self._family, self.device_name = best

        # probe optional atomic features (Vulkan 1.2 core + EXT), so
        # atomic_add kernels can be gated with precise error messages
        props = ctypes.create_string_buffer(4096)
        vk.vkGetPhysicalDeviceProperties(self._pdev, props)
        dev_api = _struct.unpack_from("<I", props, 0)[0]
        self.has_int64_atomics = False
        self.has_f64_atomic_add = False
        q12 = _Vulkan12Features(sType=ST_VULKAN_1_2_FEATURES)
        qaf = _AtomicFloatFeatures(sType=ST_ATOMIC_FLOAT_FEATURES_EXT,
                                   pNext=ctypes.cast(
                                       ctypes.byref(q12), _p))
        qf2 = _Features2(sType=ST_FEATURES_2,
                         pNext=ctypes.cast(ctypes.byref(qaf), _p))
        if dev_api >= API_VERSION_1_2:
            vk.vkGetPhysicalDeviceFeatures2(self._pdev,
                                            ctypes.byref(qf2))
            self.has_int64_atomics = bool(
                q12.f[IDX12_BUFFER_INT64_ATOMICS])
            self.has_f64_atomic_add = bool(
                qaf.f[IDX_AF_BUF_F64_ATOMICS]
                and qaf.f[IDX_AF_BUF_F64_ATOMIC_ADD])

        prio = (ctypes.c_float * 1)(1.0)
        qci = _QueueCI(sType=ST_DEVICE_QUEUE_CREATE,
                       queueFamilyIndex=self._family, queueCount=1,
                       pQueuePriorities=prio)
        feats2 = _Features2(sType=ST_FEATURES_2)
        feats2.features[IDX_SHADER_FLOAT64] = 1
        feats2.features[IDX_SHADER_INT64] = 1
        e12 = _Vulkan12Features(sType=ST_VULKAN_1_2_FEATURES)
        eaf = _AtomicFloatFeatures(sType=ST_ATOMIC_FLOAT_FEATURES_EXT)
        chain = ctypes.c_void_p(None)
        exts = []
        if self.has_int64_atomics:
            e12.f[IDX12_BUFFER_INT64_ATOMICS] = 1
            e12.pNext = chain
            chain = ctypes.cast(ctypes.byref(e12), _p)
        if self.has_f64_atomic_add:
            eaf.f[IDX_AF_BUF_F64_ATOMICS] = 1
            eaf.f[IDX_AF_BUF_F64_ATOMIC_ADD] = 1
            eaf.pNext = chain
            chain = ctypes.cast(ctypes.byref(eaf), _p)
            exts.append(EXT_ATOMIC_FLOAT)
        feats2.pNext = chain
        ext_arr = (ctypes.c_char_p * max(len(exts), 1))(*exts) \
            if exts else None
        dci = _DeviceCI(sType=ST_DEVICE_CREATE, queueCICount=1,
                        pQueueCIs=ctypes.cast(ctypes.byref(qci), _p),
                        extCount=len(exts),
                        ppExts=ctypes.cast(ext_arr, _p) if exts else None,
                        pEnabledFeatures=None)
        dci.pNext = ctypes.cast(ctypes.byref(feats2), _p)
        dev = _p()
        self._check(vk.vkCreateDevice(self._pdev, ctypes.byref(dci),
                                      None, ctypes.byref(dev)),
                    "vkCreateDevice", RuntimeUnavailable)
        self._dev = dev
        q = _p()
        vk.vkGetDeviceQueue(dev, self._family, 0, ctypes.byref(q))
        self._queue = q
        self._memprops = _MemProps()
        vk.vkGetPhysicalDeviceMemoryProperties(
            self._pdev, ctypes.byref(self._memprops))
        cpci = _CmdPoolCI(sType=ST_COMMAND_POOL_CREATE,
                          queueFamilyIndex=self._family)
        pool = _u64()
        self._check(vk.vkCreateCommandPool(dev, ctypes.byref(cpci), None,
                                           ctypes.byref(pool)),
                    "vkCreateCommandPool", RuntimeUnavailable)
        self._cmdpool = pool
        self._pending = []   # (cb, dpool) of queued sync=False launches

    def sync(self):
        """Wait for queued (sync=False) launches, then release their
        command buffers and descriptor pools."""
        if not self._pending:
            return
        self._check(self._vk.vkQueueWaitIdle(self._queue),
                    "vkQueueWaitIdle")
        pending, self._pending = self._pending, []
        for cb, dpool in pending:
            self._vk.vkFreeCommandBuffers(self._dev, self._cmdpool, 1,
                                          ctypes.byref(cb))
            self._vk.vkDestroyDescriptorPool(self._dev, dpool, None)

    def _check(self, err, what, exc=UnsupportedError):
        if err != VK_OK:
            raise exc(f"vulkan: {what} failed ({err})")

    def _memtype(self, type_bits, flags):
        mp = self._memprops
        for i in range(mp.memoryTypeCount):
            if (type_bits & (1 << i)) and \
                    (mp.memoryTypes[i].propertyFlags & flags) == flags:
                return i
        raise UnsupportedError("vulkan: no host-visible coherent memory")

    def _check_atomics(self, fn_ast, arg_types, var_types):
        """Fail early, with the exact missing device feature, instead of
        letting pipeline creation die opaquely."""
        import ast as _ast
        for n in _ast.walk(fn_ast):
            if not (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Name)
                    and n.func.id == "atomic_add"
                    and n.args and isinstance(n.args[0], _ast.Name)):
                continue
            name = n.args[0].id
            t = var_types.get(name) or arg_types.get(name)
            shared = name not in arg_types
            if t == "i64*" and not self.has_int64_atomics:
                raise UnsupportedError(
                    "vulkan: device lacks shaderBufferInt64Atomics "
                    "(Vulkan 1.2) needed for i64 atomic_add")
            if t == "f64*":
                if shared:
                    raise UnsupportedError(
                        "vulkan: f64 atomic_add on shared_f64 arrays is "
                        "not supported (buffer arguments only)")
                if not self.has_f64_atomic_add:
                    raise UnsupportedError(
                        "vulkan: device lacks VK_EXT_shader_atomic_float "
                        "shaderBufferFloat64AtomicAdd needed for f64 "
                        "atomic_add")

    # ------------------------ pipeline bundles ------------------------
    def _bundle(self, ast_payload, block):
        import ast as _ast
        from .. import spirv
        fn_ast, arg_types, var_types, ret_type = ast_payload
        self._check_atomics(fn_ast, arg_types, var_types)
        key = (hashlib.sha1(_ast.dump(fn_ast).encode()
                            + repr(sorted(arg_types.items())).encode()
                            ).hexdigest(), block)
        with self._lock:
            b = self._pipelines.get(key)
            if b is not None:
                return b
        code = spirv.generate(fn_ast, arg_types, var_types, ret_type,
                              flavor="vulkan", local_size=block)
        vk = self._vk
        n_buf = sum(1 for a in fn_ast.args.args
                    if arg_types[a.arg] in ("f64*", "i64*"))
        n_scalar = sum(1 for a in fn_ast.args.args
                       if arg_types[a.arg] in ("i64", "f64"))

        smci = _ShaderModuleCI(sType=ST_SHADER_MODULE_CREATE,
                               codeSize=len(code),
                               pCode=ctypes.cast(
                                   ctypes.c_char_p(code), _p))
        shader = _u64()
        self._check(vk.vkCreateShaderModule(
            self._dev, ctypes.byref(smci), None, ctypes.byref(shader)),
            "vkCreateShaderModule")

        bindings = (_DSLBinding * max(n_buf, 1))()
        for i in range(n_buf):
            bindings[i] = _DSLBinding(
                binding=i, descriptorType=DESC_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=SHADER_STAGE_COMPUTE)
        dslci = _DSLCreate(sType=ST_DESC_SET_LAYOUT_CREATE,
                           bindingCount=n_buf,
                           pBindings=ctypes.cast(bindings, _p))
        dsl = _u64()
        self._check(vk.vkCreateDescriptorSetLayout(
            self._dev, ctypes.byref(dslci), None, ctypes.byref(dsl)),
            "vkCreateDescriptorSetLayout")

        push = _PushRange(stageFlags=SHADER_STAGE_COMPUTE, offset=0,
                          size=8 * n_scalar)
        plci = _PipelineLayoutCI(
            sType=ST_PIPELINE_LAYOUT_CREATE, setLayoutCount=1,
            pSetLayouts=ctypes.cast(ctypes.byref(dsl), _p),
            pushRangeCount=1 if n_scalar else 0,
            pPushRanges=ctypes.cast(ctypes.byref(push), _p)
            if n_scalar else None)
        layout = _u64()
        self._check(vk.vkCreatePipelineLayout(
            self._dev, ctypes.byref(plci), None, ctypes.byref(layout)),
            "vkCreatePipelineLayout")

        entry = fn_ast.name.encode()
        cpci = _ComputePipelineCI(
            sType=ST_COMPUTE_PIPELINE_CREATE,
            stage=_StageCI(sType=ST_PIPELINE_SHADER_STAGE,
                           stage=SHADER_STAGE_COMPUTE,
                           module=shader.value, pName=entry),
            layout=layout.value)
        pipeline = _u64()
        self._check(vk.vkCreateComputePipelines(
            self._dev, None, 1, ctypes.byref(cpci), None,
            ctypes.byref(pipeline)), "vkCreateComputePipelines")

        bundle = dict(pipeline=pipeline, layout=layout, dsl=dsl,
                      shader=shader, n_buf=n_buf, n_scalar=n_scalar,
                      entry=entry)
        with self._lock:
            self._pipelines[key] = bundle
        return bundle

    # -------------------- buffers (shared helpers) --------------------
    def _make_buffer(self, nbytes):
        """Create a mapped HOST_VISIBLE|COHERENT storage buffer.
        Returns (buf, mem, mapped_ptr)."""
        vk = self._vk
        bci = _BufferCI(sType=ST_BUFFER_CREATE, size=nbytes,
                        usage=BUFFER_USAGE_STORAGE)
        buf = _u64()
        self._check(vk.vkCreateBuffer(self._dev, ctypes.byref(bci), None,
                                      ctypes.byref(buf)), "vkCreateBuffer")
        req = _MemReq()
        vk.vkGetBufferMemoryRequirements(self._dev, buf, ctypes.byref(req))
        mai = _MemAI(sType=ST_MEMORY_ALLOCATE, allocationSize=req.size,
                     memoryTypeIndex=self._memtype(
                         req.memoryTypeBits,
                         MEM_HOST_VISIBLE | MEM_HOST_COHERENT))
        mem = _u64()
        self._check(vk.vkAllocateMemory(self._dev, ctypes.byref(mai),
                                        None, ctypes.byref(mem)),
                    "vkAllocateMemory")
        self._check(vk.vkBindBufferMemory(self._dev, buf, mem, _u64(0)),
                    "vkBindBufferMemory")
        ptr = _p()
        self._check(vk.vkMapMemory(self._dev, mem, _u64(0), _u64(nbytes),
                                   0, ctypes.byref(ptr)), "vkMapMemory")
        return buf, mem, ptr

    def _drop_buffer(self, buf, mem):
        self._vk.vkUnmapMemory(self._dev, mem)
        self._vk.vkDestroyBuffer(self._dev, buf, None)
        self._vk.vkFreeMemory(self._dev, mem, None)

    # ---- resident device buffers (DeviceArray protocol) ----
    def buf_alloc(self, arr):
        buf, mem, ptr = self._make_buffer(arr.nbytes)
        return {"buf": buf, "mem": mem, "map": ptr, "nbytes": arr.nbytes}

    def buf_write(self, impl, arr):
        ctypes.memmove(impl["map"], arr.ctypes.data, impl["nbytes"])

    def buf_read(self, impl, arr):
        ctypes.memmove(arr.ctypes.data, impl["map"], impl["nbytes"])

    def buf_free(self, impl):
        self._drop_buffer(impl["buf"], impl["mem"])

    # ----------------------------- launch -----------------------------
    def launch(self, code, kernel_name, grid, block, args, sync=True):
        if not isinstance(code, tuple):
            raise UnsupportedError("vulkan: bridge expects the typed AST")
        vk = self._vk
        bundle = self._bundle(code, block)

        # unified buffer list, in pointer-argument order:
        # (kind, value) where kind is "arr" (transient) or "dev" (resident)
        pointer_args = [(k, v) for k, v in args if k in ("arr", "dev")]
        scalars = [(k, v) for k, v in args if k in ("i64", "f64")]
        if len(pointer_args) != bundle["n_buf"] or \
                len(scalars) != bundle["n_scalar"]:
            raise UnsupportedError("vulkan: argument mismatch")

        bound = []    # (buf_handle, nbytes) per binding
        owned = []    # (buf, mem, map, arr) — transient, copy back + free
        dpool = _u64()
        cb = _p()
        try:
            for kind, v in pointer_args:
                if kind == "dev":
                    bound.append((v._impl["buf"], v._impl["nbytes"]))
                else:
                    buf, mem, ptr = self._make_buffer(v.nbytes)
                    ctypes.memmove(ptr, v.ctypes.data, v.nbytes)
                    owned.append((buf, mem, ptr, v))
                    bound.append((buf, v.nbytes))

            ps = _PoolSize(type=DESC_TYPE_STORAGE_BUFFER,
                           descriptorCount=max(bundle["n_buf"], 1))
            dpci = _DescPoolCI(sType=ST_DESC_POOL_CREATE, maxSets=1,
                               poolSizeCount=1,
                               pPoolSizes=ctypes.cast(
                                   ctypes.byref(ps), _p))
            self._check(vk.vkCreateDescriptorPool(
                self._dev, ctypes.byref(dpci), None,
                ctypes.byref(dpool)), "vkCreateDescriptorPool")
            dsai = _DescSetAI(sType=ST_DESC_SET_ALLOCATE,
                              descriptorPool=dpool.value,
                              descriptorSetCount=1,
                              pSetLayouts=ctypes.cast(
                                  ctypes.byref(bundle["dsl"]), _p))
            dset = _u64()
            self._check(vk.vkAllocateDescriptorSets(
                self._dev, ctypes.byref(dsai), ctypes.byref(dset)),
                "vkAllocateDescriptorSets")

            infos = (_DescBufferInfo * max(len(bound), 1))()
            writes = (_WriteDescSet * max(len(bound), 1))()
            for i, (buf, nbytes) in enumerate(bound):
                infos[i] = _DescBufferInfo(buffer=buf.value, offset=0,
                                           range=nbytes)
                writes[i] = _WriteDescSet(
                    sType=ST_WRITE_DESC_SET, dstSet=dset.value,
                    dstBinding=i, descriptorCount=1,
                    descriptorType=DESC_TYPE_STORAGE_BUFFER,
                    pBufferInfo=ctypes.cast(ctypes.byref(infos[i]), _p))
            if bound:
                vk.vkUpdateDescriptorSets(self._dev, len(bound), writes,
                                          0, None)

            cbai = _CmdBufAI(sType=ST_COMMAND_BUFFER_ALLOCATE,
                             commandPool=self._cmdpool.value,
                             level=CB_LEVEL_PRIMARY,
                             commandBufferCount=1)  # struct field: u64 ok
            self._check(vk.vkAllocateCommandBuffers(
                self._dev, ctypes.byref(cbai), ctypes.byref(cb)),
                "vkAllocateCommandBuffers")
            begin = _CmdBufBegin(sType=ST_COMMAND_BUFFER_BEGIN)
            self._check(vk.vkBeginCommandBuffer(cb, ctypes.byref(begin)),
                        "vkBeginCommandBuffer")
            vk.vkCmdBindPipeline(cb, BIND_POINT_COMPUTE,
                                 bundle["pipeline"])
            if bound:
                vk.vkCmdBindDescriptorSets(
                    cb, BIND_POINT_COMPUTE, bundle["layout"], 0, 1,
                    ctypes.byref(dset), 0, None)
            if scalars:
                blob = b"".join(
                    _struct.pack("<q", v) if k == "i64"
                    else _struct.pack("<d", v) for k, v in scalars)
                vk.vkCmdPushConstants(cb, bundle["layout"],
                                      SHADER_STAGE_COMPUTE, 0,
                                      len(blob), blob)
            vk.vkCmdDispatch(cb, grid[0], grid[1], grid[2])
            barrier = _MemBarrier(sType=ST_MEMORY_BARRIER,
                                  srcAccessMask=ACCESS_SHADER_WRITE,
                                  dstAccessMask=ACCESS_HOST_READ)
            vk.vkCmdPipelineBarrier(cb, STAGE_COMPUTE_SHADER, STAGE_HOST,
                                    0, 1, ctypes.byref(barrier),
                                    0, None, 0, None)
            self._check(vk.vkEndCommandBuffer(cb), "vkEndCommandBuffer")

            si = _SubmitInfo(sType=ST_SUBMIT_INFO, cbCount=1,
                             pCBs=ctypes.cast(ctypes.byref(cb), _p))
            self._check(vk.vkQueueSubmit(
                self._queue, 1, ctypes.byref(si), _u64(0)),
                "vkQueueSubmit")
            if not sync:
                # DeviceArray-only (enforced upstream): nothing owned to
                # copy back; the cb + descriptor pool stay alive until
                # sync() releases them
                self._pending.append((cb, dpool))
                cb, dpool = _p(), _u64()
                return
            self._check(vk.vkQueueWaitIdle(self._queue),
                        "vkQueueWaitIdle")

            for _, _, ptr, arr in owned:
                ctypes.memmove(arr.ctypes.data, ptr, arr.nbytes)
        finally:
            if cb:
                vk.vkFreeCommandBuffers(self._dev, self._cmdpool,
                                        1, ctypes.byref(cb))
            if dpool.value:
                vk.vkDestroyDescriptorPool(self._dev, dpool, None)
            for buf, mem, _, _ in owned:
                self._drop_buffer(buf, mem)
