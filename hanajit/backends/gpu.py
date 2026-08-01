"""Multi-vendor GPU backends (experimental): NVIDIA, AMD, Intel, Vulkan.

All retarget the same LLVM IR; only the triple, datalayout, and kernel
calling convention / entry-point annotations differ:

- NVIDIA: nvptx64 triple + nvvm.annotations  -> PTX text
- AMD:    amdgcn-amd-amdhsa + amdgpu_kernel  -> GCN ISA / HSA code object
          (runtime: ROCm/HIP)
- Intel:  spirv64 + spir_kernel              -> SPIR-V (OpenCL flavor)
          (runtime: Level Zero / oneAPI / OpenCL)
- Vulkan: spirv-unknown-vulkan1.3-compute    -> SPIR-V (GLCompute /
          shader flavor; entry point annotated hlsl.shader="compute").
          Vulkan-flavor SPIR-V is NOT interchangeable with the Intel
          target's OpenCL flavor — Vulkan drivers only accept the former.

v0.1 emits device code for inspection/offline use; host-side kernel
launch bridges are on the roadmap.
"""
import re as _re

from llvmlite import binding as llvm

TARGETS = {
    "cuda":  dict(triple="nvptx64-nvidia-cuda",
                  datalayout="e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
                  cpu="sm_75", callconv=None),  # Turing+: CUDA 11–13
    "amd":   dict(triple="amdgcn-amd-amdhsa",
                  datalayout=("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-"
                              "p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-"
                              "v48:64-v96:128-v192:256-v256:256-v512:512-"
                              "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"),
                  cpu="gfx90a", callconv="amdgpu_kernel"),
    "intel": dict(triple="spirv64-unknown-unknown",
                  datalayout="e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
                             "v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64",
                  cpu="", callconv="spir_kernel"),
    "vulkan": dict(triple="spirv-unknown-vulkan1.3-compute",
                   datalayout="",  # logical addressing: let LLVM infer
                   cpu="", callconv=None),
}

_init_done = False


import os as _os

# AMDGPU HSA code-object version. v5 is the broadly-compatible default
# (ROCm 5.x..current, LLVM 15..latest). v6 requires LLVM>=19 toolchains and
# will fail to assemble on older ROCm. Override with the env var if needed.
AMD_CODE_OBJECT_VERSION = int(
    _os.environ.get("HANAJIT_AMD_CODE_OBJECT_VERSION", "5"))


def _init():
    global _init_done
    if not _init_done:
        try:
            llvm.initialize_all_targets()
            llvm.initialize_all_asmprinters()
        except (AttributeError, RuntimeError):
            pass
        # pin AMDGPU code-object version for portable GCN output
        try:
            llvm.set_option("hanajit",
                            "--amdhsa-code-object-version=%d"
                            % AMD_CODE_OBJECT_VERSION)
        except Exception:
            pass
        _init_done = True


def vulkan_local_size():
    """Workgroup local size as "x,y,z" (hlsl.numthreads). Vulkan fixes it
    at compile time; override with HANAJIT_VULKAN_LOCAL_SIZE=128,1,1."""
    v = _os.environ.get("HANAJIT_VULKAN_LOCAL_SIZE", "64,1,1")
    parts = [p.strip() for p in v.split(",")]
    if len(parts) != 3 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(
            "HANAJIT_VULKAN_LOCAL_SIZE must be three positive integers "
            f"'x,y,z', got {v!r}")
    return ",".join(parts)


def retarget(module, kernel_name, vendor):
    cfg = TARGETS[vendor]
    module.triple = cfg["triple"]
    module.data_layout = cfg["datalayout"]
    ir_text = str(module)
    if cfg["callconv"]:
        ir_text = ir_text.replace(f'define {_rettype(ir_text, kernel_name)}',
                                  f'define {cfg["callconv"]} '
                                  f'{_rettype(ir_text, kernel_name)}', 1)
    if vendor == "cuda":
        ir_text += (f'\n!nvvm.annotations = !{{!0}}\n'
                    f'!0 = !{{ptr @{kernel_name}, !"kernel", i32 1}}\n')
    if vendor == "vulkan":
        # SPIR-V shader-flavor entry points are mandatory-annotated; LLVM
        # aborts (report_fatal_error) on an entry without hlsl.shader.
        ir_text, n = _re.subn(
            r'(?m)^(define [^\n]*@"?%s"?\([^\n]*\))$'
            % _re.escape(kernel_name),
            r'\1 #9', ir_text, count=1)
        if n:
            ir_text += ('\nattributes #9 = { "hlsl.shader"="compute" '
                        f'"hlsl.numthreads"="{vulkan_local_size()}" }}\n')
    return ir_text


def _rettype(ir_text, name):
    # find `define <ty> @"name"` to splice the calling convention in front
    for line in ir_text.splitlines():
        if line.startswith("define") and f'@"{name}"' in line:
            return line[len("define "):].split(f' @"{name}"')[0] + f' @"{name}"'
    return ""


# environment overrides so users retarget without editing source:
#   HANAJIT_CUDA_ARCH=sm_90  HANAJIT_AMD_ARCH=gfx1100  HANAJIT_INTEL_ARCH=...
import os as _os

_ARCH_ENV = {"cuda": "HANAJIT_CUDA_ARCH", "amd": "HANAJIT_AMD_ARCH",
             "intel": "HANAJIT_INTEL_ARCH", "vulkan": "HANAJIT_VULKAN_ARCH"}


# NVPTX cannot lower these LLVM intrinsics itself — they need NVIDIA's
# libdevice bitcode. When libdevice is found, calls are rewritten to the
# __nv_* equivalents and the bitcode is linked in during emission.
_NV_LIBM = {"llvm.sin.f64": "__nv_sin", "llvm.cos.f64": "__nv_cos",
            "llvm.exp.f64": "__nv_exp", "llvm.log.f64": "__nv_log",
            "llvm.pow.f64": "__nv_pow",
            "llvm.powi.f64.i32": "__nv_powi"}


def find_libdevice():
    """Locate libdevice.10.bc: env var > CUDA toolkit > pip nvidia wheel.

    Returns a path or None. Install without a CUDA toolkit via
    `pip install nvidia-cuda-nvcc-cu12` (ships nvvm/libdevice)."""
    import glob
    p = _os.environ.get("HANAJIT_LIBDEVICE")
    if p and _os.path.isfile(p):
        return p
    candidates = []
    for root in (_os.environ.get("CUDA_PATH"),
                 _os.environ.get("CUDA_HOME")):
        if root:
            candidates.append(_os.path.join(
                root, "nvvm", "libdevice", "libdevice.10.bc"))
    candidates += glob.glob(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"
        r"\nvvm\libdevice\libdevice.10.bc")
    candidates += glob.glob("/usr/local/cuda*/nvvm/libdevice/"
                            "libdevice.10.bc")
    try:  # pip wheels: nvidia-cuda-nvcc-cu12 / -cu13
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            candidates += glob.glob(_os.path.join(
                sp, "nvidia", "cuda_nvcc", "nvvm", "libdevice",
                "libdevice.10.bc"))
    except Exception:
        pass
    for c in candidates:
        if c and _os.path.isfile(c):
            return c
    return None


def resolve_arch(vendor, cpu=None):
    """Explicit arg > env var > portable table default."""
    if cpu:
        return cpu
    env = _os.environ.get(_ARCH_ENV.get(vendor, ""))
    if env:
        return env
    return TARGETS[vendor]["cpu"]


def emit(module, kernel_name, vendor, cpu=None):
    """Best-effort device-code emission. Returns (text, native: bool).

    Architecture is resolved as: explicit `cpu=` > env var
    (HANAJIT_CUDA_ARCH / HANAJIT_AMD_ARCH / HANAJIT_INTEL_ARCH) > a portable
    default (CUDA sm_75 / AMD gfx90a). PTX and GCN are forward-compatible:
    the driver re-JITs device code for a newer GPU at load time, so the
    conservative default runs on the widest range of installed hardware."""
    cfg = TARGETS[vendor]
    arch = resolve_arch(vendor, cpu)
    ir_text = retarget(module, kernel_name, vendor)
    # All device emission runs in a throwaway subprocess: llvmlite's
    # backends do not fail gracefully on unsupported IR — the SPIR-V
    # shader backend hard-aborts on non-zero-index GEPs, and NVPTX
    # hard-crashes on libdevice-only intrinsics (llvm.sin.f64 etc.). A
    # crash there degrades to the annotated-IR fallback instead of
    # killing the interpreter. mem2reg runs in the subprocess (AMDGPU
    # cannot select generic-addrspace allocas).
    from . import isolated
    if vendor == "cuda" and any(f'@"{k}"' in ir_text for k in _NV_LIBM):
        lib = find_libdevice()
        if lib:
            nv_text = ir_text
            for k, v in _NV_LIBM.items():
                nv_text = nv_text.replace(f'@"{k}"', f"@{v}")
            text = isolated.emit_assembly(nv_text, cfg["triple"], arch,
                                          link_bitcode=lib,
                                          kernel=kernel_name)
            if text is not None:
                return text, True
        # no libdevice (or link failed): plain emission below crashes on
        # these intrinsics in the subprocess and falls back to IR
    text = isolated.emit_assembly(ir_text, cfg["triple"], arch)
    if text is not None:
        return text, True
    return ir_text, False  # annotated IR for offline llc/toolchain
