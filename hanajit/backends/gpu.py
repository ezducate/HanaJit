"""Multi-vendor GPU backends (experimental): NVIDIA, AMD, Intel.

All three retarget the same LLVM IR; only the triple, datalayout, and
kernel calling convention differ:

- NVIDIA: nvptx64 triple + nvvm.annotations  -> PTX text
- AMD:    amdgcn-amd-amdhsa + amdgpu_kernel  -> GCN ISA / HSA code object
          (runtime: ROCm/HIP)
- Intel:  spirv64 + spir_kernel              -> SPIR-V
          (runtime: Level Zero / oneAPI / OpenCL)

v0.1 emits device code for inspection/offline use; host-side kernel
launch bridges are on the roadmap.
"""
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
             "intel": "HANAJIT_INTEL_ARCH"}


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
    _init()
    cfg = TARGETS[vendor]
    arch = resolve_arch(vendor, cpu)
    ir_text = retarget(module, kernel_name, vendor)
    try:
        target = llvm.Target.from_triple(cfg["triple"])
        tm = target.create_target_machine(cpu=arch)
        mod = llvm.parse_assembly(ir_text)
        # mem2reg & friends: AMDGPU cannot select generic-addrspace allocas,
        # and optimized IR yields cleaner PTX/GCN anyway
        from .cpu import _optimize
        _optimize(mod, tm, 3)
        return tm.emit_assembly(mod), True
    except Exception:
        return ir_text, False  # annotated IR for offline llc/toolchain
