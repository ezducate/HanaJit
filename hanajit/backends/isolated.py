"""Crash-isolated device-code emission.

Some llvmlite target backends are unreliable in-process: the SPIR-V
shader backend hard-aborts (report_fatal_error / assertion) on IR it
cannot select, and the wasm32 emitter access-violates on alloca-bearing
modules, some -O3 output shapes, and intermittently on repeated
emissions. A crash in either would take the host interpreter down with
it, so emission for those targets runs in a throwaway subprocess: a
crash there simply degrades to the caller's annotated-IR fallback.
"""
import subprocess
import sys

_SUBPROC_EMIT = """\
import os
import sys
from llvmlite import binding as llvm
triple, arch = sys.argv[1], sys.argv[2]
link_bc = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
kernel = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()
if triple.startswith("amdgcn"):
    try:  # keep the parent's HSA code-object version pin (gpu.py)
        llvm.set_option("hanajit", "--amdhsa-code-object-version=%s"
                        % os.environ.get(
                            "HANAJIT_AMD_CODE_OBJECT_VERSION", "5"))
    except Exception:
        pass
ir_text = sys.stdin.buffer.read().decode("utf-8")
mod = llvm.parse_assembly(ir_text)
tm = llvm.Target.from_triple(triple).create_target_machine(cpu=arch)
if link_bc:
    # libdevice-style flow: link the bitcode, internalize everything but
    # the kernel (so GlobalDCE strips the unused library body), and
    # optimize with the TARGET machine — its pipeline callbacks run
    # NVVMReflect, which the linked __nv_* functions require.
    with open(link_bc, "rb") as f:
        mod.link_in(llvm.parse_bitcode(f.read()))
    for fn in mod.functions:
        if not fn.is_declaration and fn.name != kernel:
            fn.linkage = "internal"
    pto = llvm.create_pipeline_tuning_options(speed_level=3)
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod, pb)
else:
    try:  # mem2reg with the host TM; harmless to skip on old llvmlite
        host = llvm.Target.from_default_triple().create_target_machine()
        pto = llvm.create_pipeline_tuning_options(speed_level=1)
        pb = llvm.create_pass_builder(host, pto)
        pb.getModulePassManager().run(mod, pb)
    except Exception:
        pass
sys.stdout.write(tm.emit_assembly(mod))
"""


def emit_assembly(ir_text, triple, arch="", link_bitcode=None, kernel=None):
    """Emit target assembly in a subprocess; None if it fails or crashes.

    With `link_bitcode` (a .bc path, e.g. NVIDIA's libdevice), the module
    is linked against it, internalized down to `kernel`, and optimized
    with the target machine before emission.

    One retry: the wasm32 emitter's access violations are partly
    nondeterministic (heap-layout dependent), so a single fresh attempt
    recovers most transient failures without masking deterministic ones.
    """
    argv = [sys.executable, "-c", _SUBPROC_EMIT, triple, arch,
            link_bitcode or "-", kernel or "-"]
    for _ in range(2):
        try:
            r = subprocess.run(argv, input=ir_text.encode("utf-8"),
                               capture_output=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.decode("utf-8", "replace")
        except Exception:
            pass
    return None
