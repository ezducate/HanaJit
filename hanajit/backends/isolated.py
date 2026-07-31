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
import sys
from llvmlite import binding as llvm
triple, arch = sys.argv[1], sys.argv[2]
llvm.initialize_all_targets()
llvm.initialize_all_asmprinters()
ir_text = sys.stdin.buffer.read().decode("utf-8")
mod = llvm.parse_assembly(ir_text)
try:  # mem2reg with the host TM; harmless to skip on old llvmlite
    host = llvm.Target.from_default_triple().create_target_machine()
    pto = llvm.create_pipeline_tuning_options(speed_level=1)
    pb = llvm.create_pass_builder(host, pto)
    pb.getModulePassManager().run(mod, pb)
except Exception:
    pass
tm = llvm.Target.from_triple(triple).create_target_machine(cpu=arch)
sys.stdout.write(tm.emit_assembly(mod))
"""


def emit_assembly(ir_text, triple, arch=""):
    """Emit target assembly in a subprocess; None if it fails or crashes."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", _SUBPROC_EMIT, triple, arch],
            input=ir_text.encode("utf-8"), capture_output=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.decode("utf-8", "replace")
    except Exception:
        pass
    return None
