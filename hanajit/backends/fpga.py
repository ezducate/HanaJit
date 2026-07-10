"""FPGA backend (experimental).

There is no direct LLVM->bitstream path; FPGA flows go through
High-Level Synthesis (HLS). Two practical routes from our IR:

1. LLVM IR -> Vitis HLS: AMD/Xilinx Vitis HLS is itself LLVM-based and
   accepts LLVM IR through its front-end flow (inject .ll into the
   hls compilation, or use the open-source Vitis HLS LLVM fork).
2. LLVM IR -> CIRCT: the LLVM CIRCT project (circt.llvm.org) lowers to
   hardware dialects (Calyx/FIRRTL) and emits Verilog.

v0.1 exports self-contained annotated IR plus a Vitis-ready C shim so
either flow can pick it up. Iqbal-note: this pairs naturally with the
existing FPGA acceleration work — the IR here is plain scalar compute,
ideal for HLS pipelining pragmas.
"""
import textwrap


def export_for_hls(module, func_name, path_prefix):
    """Write <prefix>.ll and a Vitis HLS TCL stub. Returns file paths."""
    ll_path = f"{path_prefix}.ll"
    tcl_path = f"{path_prefix}_hls.tcl"
    with open(ll_path, "w") as f:
        f.write(str(module))
    with open(tcl_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            # Vitis HLS flow stub for {func_name}
            open_project {func_name}_prj
            set_top {func_name}
            # Inject LLVM IR via the Vitis HLS LLVM front-end flow:
            #   https://github.com/Xilinx/HLS
            open_solution sol1
            set_part xcu250-figd2104-2L-e
            create_clock -period 3.3
            csynth_design
            export_design -format ip_catalog
        """))
    return ll_path, tcl_path
