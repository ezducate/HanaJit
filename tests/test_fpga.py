"""FPGA export: the HLS project kit (IR + synthesizable C++ + testbench +
TCL) must be generated for the transpilable subset, and degrade to
IR + TCL stub outside it. No Vitis installation is required."""
import warnings

import pytest

warnings.filterwarnings("ignore")


def _scalar_kernel():
    from hanajit import jit

    @jit
    def sum_squares(n):
        total = 0.0
        for i in range(n):
            total += i * i
        return total

    sum_squares(10)
    return sum_squares


def test_full_kit_for_scalar_kernel(tmp_path):
    fn = _scalar_kernel()
    out = fn.export_fpga(str(tmp_path / "ss"))
    assert out.ll and out.tcl and out.cpp and out.tb

    cpp = open(out.cpp, encoding="utf-8").read()
    assert 'extern "C" double sum_squares(long long n)' in cpp
    assert "#pragma HLS INTERFACE s_axilite port=n" in cpp
    assert "#pragma HLS INTERFACE s_axilite port=return" in cpp
    assert "#pragma HLS PIPELINE II=1" in cpp
    assert "C truncation semantics" in cpp  # semantics caveat stated

    tcl = open(out.tcl, encoding="utf-8").read()
    assert "set_top sum_squares" in tcl
    assert "add_files ss_hls.cpp" in tcl
    assert "add_files -tb ss_tb.cpp" in tcl
    assert "csim_design" in tcl and "csynth_design" in tcl
    assert "set_part xcu250-figd2104-2L-e" in tcl   # portable default
    assert "create_clock -period 3.3" in tcl

    tb = open(out.tb, encoding="utf-8").read()
    assert "int main()" in tb and "sum_squares(" in tb


def test_pointer_args_get_m_axi_interfaces(tmp_path):
    from hanajit import jit

    @jit(signature="f64*, i64")
    def scale(x, n):
        for i in range(n):
            x[i] = x[i] * 2.0
        return 0

    out = scale.export_fpga(str(tmp_path / "scale"))
    cpp = open(out.cpp, encoding="utf-8").read()
    assert "double *x" in cpp
    assert ("#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem"
            in cpp)
    assert "#pragma HLS INTERFACE s_axilite port=x" in cpp


def test_part_and_clock_are_configurable(tmp_path):
    fn = _scalar_kernel()
    out = fn.export_fpga(str(tmp_path / "cfg"),
                         part="xcvu9p-flga2104-2L-e", clock_ns=5.0)
    tcl = open(out.tcl, encoding="utf-8").read()
    assert "set_part xcvu9p-flga2104-2L-e" in tcl
    assert "create_clock -period 5.0" in tcl


def test_math_calls_transpile_to_cmath(tmp_path):
    import math
    from hanajit import jit

    @jit
    def wave(x):
        return math.sin(x) + math.sqrt(x)

    wave(2.0)
    out = wave.export_fpga(str(tmp_path / "wave"))
    cpp = open(out.cpp, encoding="utf-8").read()
    assert "#include <cmath>" in cpp
    assert "sin((double)(x))" in cpp and "sqrt((double)(x))" in cpp


def test_array_kernel_degrades_to_ir_plus_stub(tmp_path):
    from hanajit import jit

    @jit
    def total(a):
        s = 0.0
        for i in range(len(a)):
            s += a[i]
        return s

    out = total.export_fpga(str(tmp_path / "tot"), sig="f64[]")
    assert out.cpp is None and out.tb is None
    assert out.ll and out.tcl
    tcl = open(out.tcl, encoding="utf-8").read()
    assert "LLVM front-end flow" in tcl  # stub explains the IR route
    ir_text = open(out.ll, encoding="utf-8").read()
    assert "total" in ir_text


def test_innermost_only_pipelining(tmp_path):
    from hanajit import jit

    @jit
    def nested(n):
        acc = 0.0
        for i in range(n):
            for j in range(n):
                acc += i * j
        return acc

    nested(4)
    out = nested.export_fpga(str(tmp_path / "nested"))
    cpp = open(out.cpp, encoding="utf-8").read()
    # pipelining the outer loop would force full unroll of the inner one;
    # only the innermost loop may carry the pragma
    assert cpp.count("#pragma HLS PIPELINE II=1") == 1


def test_export_return_is_namedtuple(tmp_path):
    fn = _scalar_kernel()
    out = fn.export_fpga(str(tmp_path / "nt"))
    assert out._fields == ("ll", "tcl", "cpp", "tb")
