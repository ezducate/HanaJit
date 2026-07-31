"""WebAssembly export: retargeted IR, assembly emission, loader and build
script generation. Everything here must pass without clang installed —
linking is opportunistic and `.wasm` being None is a valid outcome."""
import os
import warnings

import pytest

warnings.filterwarnings("ignore")


def _kernel():
    from hanajit import jit

    @jit
    def sum_squares(n):
        total = 0.0
        for i in range(n):
            total += i * i
        return total

    sum_squares(10)  # compile the (i64,) specialization
    return sum_squares


def test_export_writes_ir_loader_and_build_script(tmp_path):
    fn = _kernel()
    out = fn.export_wasm(str(tmp_path / "ss"))
    ir_text = open(out.ll, encoding="utf-8").read()
    assert 'target triple = "wasm32-unknown-unknown"' in ir_text
    assert "sum_squares" in ir_text
    assert "PyFloat" not in ir_text  # no fastcall wrapper leakage

    loader = open(out.mjs, encoding="utf-8").read()
    assert "ss.wasm" in loader
    assert "Math.sqrt" in loader          # env imports for libm calls
    assert "WebAssembly.instantiate" in loader
    assert "signature: sum_squares(i64) -> f64" in loader

    build = open(out.build, encoding="utf-8").read()
    assert "--target=wasm32-unknown-unknown" in build
    assert "-Wl,--export=sum_squares" in build
    assert "-Wl,--no-entry" in build


def test_native_assembly_when_wasm_target_available(tmp_path):
    fn = _kernel()
    out = fn.export_wasm(str(tmp_path / "ss"))
    text, native = fn.inspect_wasm()
    assert text
    if native:
        assert ".functype" in text and "sum_squares" in text
        assert out.s is not None
        assert ".functype" in open(out.s, encoding="utf-8").read()
    else:
        # fallback contract: retargeted IR instead of assembly
        assert "wasm32-unknown-unknown" in text
        assert out.s is None


def test_wasm64_retarget(tmp_path):
    fn = _kernel()
    out = fn.export_wasm(str(tmp_path / "ss64"), bits=64)
    ir_text = open(out.ll, encoding="utf-8").read()
    assert 'target triple = "wasm64-unknown-unknown"' in ir_text


def test_export_with_signature_string_needs_no_call(tmp_path):
    from hanajit import jit

    @jit
    def add(a, b):
        return a + b

    # never called: the signature string drives the specialization
    out = add.export_wasm(str(tmp_path / "add"), sig="f64, f64")
    ir_text = open(out.ll, encoding="utf-8").read()
    assert "double" in ir_text and "add" in ir_text


def test_export_does_not_disturb_cpu_module():
    fn = _kernel()
    sig = next(iter(fn.modules))
    before = str(fn.modules[sig])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fn.export_wasm(os.path.join(d, "x"))
    assert str(fn.modules[sig]) == before  # stored module untouched
    assert fn(10) == sum(i * i for i in range(10))  # still callable


def test_invalid_bits_rejected(tmp_path):
    fn = _kernel()
    with pytest.raises(ValueError):
        fn.export_wasm(str(tmp_path / "bad"), bits=16)


def test_uncompiled_function_without_sig_raises(tmp_path):
    from hanajit import jit

    @jit
    def never_called(x):
        return x + 1

    with pytest.raises(RuntimeError):
        never_called.export_wasm(str(tmp_path / "nc"))
