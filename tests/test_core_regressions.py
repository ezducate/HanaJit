import os
from pathlib import Path

import numpy as np
import pytest

from hanajit import UnsupportedError, jit, pmap
from hanajit import cache as disk_cache
from hanajit.backends import cpu
from hanajit.decorator import (
    _abstract_types,
    _arr_abi,
    _array_abstract,
    _hybrid_key,
    _make_array_caller,
    _parse_signature,
)
from hanajit.typeinfer import (
    AF64,
    AI64,
    BOOL,
    F32,
    F64,
    I64,
    PF64,
    PI64,
    arr_base,
    arr_contig,
    arr_nd,
    arr_ty,
    gpu_intrinsic_axis,
    lazy_of,
    seq_elem,
    unify,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("i64", (I64,)),
        ("int, float, bool", (I64, F64, BOOL)),
        ("f64*, i64*", (PF64, PI64)),
        ("f64[], i64[:]", (AF64, AI64)),
        (" f64 , i64 ", (F64, I64)),
    ],
)
def test_signature_parser_aliases_and_whitespace(text, expected):
    assert _parse_signature(text) == expected


@pytest.mark.parametrize("text", ["", "float32", "object", "i64,wat"])
def test_signature_parser_rejects_unknown_types(text):
    with pytest.raises(UnsupportedError, match="unknown type"):
        _parse_signature(text)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.arange(8, dtype=np.float64), "f64[1c]"),
        (np.arange(8, dtype=np.float64)[::2], "f64[1s]"),
        (np.arange(8, dtype=np.float64)[::-1], "f64[1s]"),
        (np.arange(12, dtype=np.int64).reshape(3, 4), "i64[2c]"),
        (np.arange(12, dtype=np.int64).reshape(3, 4).T, "i64[2s]"),
        (np.arange(8, dtype=np.float32), "f32[1c]"),
    ],
)
def test_array_layout_classification(array, expected):
    assert _array_abstract(array) == expected


@pytest.mark.parametrize(
    "array, message",
    [
        (np.arange(4, dtype=np.int32), "unsupported array dtype"),
        (np.zeros((2, 2, 2), dtype=np.float64), "unsupported array layout"),
    ],
)
def test_array_layout_rejections(array, message):
    with pytest.raises(UnsupportedError, match=message):
        _array_abstract(array)


def test_non_numpy_array_like_object_is_rejected():
    with pytest.raises(UnsupportedError, match="unsupported argument type"):
        _array_abstract([1.0, 2.0])


def test_array_abi_for_all_layout_kinds():
    a1 = np.arange(10, dtype=np.float64)
    s1 = a1[::-2]
    a2 = np.arange(12, dtype=np.int64).reshape(3, 4)
    s2 = a2.T

    assert _arr_abi(a1, "f64[1c]") == [a1.ctypes.data, 10]
    assert _arr_abi(s1, "f64[1s]") == [s1.ctypes.data, 5, -2]
    assert _arr_abi(a2, "i64[2c]") == [a2.ctypes.data, 3, 4]
    assert _arr_abi(s2, "i64[2s]") == [
        s2.ctypes.data, 4, 3, 1, 4,
    ]


def test_hybrid_key_distinguishes_dtype_dimension_and_layout():
    c1 = np.arange(8, dtype=np.float64)
    s1 = c1[::2]
    c2 = np.arange(8, dtype=np.float64).reshape(2, 4)
    i1 = np.arange(8, dtype=np.int64)
    keys = {
        _hybrid_key((c1, 1)),
        _hybrid_key((s1, 1)),
        _hybrid_key((c2, 1)),
        _hybrid_key((i1, 1)),
    }
    assert len(keys) == 4


def test_array_caller_expands_array_metadata_and_pointer_arguments():
    seen = []

    def native(*args):
        seen.append(args)
        return 77

    base = np.arange(12, dtype=np.float64)
    view = base[::-3]
    call = _make_array_caller(native, ("f64[1s]", I64))
    assert call(view, 9) == 77
    assert seen[-1] == (view.ctypes.data, len(view), -3, 9)

    pointer_call = _make_array_caller(native, (PF64,))
    pointer_call(base)
    assert seen[-1] == (base.ctypes.data,)


def test_abstract_types_combines_scalars_and_arrays():
    a = np.arange(5, dtype=np.float64)[::2]
    assert _abstract_types((1, 2.0, True, a)) == (
        I64, F64, BOOL, "f64[1s]",
    )


@pytest.mark.parametrize(
    "base, ndim, contiguous",
    [("f64", 1, True), ("i64", 1, False), ("f32", 2, True)],
)
def test_array_type_helpers_roundtrip(base, ndim, contiguous):
    ty = arr_ty(base, ndim, contiguous)
    assert arr_base(ty) == base
    assert arr_nd(ty) == ndim
    assert arr_contig(ty) is contiguous


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (I64, I64, I64),
        (BOOL, I64, I64),
        (I64, F32, F32),
        (F32, F64, F64),
        (BOOL, F64, F64),
    ],
)
def test_numeric_type_unification(left, right, expected):
    assert unify(left, right) == expected
    assert unify(right, left) == expected


def test_incompatible_type_unification_rejected():
    with pytest.raises(UnsupportedError, match="cannot unify"):
        unify(PF64, I64)


@pytest.mark.parametrize(
    "ty, expected",
    [("f64[1c]", F64), ("i64[1s]", I64), ("f32[2c]", F32),
     ("f64[2s]", None), (I64, None)],
)
def test_sequence_element_type_contract(ty, expected):
    assert seq_elem(ty) == expected
    if expected is not None:
        assert lazy_of(expected) == "~" + expected


@pytest.mark.parametrize(
    "name, expected",
    [("thread_id", ("thread_id", 0)),
     ("thread_id_y", ("thread_id", 1)),
     ("block_id_z", ("block_id", 2)),
     ("block_dim", ("block_dim", 0))],
)
def test_gpu_intrinsic_axis_mapping(name, expected):
    assert gpu_intrinsic_axis(name) == expected


def test_keyword_call_bypasses_jit_without_poisoning_positional_dispatch():
    @jit(native_dispatch=False)
    def add(a, b=2):
        return a + b

    assert add(a=3) == 5
    assert not add.cache
    assert add.gave_up is False
    assert add(3, 4) == 7
    assert (I64, I64) in add.cache


def test_specialize_rejects_unsupported_python_type():
    @jit(native_dispatch=False)
    def identity(x):
        return x

    with pytest.raises(UnsupportedError, match="unsupported type"):
        identity.specialize(str)


def test_introspection_before_compile_has_consistent_runtime_error():
    @jit(native_dispatch=False)
    def increment(x):
        return x + 1

    with pytest.raises(RuntimeError, match="call the function once first"):
        increment.inspect_llvm()
    with pytest.raises(RuntimeError, match="call the function once first"):
        increment.inspect_asm()


def test_introspection_selects_each_compiled_specialization():
    @jit(native_dispatch=False)
    def add_one(x):
        return x + 1

    assert add_one(2) == 3
    assert add_one(2.5) == 3.5
    i_ir = add_one.inspect_llvm((I64,))
    f_ir = add_one.inspect_llvm((F64,))
    assert "define i64" in i_ir
    assert "define double" in f_ir
    assert add_one.inspect_asm((I64,)).strip()


def test_unicode_function_name_compiles_and_reports_arity():
    @jit(signature="i64", native_dispatch=False)
    def café(x):
        return x + 1

    assert café(4) == 5
    with pytest.warns(UserWarning, match="falling back to CPython"):
        with pytest.raises(TypeError, match="café.*takes 1 positional"):
            café(4, 5)


def test_unicode_recursive_function_uses_ascii_internal_symbol():
    @jit(signature="i64", native_dispatch=False)
    def naïve_fib(n):
        if n < 2:
            return n
        return naïve_fib(n - 1) + naïve_fib(n - 2)

    assert naïve_fib(12) == 144
    assert (I64,) in naïve_fib.cache


def test_unknown_target_raises_when_fallback_disabled():
    @jit(target="mystery", fallback=False, native_dispatch=False)
    def identity(x):
        return x

    with pytest.raises(UnsupportedError, match="unknown target"):
        identity(1)


def test_pmap_preserves_input_order():
    def work(delay_value):
        value, loops = delay_value
        for _ in range(loops):
            pass
        return value

    values = [(i, 1000 * (8 - i)) for i in range(8)]
    assert pmap(work, [(v,) for v in values], workers=4) == list(range(8))


def test_pmap_propagates_worker_exception():
    def work(value):
        if value == 3:
            raise LookupError("worker failed")
        return value

    with pytest.raises(LookupError, match="worker failed"):
        pmap(work, [(i,) for i in range(6)], workers=3)


def test_cpu_prototype_expands_array_metadata():
    proto = cpu._proto(F64, ("f64[1c]", "i64[2s]", I64))
    # ptr+len, ptr+rows+cols+two strides, scalar = 8 ABI arguments
    assert len(proto._argtypes_) == 8
    assert proto._restype_ is not None


def test_cache_key_is_stable_and_sensitive_to_inputs():
    opts1 = {"fastmath": False, "cpu": "x86"}
    opts2 = {"cpu": "x86", "fastmath": False}
    key = disk_cache.make_key("source", (I64,), opts1)
    assert key == disk_cache.make_key("source", (I64,), opts2)
    assert len(key) == 32
    assert key != disk_cache.make_key("source2", (I64,), opts1)
    assert key != disk_cache.make_key("source", (F64,), opts1)
    assert key != disk_cache.make_key(
        "source", (I64,), {"fastmath": True, "cpu": "x86"})


def test_cache_roundtrip_corruption_and_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("HANAJIT_CACHE_DIR", str(tmp_path / "cache"))
    key = "abc123"
    disk_cache.save(key, b"object-bytes", {"ret": I64})
    assert disk_cache.load(key) == (b"object-bytes", {"ret": I64})

    (disk_cache.cache_dir() / f"{key}.json").write_text("not json")
    assert disk_cache.load(key) is None
    disk_cache.clear()
    assert not disk_cache.cache_dir().exists()


def test_cache_save_is_best_effort(monkeypatch):
    class BrokenDirectory:
        def mkdir(self, **kwargs):
            raise OSError("read only")

    monkeypatch.setattr(disk_cache, "cache_dir", BrokenDirectory)
    disk_cache.save("key", b"object", {"ret": I64})  # must not raise
