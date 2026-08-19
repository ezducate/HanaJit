import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hanajit import UnsupportedError, jit
from hanajit.backends import executable
from hanajit.typeinfer import BOOL, F32, F64, I64


HOST_OS_NAME = os.name


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "off"),
        (False, "off"),
        (True, "optional"),
        ("off", "off"),
        ("OFF", "off"),
        ("Optional", "optional"),
        ("REQUIRED", "required"),
    ],
)
def test_normalize_cuda_modes(value, expected, monkeypatch):
    monkeypatch.setattr(executable, "sys_platform_is_macos", lambda: False)
    assert executable._normalize_mode(value) == expected


@pytest.mark.parametrize("value", ["", "sometimes", 0, 2, object()])
def test_normalize_cuda_mode_rejects_unknown_values(value, monkeypatch):
    monkeypatch.setattr(executable, "sys_platform_is_macos", lambda: False)
    with pytest.raises(ValueError, match="cuda must be"):
        executable._normalize_mode(value)


@pytest.mark.parametrize("mode", ["optional", "required", True])
def test_macos_rejects_every_cuda_mode(mode, monkeypatch):
    monkeypatch.setattr(executable, "sys_platform_is_macos", lambda: True)
    with pytest.raises(UnsupportedError, match="unavailable on macOS"):
        executable._normalize_mode(mode)


def test_macos_accepts_cpu_mode(monkeypatch):
    monkeypatch.setattr(executable, "sys_platform_is_macos", lambda: True)
    assert executable._normalize_mode("off") == "off"


@pytest.mark.parametrize("machine", ["amd64", "AMD64", "x86_64", "x64"])
def test_x86_64_machine_aliases_are_accepted(machine, monkeypatch):
    monkeypatch.setattr(executable.platform, "machine", lambda: machine)
    assert executable._x86_64()


@pytest.mark.parametrize(
    "machine", ["i386", "i686", "arm64", "aarch64", "ppc64le", ""]
)
def test_non_x86_64_machines_are_rejected(machine, monkeypatch):
    monkeypatch.setattr(executable.platform, "machine", lambda: machine)
    assert not executable._x86_64()


def test_artifact_paths_create_parent_and_use_platform_suffixes(tmp_path):
    output = tmp_path / "nested dir" / "program"
    exe, obj, source, build, ptx = executable._paths(output)

    assert exe.parent.is_dir()
    if os.name == "nt":
        assert exe.name == "program.exe"
        assert obj.name == "program.obj"
        assert build.name == "program.build.bat"
    else:
        assert exe.name == "program"
        assert obj.name == "program.o"
        assert build.name == "program.build.sh"
    assert source.name == "program.standalone.c"
    assert ptx.name == "program.ptx"


def test_artifact_paths_preserve_explicit_output_suffix(tmp_path):
    exe, obj, source, _, _ = executable._paths(tmp_path / "tool.custom")
    if os.name == "nt":
        assert exe.name == "tool.custom.exe"
        assert obj.name == "tool.custom.obj"
        assert source.name == "tool.custom.standalone.c"
    else:
        assert exe.name == "tool.custom"
        assert obj.name == "tool.o"
        assert source.name == "tool.standalone.c"


def test_ptx_byte_array_is_utf8_null_terminated():
    rendered = executable._ptx_array("A\nλ")
    expected = list("A\nλ".encode("utf-8")) + [0]
    for byte in expected:
        assert f"0x{byte:02x}" in rendered
    assert rendered.rstrip().endswith("0x00")


def test_cli_parser_generation_covers_all_scalar_types():
    text = executable._parse_lines((I64, F64, BOOL))
    assert "int64_t a0" in text and "hj_parse_i64(argv[1]" in text
    assert "double a1" in text and "hj_parse_f64(argv[2]" in text
    assert "bool a2" in text and "hj_parse_bool(argv[3]" in text
    assert "hj_bad_arg(3" in text


def test_msvc_build_commands_use_static_runtime_and_c11(tmp_path):
    src, obj, exe = (tmp_path / "x.c", tmp_path / "x.obj",
                     tmp_path / "x.exe")
    compile_cmd, link_cmd = executable._build_commands(
        ["cl.exe"], True, src, obj, exe, with_cuda=True)

    assert "/MT" in compile_cmd and "/MT" in link_cmd
    assert "/std:c11" in compile_cmd
    assert "/c" in compile_cmd
    assert "/Fo:" + str(obj) in compile_cmd
    assert "/Fe:" + str(exe) in link_cmd
    assert "-ldl" not in link_cmd


@pytest.mark.parametrize("with_cuda", [False, True])
def test_posix_build_commands_link_expected_system_libraries(
        tmp_path, monkeypatch, with_cuda):
    src, obj, exe = (tmp_path / "x.c", tmp_path / "x.o",
                     tmp_path / "x")
    monkeypatch.setattr(executable.os, "name", "posix")
    compile_cmd, link_cmd = executable._build_commands(
        ["cc"], False, src, obj, exe, with_cuda=with_cuda)

    assert compile_cmd[:4] == ["cc", "-std=c11", "-O3", "-fwrapv"]
    assert "-march=x86-64" in compile_cmd
    assert "-lm" in link_cmd
    assert ("-ldl" in link_cmd) is with_cuda


def test_windows_build_script_initializes_msvc_environment(tmp_path,
                                                           monkeypatch):
    script = tmp_path / "build.bat"
    monkeypatch.setattr(executable.os, "name", "nt")
    executable._write_build_script(
        script,
        [["C:\\Tool Path\\cl.exe", "/c", "source file.c"]],
        vcvars="C:\\VS Path\\vcvars64.bat",
    )
    text = script.read_text(encoding="utf-8")
    assert text.startswith("@echo off\ncall \"C:\\VS Path\\vcvars64.bat\" >nul")
    assert '"C:\\Tool Path\\cl.exe"' in text
    assert '"source file.c"' in text


def test_posix_build_script_is_executable_and_fail_fast(tmp_path,
                                                        monkeypatch):
    script = tmp_path / "build.sh"
    monkeypatch.setattr(executable.os, "name", "posix")
    executable._write_build_script(
        script, [["cc", "source file.c", "-o", "program"]])
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\nset -eu\n")
    assert "'source file.c'" in text
    if HOST_OS_NAME != "nt":
        assert script.stat().st_mode & 0o111


@pytest.mark.parametrize(
    "override, expected_argv, is_msvc",
    [
        ("custom-cc --flag", ["custom-cc", "--flag"], False),
        ("cl", ["cl"], True),
    ],
)
def test_compiler_override_parsing(override, expected_argv, is_msvc):
    executable._compiler_cached.cache_clear()
    try:
        info = executable._compiler_cached(override)
        assert info["argv"] == expected_argv
        assert info["msvc"] is is_msvc
    finally:
        executable._compiler_cached.cache_clear()


def test_compiler_reads_changed_environment_override(monkeypatch):
    executable._compiler_cached.cache_clear()
    monkeypatch.setenv("HANAJIT_CC", "first-cc --one")
    assert executable._compiler()["argv"] == ["first-cc", "--one"]
    monkeypatch.setenv("HANAJIT_CC", "second-cc --two")
    assert executable._compiler()["argv"] == ["second-cc", "--two"]
    executable._compiler_cached.cache_clear()


def test_posix_compiler_discovery_order(monkeypatch):
    seen = []

    def which(name):
        seen.append(name)
        return "/toolchain/gcc" if name == "gcc" else None

    executable._compiler_cached.cache_clear()
    monkeypatch.setattr(executable.os, "name", "posix")
    monkeypatch.setattr(executable.shutil, "which", which)
    info = executable._compiler_cached(None)
    assert info["argv"] == ["/toolchain/gcc"]
    assert info["msvc"] is False
    assert seen == ["clang", "gcc"]
    executable._compiler_cached.cache_clear()


def test_windows_compiler_discovery_uses_vcvars_fallback(monkeypatch):
    cl = Path("C:/VS/cl.exe")
    vcvars = Path("C:/VS/vcvars64.bat")
    env = {"PATH": "C:/VS/bin"}
    executable._compiler_cached.cache_clear()
    monkeypatch.setattr(executable.os, "name", "nt")
    monkeypatch.setattr(executable.shutil, "which", lambda _: None)
    monkeypatch.setattr(executable, "_msvc_install", lambda: (cl, vcvars))
    monkeypatch.setattr(executable, "_msvc_environment", lambda _: env)
    info = executable._compiler_cached(None)
    assert info == {
        "argv": [str(cl)], "msvc": True, "env": env,
        "vcvars": str(vcvars),
    }
    executable._compiler_cached.cache_clear()


def test_msvc_environment_parses_output_and_removes_temp_script(monkeypatch):
    invoked = {}

    def run(argv, **kwargs):
        invoked["script"] = argv[-1]
        assert Path(argv[-1]).is_file()
        return SimpleNamespace(returncode=0,
                               stdout="HJ_ONE=one\nHJ_TWO=two=three\n")

    monkeypatch.setattr(executable.subprocess, "run", run)
    env = executable._msvc_environment("C:/VS/vcvars64.bat")
    assert env["HJ_ONE"] == "one"
    assert env["HJ_TWO"] == "two=three"
    assert not Path(invoked["script"]).exists()


@pytest.mark.parametrize(
    "mode, has_cuda, required, has_fallback",
    [
        ("off", False, False, True),
        ("optional", True, False, True),
        ("required", True, True, True),
    ],
)
def test_generated_source_distinguishes_cuda_modes(
        mode, has_cuda, required, has_fallback):
    kernel = "static int64_t hj_cpu_kernel(int64_t a0) { return a0; }"
    source = executable._c_source(
        kernel, (I64,), I64, mode,
        kernel_name="cuda_kernel" if has_cuda else None,
        ptx="fake ptx" if has_cuda else None,
    )
    assert ("hj_cuda_run" in source) is has_cuda
    assert ("if (1)" in source) is required
    fallback = "result = hj_cpu_kernel(a0);"
    assert (fallback in source) is has_fallback
    assert "Python.h" not in source


def test_cuda_source_embeds_ptx_and_both_driver_names():
    source = executable._cuda_source("kernel", "PTX DATA", (I64,), I64)
    assert "hj_embedded_ptx" in source
    assert "nvcuda.dll" in source
    assert "libcuda.so.1" in source
    assert 'cuModuleGetFunction(&fn, mod, "kernel")' in source
    assert "PTX DATA" not in source  # encoded bytes, not a sidecar string


def test_cuda_boolean_result_does_not_use_c_bool_as_device_storage():
    source = executable._cuda_source("kernel", "ptx", (BOOL,), BOOL)
    assert "int64_t hj_cuda_bool_result = 0" in source
    assert "cuMemAlloc(&dout, sizeof(hj_cuda_bool_result))" in source
    assert "cuMemcpyDtoH(&hj_cuda_bool_result" in source
    assert "*out = hj_cuda_bool_result != 0" in source


def test_return_to_output_rewrites_every_return():
    fn = ast.parse(
        "def f(x):\n"
        "    if x:\n"
        "        return 10\n"
        "    return 20\n"
    ).body[0]
    transformed = executable._ReturnToOutput("result").visit(fn)
    assignments = [n for n in ast.walk(transformed)
                   if isinstance(n, ast.Assign)]
    returns = [n for n in ast.walk(transformed)
               if isinstance(n, ast.Return)]
    assert len(assignments) == 2
    assert all(isinstance(a.targets[0], ast.Subscript) for a in assignments)
    assert all(isinstance(r.value, ast.Constant) and r.value.value == 0
               for r in returns)


def test_return_to_output_rejects_bare_return():
    with pytest.raises(UnsupportedError, match="bare return"):
        executable._ReturnToOutput("result").visit(ast.Return(value=None))


def test_no_compiler_still_writes_reproducible_artifacts(tmp_path,
                                                          monkeypatch):
    @jit(signature="i64", native_dispatch=False)
    def increment(x):
        return x + 1

    monkeypatch.setattr(executable, "_x86_64", lambda: True)
    monkeypatch.setattr(executable, "_compiler", lambda: None)
    with pytest.warns(UserWarning, match="no C compiler found"):
        out = increment.export_executable(tmp_path / "nested" / "increment")
    assert out.executable is None
    assert out.object is None
    assert Path(out.source).is_file()
    assert Path(out.build).is_file()
    assert out.ptx is None
    assert "hj_cpu_kernel" in Path(out.source).read_text(encoding="utf-8")


def test_compiler_failure_returns_sources_not_stale_executable(
        tmp_path, monkeypatch):
    @jit(signature="i64", native_dispatch=False)
    def identity(x):
        return x

    output = tmp_path / ("identity.exe" if os.name == "nt" else "identity")
    output.write_bytes(b"stale")
    compiler = {"argv": ["failing-cc"], "msvc": False,
                "env": None, "vcvars": None}
    monkeypatch.setattr(executable, "_x86_64", lambda: True)
    monkeypatch.setattr(executable, "_compiler", lambda: compiler)
    monkeypatch.setattr(
        executable.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="",
                                         stderr="compile failed"),
    )
    with pytest.warns(UserWarning, match="standalone linker failed"):
        out = identity.export_executable(output)
    assert out.executable is None
    assert out.object is None
    assert Path(out.source).is_file()
    assert Path(out.build).is_file()


def test_dispatcher_forwards_codegen_options_and_default_arch(
        tmp_path, monkeypatch):
    captured = {}

    @jit(signature="f64", native_dispatch=False, fastmath=True,
         reduce_reassoc=True, gpu_arch="sm_80")
    def identity(x):
        return x

    def fake_export(*args, **kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(executable, "export_executable", fake_export)
    assert identity.export_executable(
        tmp_path / "identity", cuda="optional") == "sentinel"
    assert captured == {
        "cuda": "optional", "fastmath": True,
        "reduce_reassoc": True, "cuda_arch": "sm_80",
    }


def test_explicit_cuda_arch_overrides_dispatcher_default(tmp_path,
                                                         monkeypatch):
    captured = {}

    @jit(signature="i64", native_dispatch=False, gpu_arch="sm_70")
    def identity(x):
        return x

    def fake_export(*args, **kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(executable, "export_executable", fake_export)
    identity.export_executable(tmp_path / "identity", cuda_arch="sm_90")
    assert captured["cuda_arch"] == "sm_90"


def test_export_requires_signature_or_existing_specialization(tmp_path):
    @jit(native_dispatch=False)
    def identity(x):
        return x

    with pytest.raises(RuntimeError, match="call the function once first"):
        identity.export_executable(tmp_path / "identity")


def test_export_rejects_signature_arity_mismatch(tmp_path):
    @jit(native_dispatch=False)
    def add(a, b):
        return a + b

    with pytest.raises(UnsupportedError, match="arity mismatch"):
        add.export_executable(tmp_path / "add", sig="i64")


def test_export_rejects_non_cli_scalar_type(tmp_path, monkeypatch):
    @jit(native_dispatch=False)
    def identity(x):
        return x

    monkeypatch.setattr(executable, "_x86_64", lambda: True)
    with pytest.raises(UnsupportedError, match="i64/f64/bool only"):
        identity.export_executable(tmp_path / "identity", sig=(F32,))


def test_export_rejects_non_ascii_c_function_name(tmp_path, monkeypatch):
    @jit(native_dispatch=False)
    def café(x):
        return x

    monkeypatch.setattr(executable, "_x86_64", lambda: True)
    with pytest.raises(UnsupportedError, match="ASCII C identifiers"):
        café.export_executable(tmp_path / "unicode", sig="i64")


def test_import_statement_inside_kernel_is_rejected(tmp_path):
    @jit(native_dispatch=False)
    def imports_at_runtime(x):
        import decimal
        return x

    with pytest.raises(UnsupportedError, match="unsupported statement"):
        imports_at_runtime.export_executable(
            tmp_path / "runtime_import", sig="i64")


def test_arbitrary_module_call_is_not_silently_bundled(tmp_path):
    @jit(native_dispatch=False)
    def process_id_plus(x):
        return os.getpid() + x

    with pytest.raises(UnsupportedError):
        process_id_plus.export_executable(tmp_path / "pid", sig="i64")


def test_dynamic_import_is_not_silently_bundled(tmp_path):
    @jit(native_dispatch=False)
    def dynamic_import(x):
        return __import__("math").floor(x)

    with pytest.raises(UnsupportedError):
        dynamic_import.export_executable(tmp_path / "dynamic", sig="f64")


def test_cuda_ptx_rejects_gpu_thread_intrinsics_for_scalar_cli():
    fn = ast.parse(
        "def kernel(x):\n"
        "    return x + thread_id()\n"
    ).body[0]
    with pytest.raises(UnsupportedError, match="thread intrinsics"):
        executable._cuda_ptx(fn, (I64,), I64)
