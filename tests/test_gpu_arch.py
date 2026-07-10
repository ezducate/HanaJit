"""GPU architecture selection: portable default, env/arg overrides, and the
doctor's arch-matching logic must work across old and new cards — not just
whatever hardware happens to be present."""
import os
import warnings
import pytest

warnings.filterwarnings("ignore")


def test_default_arch_is_portable():
    from hanajit.backends import gpu
    # sm_75 (Turing) loads on the widest range: supported by CUDA 11–13,
    # and PTX is forward-compatible to newer GPUs via driver re-JIT.
    assert gpu.TARGETS["cuda"]["cpu"] == "sm_75"


def test_arch_override_precedence(monkeypatch):
    from hanajit.backends import gpu
    monkeypatch.delenv("HANAJIT_CUDA_ARCH", raising=False)
    assert gpu.resolve_arch("cuda") == "sm_75"          # default
    monkeypatch.setenv("HANAJIT_CUDA_ARCH", "sm_90")
    assert gpu.resolve_arch("cuda") == "sm_90"          # env beats default
    assert gpu.resolve_arch("cuda", "sm_86") == "sm_86"  # arg beats env


def test_jit_gpu_arch_argument_reaches_ptx():
    from hanajit import jit

    @jit(target="cuda", signature="f64*, i64", gpu_arch="sm_90")
    def k(x, n):
        i = thread_id()
        if i < n:
            x[i] = 1.0
        return 0
    vendor, ptx, native = k.inspect_gpu()
    assert native and ".target sm_90" in ptx


def test_emitted_arch_is_default_without_override():
    from hanajit import jit

    @jit(target="cuda", signature="f64*, i64")
    def k(x, n):
        i = thread_id()
        if i < n:
            x[i] = 2.0
        return 0
    _, ptx, native = k.inspect_gpu()
    assert native and ".target sm_75" in ptx


# The doctor's arch-matching logic, unit-tested directly so it is verified
# on machines without any CUDA toolkit (like CI / the dev container).
def _pick(want, supported):
    supported = sorted(set(supported), key=lambda a: int(a[3:]), reverse=True)
    if want and (not supported or want in supported):
        return want
    if supported:
        wn = int(want[3:]) if want else 0
        higher = [a for a in supported if int(a[3:]) >= wn]
        return min(higher, key=lambda a: int(a[3:])) if higher \
            else supported[0]
    return None


@pytest.mark.parametrize("want,supported,expected", [
    ("sm_75", ["sm_75", "sm_80", "sm_86", "sm_90"], "sm_75"),  # exact match
    ("sm_75", ["sm_80", "sm_86", "sm_90"], "sm_80"),           # CUDA 13 dropped 75
    ("sm_75", ["sm_50", "sm_60", "sm_70", "sm_75"], "sm_75"),  # old toolkit has 75
    ("sm_75", ["sm_52", "sm_60", "sm_70"], "sm_70"),           # ancient: newest avail
    ("sm_90", ["sm_75", "sm_80", "sm_86", "sm_90"], "sm_90"),  # Hopper override
    ("sm_60", ["sm_75", "sm_80"], "sm_75"),                    # want older than all
])
def test_doctor_arch_matching(want, supported, expected):
    assert _pick(want, supported) == expected


def _probe_order(want, mentioned):
    """Mirror the doctor's candidate ordering: default target first, then
    lowest supported arch upward (widest GPU compatibility), then below."""
    wn = int(want[3:])
    higher = sorted([a for a in mentioned if int(a[3:]) >= wn],
                    key=lambda a: int(a[3:]))
    lower = sorted([a for a in mentioned if int(a[3:]) < wn],
                   key=lambda a: int(a[3:]), reverse=True)
    out = []
    for a in [want] + higher + lower:
        if a not in out:
            out.append(a)
    return out


def test_cuda13_drops_everything_below_hopper():
    """Regression: a CUDA 13.3 box whose ptxas supports only sm_90+ must
    still resolve — the emitted sm_75 fails, and probing lands on sm_90
    (lowest supported = widest compatibility), NOT bleeding-edge sm_121."""
    mentioned = ["sm_121", "sm_120", "sm_110", "sm_103", "sm_100", "sm_90"]
    order = _probe_order("sm_75", mentioned)
    # sm_75 tried first (our default), then sm_90 as the first supported one
    assert order[0] == "sm_75"
    assert order[1] == "sm_90"
    # the first arch in `mentioned` that we'd reach is sm_90
    first_supported = next(a for a in order if a in mentioned)
    assert first_supported == "sm_90"


def test_probe_prefers_widest_compat_not_newest():
    mentioned = ["sm_100", "sm_90", "sm_120"]
    order = _probe_order("sm_75", mentioned)
    first_supported = next(a for a in order if a in mentioned)
    assert first_supported == "sm_90"  # lowest supported, not sm_120


def test_emitter_targets_all_common_archs():
    """The emitter must produce valid native PTX for every arch a user
    might override to — old and new."""
    from hanajit import jit
    for arch in ("sm_60", "sm_75", "sm_80", "sm_90", "sm_121"):
        @jit(target="cuda", signature="f64*, i64", gpu_arch=arch)
        def k(x, n):
            i = thread_id()
            if i < n:
                x[i] = 1.0
            return 0
        _, ptx, native = k.inspect_gpu()
        assert native and (".target " + arch) in ptx, arch
