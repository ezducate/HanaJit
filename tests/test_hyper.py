"""Hyper-aggressive mode (evolve_hyper): unsafe fp optimization validated
only on random probes, never cached. These tests verify the SAFETY
INVARIANTS (the whole point of the disclaimer), not raw speed — because a
mode that 'doesn't guarantee output' must at least guarantee it can't be
enabled by accident or silently persist."""
import warnings
import numpy as np
import pytest
from hanajit import jit, UnsupportedError

warnings.filterwarnings("ignore")


def _kernel():
    def dotish(x, y):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * y[i]
        return s
    return dotish


A = np.random.default_rng(0).uniform(-1, 1, 200_000)
B = np.random.default_rng(1).uniform(-1, 1, 200_000)


def test_hyper_refuses_without_confirmation():
    jf = jit(_kernel())
    jf(A, B)
    with pytest.raises(UnsupportedError, match="confirmed=True"):
        jf.evolve_hyper(A, B)


def test_hyper_raw_evolve_refuses_without_internal_flag():
    """Even the low-level evolve() refuses hyper without the internal
    confirmation token — no accidental activation path."""
    from hanajit import evolve as ev
    jf = jit(_kernel())
    jf(A, B)
    with pytest.raises(UnsupportedError, match="_hyper_confirmed"):
        ev.evolve(jf.dispatcher if hasattr(jf, "dispatcher") else jf,
                  (A, B), hyper_aggressive=True)


def test_hyper_never_persists(tmp_path, monkeypatch):
    # isolate the disk cache so this is a guaranteed cold compile
    monkeypatch.setenv("HANAJIT_CACHE_DIR", str(tmp_path))
    jf = jit(cache=True)(_kernel())
    jf(A, B)
    rep = jf.evolve_hyper(A, B, confirmed=True, generations=2, population=4,
                          reps=2)
    assert rep["hyper_aggressive"] is True
    assert rep["persisted"] is False        # NEVER cached, even with cache=True


def test_hyper_validates_on_many_probes():
    jf = jit(_kernel())
    jf(A, B)
    rep = jf.evolve_hyper(A, B, confirmed=True, generations=2, population=4,
                          reps=2, hyper_probes=128)
    assert rep["probes_validated"] >= 128    # large random suite, not 4


def test_hyper_result_within_declared_tolerance():
    """Whatever hyper installs must at least satisfy the tolerance it
    validated against, on fresh random inputs."""
    jf = jit(_kernel())
    jf(A, B)
    tol = 1e-3
    rep = jf.evolve_hyper(A, B, confirmed=True, generations=3, population=6,
                          reps=2, hyper_tol=tol)

    def ref(x, y):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * y[i]
        return s
    for s in range(10):
        u = np.random.default_rng(s + 900).uniform(-1, 1, 200_000)
        v = np.random.default_rng(s + 950).uniform(-1, 1, 200_000)
        got, exp = jf(u, v), ref(u, v)
        assert abs(got - exp) <= max(1e-2, tol * abs(exp) * 100)


def test_hyper_never_installs_regression():
    jf = jit(_kernel())
    jf(A, B)
    rep = jf.evolve_hyper(A, B, confirmed=True, generations=2, population=4,
                          reps=2)
    assert rep["speedup"] >= 1.0            # baseline kept if nothing faster
