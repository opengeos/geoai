"""Tests for keeping the managed venv's SamGeo out of the QGIS process.

QGIS imports its own NumPy at startup, so the venv's NumPy-2-era SciPy/scikit
stack breaks when imported in-process (issues #688 and #854). These cover the
gate that routes model loading into the venv subprocess instead.
"""

import sys
import types

from geoai.dialogs import samgeo
from geoai.workers import samgeo_worker


def test_use_subprocess_on_windows_without_venv(monkeypatch):
    """Windows needs the subprocess for the PyTorch DLL conflict regardless."""
    monkeypatch.setattr(samgeo.os, "name", "nt")

    assert samgeo._use_samgeo_subprocess() is True


def test_use_subprocess_on_posix_when_venv_exists(monkeypatch):
    """Regression test for issue #854: macOS/Linux must not import in-process."""
    from geoai.core import venv_manager

    monkeypatch.setattr(samgeo.os, "name", "posix")
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)

    assert samgeo._use_samgeo_subprocess() is True


def test_in_process_on_posix_without_venv(monkeypatch):
    """No managed venv means no version skew, so keep the in-process path."""
    from geoai.core import venv_manager

    monkeypatch.setattr(samgeo.os, "name", "posix")
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: False)

    assert samgeo._use_samgeo_subprocess() is False


def _fake_torch(mps_available=False, cuda_available=False):
    torch = types.ModuleType("torch")
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: mps_available
    backends.mps = mps
    torch.backends = backends
    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch.cuda = cuda
    return torch


def _install_torch(monkeypatch, torch):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.backends", torch.backends)


def test_resolve_device_prefers_mps(monkeypatch):
    """The dialog used to pick MPS in auto mode; the worker must keep doing so."""
    _install_torch(monkeypatch, _fake_torch(mps_available=True, cuda_available=True))

    assert samgeo_worker._resolve_device("auto") == "mps"
    assert samgeo_worker._resolve_device(None) == "mps"


def test_resolve_device_falls_back_to_cuda(monkeypatch):
    _install_torch(monkeypatch, _fake_torch(mps_available=False, cuda_available=True))

    assert samgeo_worker._resolve_device("auto") == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    _install_torch(monkeypatch, _fake_torch(mps_available=False, cuda_available=False))

    assert samgeo_worker._resolve_device("auto") == "cpu"


def test_resolve_device_respects_explicit_choice(monkeypatch):
    _install_torch(monkeypatch, _fake_torch(mps_available=True, cuda_available=True))

    assert samgeo_worker._resolve_device("cpu") == "cpu"
    assert samgeo_worker._resolve_device("cuda") == "cuda"


def test_resolve_device_survives_broken_torch(monkeypatch):
    """Detection must not turn an import problem into a device crash."""
    torch = _fake_torch()

    def boom():
        raise RuntimeError("mps probe failed")

    torch.backends.mps.is_available = boom
    torch.cuda.is_available = boom
    _install_torch(monkeypatch, torch)

    assert samgeo_worker._resolve_device("auto") == "cpu"
