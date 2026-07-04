import subprocess

from geoai.core import venv_manager


def test_windows_dependency_specs_include_python_312_and_triton():
    specs = venv_manager.get_qgis_dependency_specs(platform_name="win32")

    assert "python==3.12.*" in specs
    assert "triton-windows" in specs
    assert "transformers>=4.56.2" in specs


def test_macos_sam3_is_optional_without_triton_windows():
    specs = venv_manager.get_qgis_dependency_specs(platform_name="darwin")

    assert "python==3.12.*" in specs
    assert "sam3" in specs
    assert "triton-windows" not in specs
    assert venv_manager._is_optional_verify_package("sam3", "darwin") is True
    assert venv_manager._is_optional_install_package("sam3", "darwin") is True


def test_resolution_failure_diagnostic_identifies_impossible_constraint():
    output = """
    Because transformers==4.57.6 has no wheels with a matching Python requirement
    and you require transformers==4.57.6, we can conclude that your requirements
    are unsatisfiable for Python 3.12 on win_amd64.
    No solution found when resolving dependencies.
    """

    message = venv_manager.format_dependency_resolution_diagnostic(
        output,
        package_specs=["transformers==4.57.6", "sam3"],
        python_version="3.12",
        platform_name="win32",
    )

    assert "Dependency resolution failed before installation" in message
    assert "Python 3.12" in message
    assert "Windows" in message
    assert "transformers==4.57.6" in message
    assert "impossible" in message.lower() or "unsatisfiable" in message.lower()


def test_dependency_resolver_dry_run_uses_uv_compile(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Resolved 10 packages", stderr=""
        )

    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    ok, message = venv_manager.resolve_qgis_dependencies(
        python_version="3.12",
        platform_name="win32",
        resolver="uv",
        timeout=5,
    )

    assert ok is True
    assert "Dependency resolution succeeded" in message
    assert calls
    cmd = calls[0]["cmd"]
    assert cmd[:4] == ["uv", "pip", "compile", "--python-version"]
    assert "3.12" in cmd
    assert "--python-platform" in cmd
    assert "windows" in cmd
    requirements = calls[0]["input"]
    assert requirements is not None
    assert "transformers>=4.56.2" in requirements
    assert "triton-windows" in requirements


def test_dependency_resolver_adds_pytorch_index_and_strategy_for_cuda(monkeypatch):
    """CUDA resolve must mirror the real install: PyTorch extra index + strategy.

    Regression guard for issue #829: without the multi-index strategy, uv only
    considers the (stale) PyTorch index for shared packages like ``requests``,
    making geoai-py's ``datasets`` requirement unsatisfiable.
    """
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd})
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Resolved 10 packages", stderr=""
        )

    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    ok, _ = venv_manager.resolve_qgis_dependencies(
        python_version="3.12",
        platform_name="win32",
        resolver="uv",
        timeout=5,
        cuda_index="cu126",
    )

    assert ok is True
    cmd = calls[0]["cmd"]
    assert "--extra-index-url" in cmd
    assert "https://download.pytorch.org/whl/cu126" in cmd
    assert "--index-strategy" in cmd
    assert "unsafe-best-match" in cmd
    # The strategy flags must come before the trailing "-" stdin marker.
    assert cmd[-1] == "-"


def test_dependency_resolver_omits_pytorch_index_without_cuda(monkeypatch):
    """The default (CPU) resolve must not add the PyTorch index or strategy."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd})
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Resolved 10 packages", stderr=""
        )

    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    venv_manager.resolve_qgis_dependencies(
        python_version="3.12",
        platform_name="win32",
        resolver="uv",
        timeout=5,
    )

    cmd = calls[0]["cmd"]
    assert "--extra-index-url" not in cmd
    assert "--index-strategy" not in cmd
