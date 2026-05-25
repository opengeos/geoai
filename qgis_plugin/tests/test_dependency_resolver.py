import subprocess

from geoai.core import venv_manager


def test_windows_dependency_specs_include_python_312_and_triton():
    specs = venv_manager.get_qgis_dependency_specs(platform_name="win32")

    assert "python==3.12.*" in specs
    assert "triton-windows" in specs
    assert "transformers>=5.6.2" in specs


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
    commands = []

    def fake_run(cmd, capture_output, text, timeout):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Resolved 10 packages", stderr="")

    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    ok, message = venv_manager.resolve_qgis_dependencies(
        python_version="3.12",
        platform_name="win32",
        resolver="uv",
        timeout=5,
    )

    assert ok is True
    assert "Dependency resolution succeeded" in message
    assert commands
    assert commands[0][:4] == ["uv", "pip", "compile", "--python-version"]
    assert "3.12" in commands[0]
    assert "--python-platform" in commands[0]
    assert "windows" in commands[0]
