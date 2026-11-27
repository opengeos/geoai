"""Regression tests for QGIS plugin dependency-installation issues.

Covers issues #911 (QGIS-Python fallback without ssl, misreported as a
resolver conflict), #925/#977 (venv interpreter not activating, user-site
packages leaking in) and #964 (venv built for a stale Python version).
"""

import importlib.util
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from geoai.core import python_manager, venv_manager

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# Output pip printed in issue #911 when the venv Python could not import ssl.
SSL_MODULE_MISSING_OUTPUT = (
    "WARNING: pip is configured with locations that require TLS/SSL, however "
    "the ssl module in Python is not available.\n"
    "WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, "
    "status=None)) after connection broken by 'SSLError(\"Can't connect to "
    "HTTPS URL because the SSL module is not available.\")': /simple/torch/\n"
    "ERROR: Could not find a version that satisfies the requirement "
    "torch>=2.0.0 (from versions: none)\n"
    "ERROR: No matching distribution found for torch>=2.0.0"
)


def _write_pyvenv_cfg(venv_dir: Path, text: str) -> None:
    """Write a ``pyvenv.cfg`` file into a fake venv directory.

    Args:
        venv_dir: Directory standing in for a venv.
        text: File contents.
    """
    venv_dir.mkdir(parents=True, exist_ok=True)
    (venv_dir / "pyvenv.cfg").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Error classification (#911, #977)
# ---------------------------------------------------------------------------


def test_ssl_module_missing_is_not_reported_as_resolver_conflict():
    message = venv_manager._describe_install_failure(
        "torch", SSL_MODULE_MISSING_OUTPUT, ["torch>=2.0.0"]
    )

    assert "cannot load its SSL module" in message
    assert "Dependency resolution failed" not in message
    assert "ssl module in Python is not available" in message


def test_ssl_certificate_error_message_keeps_installer_output():
    output = (
        "error: Request failed after 3 retries\n"
        "  Caused by: invalid peer certificate: UnknownIssuer"
    )
    message = venv_manager._describe_install_failure(
        "deepforest", output, ["deepforest>=2.1.0"]
    )

    assert message.startswith("Failed to install deepforest: SSL certificate error")
    assert "UnknownIssuer" in message


def test_package_name_followed_by_colon_is_not_a_tls_error():
    output = "pyopenssl: no matching distribution for this platform"

    assert venv_manager._is_ssl_error(output) is False


def test_tls_error_in_the_middle_of_long_output_is_classified_first():
    output = (
        "Collecting torch\n"
        + "x" * 5000
        + "\nSSLError(SSLCertVerificationError(1, '[SSL: "
        "CERTIFICATE_VERIFY_FAILED] certificate verify failed'))\n"
        + "y" * 5000
        + "\nERROR: No matching distribution found for torch>=2.0.0"
    )
    message = venv_manager._describe_install_failure("torch", output, ["torch>=2.0.0"])

    assert "SSL certificate error" in message
    assert len(message) < 2000


def test_windows_crash_message_keeps_installer_output(monkeypatch):
    monkeypatch.setattr(venv_manager.sys, "platform", "win32")
    message = venv_manager._describe_install_failure(
        "torch", "fatal: c10.dll failed", ["torch>=2.0.0"], returncode=-1073741819
    )

    assert "process crashed (code -1073741819)" in message
    assert "c10.dll failed" in message


def test_package_names_containing_ssl_are_not_tls_errors():
    output = (
        "No solution found when resolving dependencies:\n"
        "Because pyopenssl==99 has no wheels with a matching platform tag "
        "and you require pyopenssl==99, we can conclude that your "
        "requirements are unsatisfiable."
    )

    assert venv_manager._is_ssl_error(output) is False
    message = venv_manager._describe_install_failure(
        "pyopenssl", output, ["pyopenssl==99"]
    )
    assert "Dependency resolution failed" in message


@pytest.mark.parametrize(
    "output",
    [
        "SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]'))",
        "invalid peer certificate: UnknownIssuer",
        "unable to get local issuer certificate",
        "requests.exceptions.SSLError: HTTPSConnectionPool",
        "pip is configured with locations that require TLS/SSL, however "
        "the ssl module in Python is not available.",
    ],
)
def test_real_tls_errors_are_still_detected(output):
    assert venv_manager._is_ssl_error(output) is True


def test_network_error_message_is_not_a_resolver_conflict():
    output = (
        "ConnectionResetError(10054, 'An existing connection was forcibly "
        "closed by the remote host')\n"
        "ERROR: No matching distribution found for geoai-py>=0.39.0"
    )
    message = venv_manager._describe_install_failure(
        "geoai-py", output, ["geoai-py>=0.39.0"]
    )

    assert "network error" in message
    assert "Dependency resolution failed" not in message


# ---------------------------------------------------------------------------
# Subprocess environment (#925, #977)
# ---------------------------------------------------------------------------


def test_clean_env_disables_user_site_and_strips_python_executable(monkeypatch):
    monkeypatch.setenv("PYTHONEXECUTABLE", r"C:\QGIS\bin\python3.exe")
    monkeypatch.setattr(venv_manager, "_get_qgis_proxy_settings", lambda: None)

    venv_env = venv_manager._get_clean_env_for_venv()
    standalone_env = python_manager._get_clean_env()

    for env in (venv_env, standalone_env):
        assert "PYTHONEXECUTABLE" not in env
        assert env["PYTHONNOUSERSITE"] == "1"


def test_real_venv_activates_despite_inherited_python_executable(tmp_path, monkeypatch):
    """QGIS sets PYTHONEXECUTABLE; it must not stop the venv from activating.

    With PYTHONEXECUTABLE pointing at QGIS's interpreter, a Windows venv
    launcher runs as the base interpreter: packages then go to the wrong
    site-packages and user-site packages shadow the venv (issues #925, #977).
    """
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(str(venv_dir))
    monkeypatch.setattr(venv_manager, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(venv_manager, "_get_qgis_proxy_settings", lambda: None)
    monkeypatch.setenv("PYTHONEXECUTABLE", sys.executable)

    assert venv_manager.venv_python_works(str(venv_dir)) is True

    result = subprocess.run(
        [
            venv_manager.get_venv_python_path(str(venv_dir)),
            "-c",
            "import site, sys; "
            "print(sys.prefix != sys.base_prefix, site.ENABLE_USER_SITE)",
        ],
        capture_output=True,
        text=True,
        env=venv_manager._get_clean_env_for_venv(),
        **venv_manager._get_subprocess_kwargs(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "False"]


# ---------------------------------------------------------------------------
# Stale Python versions (#964)
# ---------------------------------------------------------------------------


def test_pyvenv_cfg_version_is_read_from_uv_and_stdlib_venvs(tmp_path):
    uv_dir = tmp_path / "uv"
    _write_pyvenv_cfg(
        uv_dir,
        "home = C:\\Users\\u\\.qgis_geoai\\python_standalone\\python\n"
        "implementation = CPython\n"
        "uv = 0.10.6\n"
        "version_info = 3.9.21\n"
        "include-system-site-packages = false\n",
    )
    stdlib_dir = tmp_path / "stdlib"
    _write_pyvenv_cfg(
        stdlib_dir,
        "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.8\n",
    )

    assert venv_manager._read_pyvenv_cfg_version(str(uv_dir)) == (3, 9)
    assert venv_manager._read_pyvenv_cfg_version(str(stdlib_dir)) == (3, 12)
    assert venv_manager._read_pyvenv_cfg_version(str(tmp_path / "missing")) is None


def test_venv_with_stale_python_version_is_not_reused(tmp_path, monkeypatch):
    stale = (3, 8) if sys.version_info[:2] != (3, 8) else (3, 9)
    _write_pyvenv_cfg(tmp_path, "version_info = {}.{}.0\n".format(stale[0], stale[1]))
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("cfg mismatch must be caught without a subprocess")

    monkeypatch.setattr(venv_manager.subprocess, "run", fail_run)

    assert venv_manager.venv_python_version_matches(str(tmp_path)) is False
    assert venv_manager.venv_python_works(str(tmp_path)) is False


def test_venv_with_matching_python_version_is_accepted(tmp_path):
    _write_pyvenv_cfg(
        tmp_path,
        "version_info = {}.{}.1\n".format(*sys.version_info[:2]),
    )

    assert venv_manager.venv_python_version_matches(str(tmp_path)) is True


def test_venv_probe_rejects_interpreter_with_other_python_version(
    tmp_path, monkeypatch
):
    captured = []

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir=None: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", dict)
    real_run = subprocess.run
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)
    venv_manager.venv_python_works(str(tmp_path))

    probe = captured[0][-1]
    expected = repr(tuple(sys.version_info[:2]))
    assert expected in probe
    other_version_probe = probe.replace(expected, "(2, 7)")
    fake_venv = "import sys; sys.base_prefix = sys.prefix + '-base'; "
    assert real_run([sys.executable, "-c", fake_venv + probe]).returncode == 0
    assert (
        real_run([sys.executable, "-c", fake_venv + other_version_probe]).returncode
        == 2
    )


def test_get_venv_status_flags_stale_python_version(tmp_path, monkeypatch):
    _write_pyvenv_cfg(tmp_path, "version_info = 2.7.18\n")
    monkeypatch.setattr(venv_manager, "VENV_DIR", str(tmp_path))
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(python_manager, "standalone_python_exists", lambda: True)

    ready, message = venv_manager.get_venv_status()

    assert ready is False
    assert "different Python version" in message


def test_standalone_version_mismatch_is_detected(monkeypatch):
    monkeypatch.setattr(python_manager, "get_standalone_python_version", lambda: (3, 9))
    monkeypatch.setattr(python_manager, "get_qgis_python_version", lambda: (3, 12))
    assert python_manager.standalone_python_matches_qgis() is False

    monkeypatch.setattr(
        python_manager, "get_standalone_python_version", lambda: (3, 12)
    )
    assert python_manager.standalone_python_matches_qgis() is True

    # An interpreter that does not run is not proof of a stale version.
    monkeypatch.setattr(python_manager, "get_standalone_python_version", lambda: None)
    assert python_manager.standalone_python_matches_qgis() is True


def test_standalone_with_installed_packages_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(python_manager, "STANDALONE_DIR", str(tmp_path))
    site_packages = Path(python_manager._get_standalone_site_packages())
    site_packages.mkdir(parents=True)
    assert python_manager.standalone_has_installed_packages() is False

    (site_packages / "torch").mkdir()
    assert python_manager.standalone_has_installed_packages() is True


def test_installer_replaces_stale_standalone_and_its_venv(tmp_path, monkeypatch):
    removed = []
    cleaned = []
    exists = iter([True, False])

    monkeypatch.setattr(venv_manager, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(venv_manager, "VENV_DIR", str(tmp_path / "venv"))
    monkeypatch.setattr(venv_manager, "_log_system_info", lambda: None)
    monkeypatch.setattr(venv_manager, "_check_rosetta_warning", lambda: None)
    monkeypatch.setattr(venv_manager, "cleanup_old_venv_directories", lambda: [])
    monkeypatch.setattr(venv_manager, "_get_qgis_python", lambda: None)
    monkeypatch.setattr(venv_manager, "_get_linux_system_python", lambda: None)
    monkeypatch.setattr(venv_manager, "_cleanup_partial_venv", cleaned.append)
    monkeypatch.setattr(
        python_manager, "standalone_python_exists", lambda: next(exists)
    )
    monkeypatch.setattr(python_manager, "standalone_python_matches_qgis", lambda: False)
    monkeypatch.setattr(
        python_manager,
        "remove_standalone_python",
        lambda: removed.append(True) or (True, "removed"),
    )
    monkeypatch.setattr(
        python_manager,
        "download_python_standalone",
        lambda **_kwargs: (False, "offline"),
    )
    from geoai.core import uv_manager

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: False)
    monkeypatch.setattr(
        venv_manager, "create_venv", lambda **_kwargs: (False, "no python")
    )

    # Linux is the platform whose download failure continues to a
    # uv-managed Python; pin it so the test behaves the same on every host.
    monkeypatch.setattr(venv_manager.sys, "platform", "linux")

    ok, message = venv_manager.create_venv_and_install()

    assert removed == [True]
    assert cleaned == [str(tmp_path / "venv")]
    assert ok is False
    assert message == "no python"


def test_linux_system_python_must_match_qgis_version(monkeypatch):
    """A mismatched fallback would build a venv the version check rejects."""
    monkeypatch.setattr(venv_manager.sys, "platform", "linux")
    monkeypatch.setattr(venv_manager.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", dict)
    major, minor = sys.version_info[:2]

    def run_reporting(version):
        return lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=version, stderr=""
        )

    monkeypatch.setattr(
        venv_manager.subprocess, "run", run_reporting(f"{major} {minor + 1} 0")
    )
    assert venv_manager._get_linux_system_python() is None

    monkeypatch.setattr(
        venv_manager.subprocess, "run", run_reporting(f"{major} {minor} 4")
    )
    assert venv_manager._get_linux_system_python() == "/usr/bin/python3"


def test_managed_python_timeout_falls_back_to_qgis_python(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--managed-python" in cmd:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    from geoai.core import uv_manager

    qgis_python = str(Path(sys.prefix) / "python.exe")
    monkeypatch.setattr(venv_manager.sys, "platform", "win32")
    monkeypatch.setattr(venv_manager, "_get_system_python", lambda: qgis_python)
    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "uv")
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(venv_manager, "_cleanup_partial_venv", lambda _d: None)
    monkeypatch.setattr(
        venv_manager, "_ensure_venv_ssl", lambda *_args: (True, "ssl available")
    )
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    ok, _message = venv_manager.create_venv(str(tmp_path / "venv"))

    assert ok is True
    assert "--managed-python" in calls[0]
    assert qgis_python in calls[1]


# ---------------------------------------------------------------------------
# QGIS-Python fallback without ssl (#911)
# ---------------------------------------------------------------------------


def test_ensure_venv_ssl_registers_qgis_dll_directories(tmp_path, monkeypatch):
    dll_dir = tmp_path / "qgis" / "bin"
    dll_dir.mkdir(parents=True)
    site_packages = tmp_path / "venv" / "Lib" / "site-packages"
    probes = iter([False, True])

    monkeypatch.setattr(
        venv_manager, "_venv_can_import_ssl", lambda *_args: next(probes)
    )
    monkeypatch.setattr(venv_manager, "_qgis_dll_directories", lambda: [str(dll_dir)])
    monkeypatch.setattr(
        venv_manager,
        "get_venv_site_packages",
        lambda _venv_dir=None: str(site_packages),
    )

    ok, _message = venv_manager._ensure_venv_ssl(str(tmp_path / "venv"), {}, {})

    assert ok is True
    pth = (site_packages / "geoai_qgis_dlls.pth").read_text(encoding="utf-8")
    assert pth.startswith("import os;")
    assert repr(str(dll_dir)) in pth
    # The .pth line must be valid Python that the site module can execute.
    compile(pth, "geoai_qgis_dlls.pth", "exec")


def test_ensure_venv_ssl_reports_unrecoverable_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(venv_manager, "_venv_can_import_ssl", lambda *_args: False)
    monkeypatch.setattr(venv_manager, "_qgis_dll_directories", lambda: [])

    ok, message = venv_manager._ensure_venv_ssl(str(tmp_path), {}, {})

    assert ok is False
    assert "SSL module" in message


def test_qgis_fallback_python_detection(monkeypatch):
    monkeypatch.setattr(venv_manager.sys, "platform", "win32")
    qgis_python = str(Path(sys.prefix) / "python.exe")

    assert venv_manager._is_qgis_fallback_python(qgis_python) is True
    assert venv_manager._is_qgis_fallback_python("/elsewhere/python.exe") is False
    assert venv_manager._is_qgis_fallback_python(None) is False


# ---------------------------------------------------------------------------
# install.py: QGIS 4 profiles
# ---------------------------------------------------------------------------


def _load_install_script():
    """Import ``qgis_plugin/install.py`` as a module.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(
        "geoai_plugin_install_script", PLUGIN_ROOT / "install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_script_targets_every_installed_qgis_version(tmp_path, monkeypatch):
    install = _load_install_script()
    monkeypatch.setattr(install, "get_qgis_settings_root", lambda: tmp_path)
    plugins = Path("profiles") / "default" / "python" / "plugins"

    # Nothing installed yet: fall back to the QGIS 3 location.
    assert install.get_qgis_plugin_dirs() == [tmp_path / "QGIS3" / plugins]

    (tmp_path / "QGIS3").mkdir()
    (tmp_path / "QGIS4").mkdir()
    assert install.get_qgis_plugin_dirs() == [
        tmp_path / "QGIS4" / plugins,
        tmp_path / "QGIS3" / plugins,
    ]
    assert install.get_qgis_plugin_dirs("work") == [
        tmp_path / "QGIS4" / "profiles" / "work" / "python" / "plugins",
        tmp_path / "QGIS3" / "profiles" / "work" / "python" / "plugins",
    ]
