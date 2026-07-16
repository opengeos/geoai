import subprocess
from pathlib import Path

from geoai.core import venv_manager


def _write_dist_metadata(site_packages, dist_name, version):
    """Write minimal dist-info metadata for quick-check tests.

    Args:
        site_packages: Temporary site-packages path.
        dist_name: Distribution name to write.
        version: Distribution version to write.
    """
    dist_info = site_packages / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Name: {dist_name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_quick_check_rejects_stale_geoai_distribution(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    for package_dir in ("torch", "torchvision", "geoai"):
        (site_packages / package_dir).mkdir()
    _write_dist_metadata(site_packages, "geoai-py", "0.9.0")

    monkeypatch.setattr(
        venv_manager,
        "get_venv_site_packages",
        lambda _venv_dir=None: str(site_packages),
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("geoai-py", ">=0.39.0")],
    )

    ready, message = venv_manager._quick_check_packages(str(tmp_path / "venv"))

    assert ready is False
    assert "geoai-py 0.9.0 does not satisfy >=0.39.0" in message


def test_quick_check_accepts_current_geoai_distribution(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    for package_dir in ("torch", "torchvision", "geoai"):
        (site_packages / package_dir).mkdir()
    _write_dist_metadata(site_packages, "geoai-py", "0.39.0")

    monkeypatch.setattr(
        venv_manager,
        "get_venv_site_packages",
        lambda _venv_dir=None: str(site_packages),
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("geoai-py", ">=0.39.0")],
    )

    ready, message = venv_manager._quick_check_packages(str(tmp_path / "venv"))

    assert ready is True
    assert message == "All packages found"


def test_version_satisfies_rejects_prerelease_for_minimum_version():
    assert venv_manager._version_satisfies("0.39.0rc1", ">=0.39.0") is False
    assert venv_manager._version_satisfies("0.39.0", ">=0.39.0") is True


def test_version_satisfies_fails_closed_for_unsupported_specifier():
    assert venv_manager._version_satisfies("0.39.0", "~=0.39.0") is False


def test_pip_ssl_flags_trust_pytorch_wheel_host():
    flags = venv_manager._get_pip_ssl_flags()

    assert "--trusted-host" in flags
    assert "download.pytorch.org" in flags


def test_insecure_package_hosts_is_immutable():
    assert isinstance(venv_manager._INSECURE_PACKAGE_HOSTS, tuple)


def test_uv_ssl_flags_use_native_tls_without_insecure_hosts():
    flags = venv_manager._get_uv_ssl_flags()

    assert "--native-tls" in flags
    assert "--allow-insecure-host" not in flags


def test_uv_insecure_host_flags_allow_pytorch_wheel_host():
    flags = venv_manager._get_uv_insecure_host_flags()

    assert "--allow-insecure-host" in flags
    assert "download.pytorch.org" in flags


def test_apply_package_host_env_preserves_existing_values():
    env = {"PIP_TRUSTED_HOST": "internal.example.com pypi.org"}

    venv_manager._apply_package_host_env(env)

    assert env["UV_NATIVE_TLS"] == "true"
    assert "internal.example.com" in env["PIP_TRUSTED_HOST"].split()
    assert env["PIP_TRUSTED_HOST"].split().count("pypi.org") == 1
    assert "files.pythonhosted.org" in env["PIP_TRUSTED_HOST"].split()
    assert "pypi.org" in env["UV_INSECURE_HOST"].split()
    assert "download.pytorch.org" in env["UV_INSECURE_HOST"].split()


def test_ssl_error_detects_uv_rustls_unknown_issuer():
    assert (
        venv_manager._is_ssl_error(
            "invalid peer certificate: UnknownIssuer when contacting pypi.org"
        )
        is True
    )


def test_uv_insecure_host_retry_only_after_ssl_error(monkeypatch):
    calls = []

    def fake_run_pip_install(**kwargs):
        calls.append(kwargs["cmd"])
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)

    result = venv_manager._retry_uv_install_with_insecure_hosts(
        result=venv_manager._PipResult(1, "", "CERTIFICATE_VERIFY_FAILED"),
        cmd=["uv", "pip", "install", "--native-tls", "torch"],
        timeout=1,
        env={},
        subprocess_kwargs={},
        label="torch",
        progress_start=0,
        progress_end=1,
    )

    assert result.returncode == 0
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--native-tls",
            "torch",
        ]
        + venv_manager._get_uv_insecure_host_flags()
    ]


def test_uv_insecure_host_retry_skips_non_ssl_errors(monkeypatch):
    calls = []

    def fake_run_pip_install(**kwargs):
        calls.append(kwargs["cmd"])
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)
    original = venv_manager._PipResult(1, "", "No matching distribution found")

    result = venv_manager._retry_uv_install_with_insecure_hosts(
        result=original,
        cmd=["uv", "pip", "install", "--native-tls", "torch"],
        timeout=1,
        env={},
        subprocess_kwargs={},
        label="torch",
        progress_start=0,
        progress_end=1,
    )

    assert result is original
    assert calls == []


def test_windows_dll_setup_code_compiles_and_registers_torch_dirs():
    code = venv_manager._get_windows_dll_setup_code()

    compile(code, "<geoai_windows_dll_setup>", "exec")

    assert "add_dll_directory" in code
    assert '"torch", "lib"' in code
    assert '"torchvision"' in code


def test_segment_geospatial_verification_bootstraps_torch_before_samgeo():
    code = venv_manager._get_verification_code("segment-geospatial")

    assert "add_dll_directory" in code
    assert "import torch; import samgeo" in code


def test_omniwatermask_verification_bootstraps_torch_before_import():
    code = venv_manager._get_verification_code("omniwatermask")

    assert "add_dll_directory" in code
    assert "import torch; import omniwatermask" in code


def test_write_constraints_file_writes_requested_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(venv_manager, "CACHE_DIR", str(tmp_path))

    path = venv_manager._write_constraints_file(
        ["torch==2.12.0+cu128", "torchvision==0.27.0+cu128"]
    )

    assert path is not None
    path_obj = Path(path)
    assert path_obj.read_text(encoding="utf-8") == (
        "torch==2.12.0+cu128\n" "torchvision==0.27.0+cu128\n"
    )

    venv_manager._remove_temp_file(path)
    assert not path_obj.exists()


def test_cuda_batch_install_constrains_existing_torch_wheels(tmp_path, monkeypatch):
    from geoai.core import uv_manager

    calls = []
    written_constraints = []
    constraints_path = str(tmp_path / "torch-constraints.txt")

    def fake_run(cmd, *args, **kwargs):
        script = cmd[-1]
        if "torch.version.cuda" in script:
            return subprocess.CompletedProcess(cmd, 0, stdout="12.8\n", stderr="")
        if "metadata.version('torch')" in script:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="2.12.0+cu128\n", stderr=""
            )
        if "metadata.version('torchvision')" in script:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="0.27.0+cu128\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_run_pip_install(**kwargs):
        calls.append(kwargs)
        return venv_manager._PipResult(0, "ok", "")

    def fake_write_constraints(constraints):
        written_constraints.extend(constraints)
        return constraints_path

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: False)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [
            ("torch", ">=2.0.0"),
            ("torchvision", ">=0.15.0"),
            ("geoai-py", ">=0.39.0"),
            ("segment-geospatial", ""),
        ],
    )
    monkeypatch.setattr(
        venv_manager,
        "detect_nvidia_gpu",
        lambda: (True, {"compute_cap": 12.0, "driver_version": "572.76"}),
    )
    monkeypatch.setattr(venv_manager, "_select_cuda_index", lambda _gpu_info: "cu128")
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)
    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)
    monkeypatch.setattr(venv_manager, "_write_constraints_file", fake_write_constraints)

    ok, message = venv_manager.install_dependencies(
        venv_dir=str(tmp_path / "venv"),
        cuda_enabled=True,
    )

    assert ok is True
    assert message == "All dependencies installed successfully"
    assert written_constraints == [
        "torch==2.12.0+cu128",
        "torchvision==0.27.0+cu128",
    ]
    batch_cmd = calls[-1]["cmd"]
    assert "--extra-index-url" in batch_cmd
    assert "https://download.pytorch.org/whl/cu128" in batch_cmd
    assert "--constraint" in batch_cmd
    assert constraints_path in batch_cmd
    assert "geoai-py>=0.39.0" in batch_cmd
    assert "segment-geospatial" in batch_cmd


def test_verify_venv_treats_macos_sam3_import_failure_as_optional(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        package_name = cmd[-1]
        calls.append(package_name)
        if package_name == "sam3":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'triton'",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(venv_manager.sys, "platform", "darwin")
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("torch", ""), ("sam3", ""), ("geoai-py", "")],
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_verification_code",
        lambda package_name: package_name,
    )
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    ready, message = venv_manager.verify_venv("/tmp/geoai-test-venv")

    assert ready is True
    assert message == "Virtual environment ready (optional packages unavailable: sam3)"
    assert calls == ["torch", "sam3", "geoai-py"]


def test_build_constraint_args_uses_relative_path_for_uv(tmp_path):
    """uv splits --constraint on whitespace, so the value must have none."""
    cache_dir = tmp_path / "CPDO USER" / ".qgis_geoai"
    cache_dir.mkdir(parents=True)
    constraints_file = str(cache_dir / "geoai_torch_constraints_ab.txt")

    args = venv_manager._build_constraint_args(
        constraints_file,
        {"cwd": str(cache_dir)},
        use_uv=True,
    )

    assert args == ["--constraint", "geoai_torch_constraints_ab.txt"]
    assert not any(c.isspace() for c in args[1])


def test_build_constraint_args_keeps_absolute_path_for_pip(tmp_path):
    """pip does not split the value, so it keeps the unambiguous path."""
    cache_dir = tmp_path / "Louis Roy" / ".qgis_geoai"
    cache_dir.mkdir(parents=True)
    constraints_file = str(cache_dir / "constraints.txt")

    args = venv_manager._build_constraint_args(
        constraints_file,
        {"cwd": str(cache_dir)},
        use_uv=False,
    )

    assert args == ["--constraint", constraints_file]


def test_build_constraint_args_drops_constraint_when_path_has_space(tmp_path):
    """A value uv would mis-split is worse than no constraint at all."""
    constraints_file = str(tmp_path / "Louis Roy" / "constraints.txt")

    args = venv_manager._build_constraint_args(
        constraints_file,
        {"cwd": str(tmp_path / "elsewhere dir")},
        use_uv=True,
    )

    assert args == []


def test_cuda_batch_install_constraint_arg_is_whitespace_free_for_uv(
    tmp_path, monkeypatch
):
    """Regression test for issue #853: home directories containing a space."""
    from geoai.core import uv_manager

    calls = []
    cache_dir = tmp_path / "Louis Roy Byaruhanga" / ".qgis_geoai"
    cache_dir.mkdir(parents=True)
    constraints_path = str(cache_dir / "geoai_torch_constraints_xy.txt")

    def fake_run(cmd, *args, **kwargs):
        script = cmd[-1]
        if "torch.version.cuda" in script:
            return subprocess.CompletedProcess(cmd, 0, stdout="12.8\n", stderr="")
        if "metadata.version('torch')" in script:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="2.12.0+cu128\n", stderr=""
            )
        if "metadata.version('torchvision')" in script:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="0.27.0+cu128\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_run_pip_install(**kwargs):
        calls.append(kwargs)
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "uv")
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(
        venv_manager, "_get_subprocess_kwargs", lambda: {"cwd": str(cache_dir)}
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("torch", ">=2.0.0"), ("geoai-py", ">=0.39.0")],
    )
    monkeypatch.setattr(
        venv_manager,
        "detect_nvidia_gpu",
        lambda: (True, {"compute_cap": 12.0, "driver_version": "572.76"}),
    )
    monkeypatch.setattr(venv_manager, "_select_cuda_index", lambda _gpu_info: "cu128")
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)
    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)
    monkeypatch.setattr(
        venv_manager, "_write_constraints_file", lambda _c: constraints_path
    )

    ok, _message = venv_manager.install_dependencies(
        venv_dir=str(tmp_path / "venv"),
        cuda_enabled=True,
    )

    assert ok is True
    batch_cmd = calls[-1]["cmd"]
    assert "--constraint" in batch_cmd
    value = batch_cmd[batch_cmd.index("--constraint") + 1]
    assert not any(c.isspace() for c in value)
    assert value == "geoai_torch_constraints_xy.txt"


def test_truncate_error_keeps_head_and_tail():
    """uv nests the real cause at the end, so a head-only slice loses it."""
    output = "headline error\n" + ("x" * 5000) + "\nCaused by: exit status 1"

    truncated = venv_manager._truncate_error(output, limit=200)

    assert len(truncated) <= 200
    assert truncated.startswith("headline error")
    assert truncated.endswith("Caused by: exit status 1")


def test_truncate_error_returns_short_output_unchanged():
    assert venv_manager._truncate_error("  boom  ") == "boom"


def test_venv_python_works_false_when_interpreter_fails(tmp_path, monkeypatch):
    """Regression test for issue #850: an unrunnable interpreter is not healthy."""
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir=None: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stdout="", stderr="boom"),
    )

    assert venv_manager.venv_python_works(str(tmp_path)) is False


def test_venv_python_works_false_when_interpreter_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: False)

    assert venv_manager.venv_python_works(str(tmp_path)) is False


def test_venv_python_works_true_when_interpreter_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir=None: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""),
    )

    assert venv_manager.venv_python_works(str(tmp_path)) is True


def test_venv_python_works_true_on_timeout(tmp_path, monkeypatch):
    """A slow launch must not trigger deletion of a multi-GB healthy venv."""

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=60)

    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir=None: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    assert venv_manager.venv_python_works(str(tmp_path)) is True


def test_venv_python_works_false_when_interpreter_cannot_launch(tmp_path, monkeypatch):
    """OSError is how a missing python3XX.dll surfaces (issue #850)."""

    def fake_run(*_args, **_kwargs):
        raise OSError("dll load failed")

    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir=None: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    assert venv_manager.venv_python_works(str(tmp_path)) is False


def test_single_package_install_failure_message_is_bounded(tmp_path, monkeypatch):
    """Raw installer output must not reach the UI dialog unbounded."""
    from geoai.core import uv_manager

    huge_stderr = "error: could not build wheel\n" + ("y" * 50000) + "\nCaused by: boom"

    def fake_run_pip_install(**_kwargs):
        return venv_manager._PipResult(1, "", huge_stderr)

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: False)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager, "_get_required_packages", lambda: [("torch", ">=2.0.0")]
    )
    monkeypatch.setattr(
        venv_manager,
        "detect_nvidia_gpu",
        lambda: (True, {"compute_cap": 3.0, "driver_version": "390.00"}),
    )
    # Driver too old: torch installs as a plain package, so a failure returns
    # the raw stderr directly instead of going through the CPU fallback.
    monkeypatch.setattr(venv_manager, "_select_cuda_index", lambda _gpu_info: None)
    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)

    ok, message = venv_manager.install_dependencies(
        venv_dir=str(tmp_path / "venv"),
        cuda_enabled=True,
    )

    assert ok is False
    assert len(message) < 2000
    assert "could not build wheel" in message
    assert message.endswith("Caused by: boom")


def test_cpu_fallback_failure_logs_full_output(tmp_path, monkeypatch):
    """The CPU fallback is issue #850's path, so its raw output must be logged."""
    from geoai.core import uv_manager

    logged = []
    huge_stderr = "error: no matching distribution\n" + ("z" * 40000) + "\nCaused by: 1"

    def fake_run_pip_install(**kwargs):
        # CUDA attempt fails, then the CPU fallback fails too.
        return venv_manager._PipResult(1, "", huge_stderr)

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: False)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _venv_dir=None: True)
    monkeypatch.setattr(
        venv_manager, "get_venv_python_path", lambda _venv_dir: "python"
    )
    monkeypatch.setattr(venv_manager, "_get_clean_env_for_venv", lambda: {})
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        venv_manager, "_get_required_packages", lambda: [("torch", ">=2.0.0")]
    )
    monkeypatch.setattr(
        venv_manager,
        "detect_nvidia_gpu",
        lambda: (True, {"compute_cap": 12.0, "driver_version": "572.76"}),
    )
    monkeypatch.setattr(venv_manager, "_select_cuda_index", lambda _gpu_info: "cu128")
    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)
    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(venv_manager, "_log", lambda msg, *a, **k: logged.append(msg))

    ok, message = venv_manager.install_dependencies(
        venv_dir=str(tmp_path / "venv"),
        cuda_enabled=True,
    )

    assert ok is False
    # The user-facing message stays bounded...
    assert len(message) < 2000
    assert "CUDA and CPU install both failed for torch" in message
    # ...but the log keeps the CPU fallback output in full for diagnosis.
    fallback_logs = [m for m in logged if "(CPU fallback)" in m]
    assert any(huge_stderr in m for m in fallback_logs), "full CPU output not logged"
