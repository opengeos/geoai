from geoai.core import diagnostics


def test_diagnostics_report_is_github_markdown(monkeypatch):
    monkeypatch.setenv("HOME", "/home/testuser")
    monkeypatch.setattr(
        diagnostics,
        "_get_nvidia_gpu_detection",
        lambda: {"detected": True, "details": {"name": "Test GPU"}, "error": None},
    )
    monkeypatch.setattr(
        diagnostics,
        "_get_venv_info",
        lambda: {
            "cache_dir": "/home/testuser/.qgis_geoai",
            "venv_dir": "/home/testuser/.qgis_geoai/venv_py3.12",
            "exists": True,
            "python_path": "/home/testuser/.qgis_geoai/venv_py3.12/bin/python3",
            "site_packages": (
                "/home/testuser/.qgis_geoai/venv_py3.12/lib/python3.12/site-packages"
            ),
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_collect_venv_runtime_info",
        lambda _venv_info: {
            "python_version": "3.12.8",
            "python_executable": "/home/testuser/.qgis_geoai/venv_py3.12/bin/python3",
            "python_prefix": "/home/testuser/.qgis_geoai/venv_py3.12",
            "platform": "Linux-test",
            "torch_runtime": {
                "torch_import_ok": True,
                "torch_version": "2.7.0",
                "cuda_version": "12.8",
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_devices": ["Test GPU"],
                "cudnn_version": 91002,
                "mps_available": False,
                "mps_built": False,
            },
            "packages": [
                {
                    "label": "GeoAI",
                    "dist_name": "geoai-py",
                    "module_name": "geoai",
                    "dist_version": "0.39.0",
                    "module_version": "0.39.0",
                    "module_file": (
                        "/home/testuser/.qgis_geoai/venv_py3.12/"
                        "lib/python3.12/site-packages/geoai/__init__.py"
                    ),
                    "import_ok": True,
                    "import_error": None,
                },
                {
                    "label": "Transformers",
                    "dist_name": "transformers",
                    "module_name": "transformers",
                    "dist_version": "4.57.6",
                    "module_version": "4.57.6",
                    "module_file": (
                        "/home/testuser/.qgis_geoai/venv_py3.12/"
                        "lib/python3.12/site-packages/transformers/__init__.py"
                    ),
                    "import_ok": True,
                    "import_error": None,
                },
            ],
        },
    )

    report = diagnostics.generate_diagnostics_report()

    assert report.startswith("# GeoAI QGIS Diagnostics Report")
    assert "## Managed Environment" in report
    assert "- Virtual environment path: `~/.qgis_geoai/venv_py3.12`" in report
    assert "/home/testuser" not in report
    assert "`~/.qgis_geoai/venv_py3.12/bin/python3`" in report
    assert "`~/.qgis_geoai/venv_py3.12/lib/python3.12/site-packages" in report
    assert "## CUDA / Accelerator" in report
    assert "- CUDA available: `Yes`" in report
    assert "| GeoAI | `geoai-py` | `geoai` | `0.39.0` | `OK` |" in report
    assert (
        "| Transformers | `transformers` | `transformers` | `4.57.6` | `OK` |" in report
    )


def test_diagnostics_report_renders_import_errors_in_fenced_block(monkeypatch):
    """Multi-line tracebacks with backticks must not break Markdown rendering."""
    monkeypatch.setattr(
        diagnostics,
        "_get_nvidia_gpu_detection",
        lambda: {"detected": False, "details": {}, "error": None},
    )
    monkeypatch.setattr(
        diagnostics,
        "_get_venv_info",
        lambda: {
            "cache_dir": "/tmp/geoai-cache",
            "venv_dir": "/tmp/geoai-cache/venv_py3.12",
            "exists": True,
            "python_path": "/tmp/geoai-cache/venv_py3.12/bin/python3",
            "site_packages": (
                "/tmp/geoai-cache/venv_py3.12/lib/python3.12/site-packages"
            ),
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_collect_venv_runtime_info",
        lambda _venv_info: {
            "python_version": "3.12.8",
            "python_executable": "/tmp/geoai-cache/venv_py3.12/bin/python3",
            "python_prefix": "/tmp/geoai-cache/venv_py3.12",
            "platform": "Linux-test",
            "torch_runtime": {"torch_import_ok": True, "torch_version": "2.7.0"},
            "packages": [
                {
                    "label": "Broken",
                    "dist_name": "broken-pkg",
                    "module_name": "broken_pkg",
                    "dist_version": "1.0.0",
                    "module_version": None,
                    "module_file": None,
                    "import_ok": False,
                    "import_error": (
                        "SyntaxError: invalid syntax\n" "  File '`/tmp/foo.py`', line 1"
                    ),
                }
            ],
        },
    )

    report = diagnostics.generate_diagnostics_report()

    assert "- import error:" in report
    assert "SyntaxError: invalid syntax" in report
    assert "  File '`/tmp/foo.py`', line 1" in report
    # Fenced block must use a fence longer than any backtick run in the body.
    assert "\n```\nSyntaxError" in report
    # No inline-code wrapper around the raw multi-line error.
    assert "`SyntaxError: invalid syntax" not in report


def test_diagnostics_report_handles_missing_venv(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "_get_nvidia_gpu_detection",
        lambda: {"detected": False, "details": {}, "error": None},
    )
    monkeypatch.setattr(
        diagnostics,
        "_get_venv_info",
        lambda: {
            "cache_dir": "/tmp/geoai-cache",
            "venv_dir": "/tmp/geoai-cache/venv_py3.12",
            "exists": False,
            "python_path": "/tmp/geoai-cache/venv_py3.12/bin/python3",
            "site_packages": (
                "/tmp/geoai-cache/venv_py3.12/lib/python3.12/site-packages"
            ),
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_collect_venv_runtime_info",
        lambda _venv_info: {"error": "Virtual environment does not exist."},
    )

    report = diagnostics.generate_diagnostics_report()

    assert "- Virtual environment exists: `No`" in report
    assert "- Error: `Virtual environment does not exist.`" in report
