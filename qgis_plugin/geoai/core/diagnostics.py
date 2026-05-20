"""Diagnostics report generation for the GeoAI QGIS plugin."""

import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .venv_manager import (
    CACHE_DIR,
    VENV_DIR,
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    detect_nvidia_gpu,
    get_venv_python_path,
    get_venv_site_packages,
    venv_exists,
)

PACKAGE_SPECS = [
    ("geoai-py", "geoai", "GeoAI"),
    ("torch", "torch", "PyTorch"),
    ("torchvision", "torchvision", "TorchVision"),
    ("sam3", "sam3", "SAM3"),
    ("transformers", "transformers", "Transformers"),
    ("segment-geospatial", "samgeo", "Segment Geospatial"),
    ("segmentation-models-pytorch", "segmentation_models_pytorch", "SMP"),
    ("deepforest", "deepforest", "DeepForest"),
    ("omniwatermask", "omniwatermask", "OmniWaterMask"),
    ("rasterio", "rasterio", "Rasterio"),
    ("geopandas", "geopandas", "GeoPandas"),
    ("numpy", "numpy", "NumPy"),
]

_WINDOWS_CRASH_CODES = {
    3221225477: "Windows access violation (0xC0000005)",
    -1073741819: "Windows access violation (0xC0000005)",
    3221225725: "Windows stack overflow (0xC00000FD)",
    -1073741571: "Windows stack overflow (0xC00000FD)",
    3221225781: "Windows DLL not found (0xC0000135)",
    -1073741515: "Windows DLL not found (0xC0000135)",
}


def generate_diagnostics_report() -> str:
    """Generate a Markdown diagnostics report for support and bug reports."""
    plugin_info = _get_plugin_info()
    qgis_info = _get_qgis_info()
    system_info = _get_system_info()
    gpu_detect_info = _get_nvidia_gpu_detection()
    venv_info = _get_venv_info()
    venv_runtime = _collect_venv_runtime_info(venv_info)

    lines = [
        "# GeoAI QGIS Diagnostics Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Plugin",
        "",
        f"- GeoAI plugin version: `{_value(plugin_info.get('version'))}`",
        f"- Plugin path: `{_value(plugin_info.get('path'))}`",
        "",
        "## QGIS Process",
        "",
        f"- QGIS version: `{_value(qgis_info.get('qgis_version'))}`",
        f"- Python version: `{_value(qgis_info.get('python_version'))}`",
        f"- Python executable: `{_value(qgis_info.get('python_executable'))}`",
        f"- Python prefix: `{_value(qgis_info.get('python_prefix'))}`",
        "",
        "## Operating System",
        "",
        f"- OS: `{_value(system_info.get('platform'))}`",
        f"- System: `{_value(system_info.get('system'))}`",
        f"- Release: `{_value(system_info.get('release'))}`",
        f"- Machine: `{_value(system_info.get('machine'))}`",
        f"- Processor: `{_value(system_info.get('processor'))}`",
        "",
        "## Managed Environment",
        "",
        f"- Cache directory: `{_value(venv_info['cache_dir'])}`",
        f"- Virtual environment path: `{_value(venv_info['venv_dir'])}`",
        f"- Virtual environment exists: `{_yes_no(venv_info['exists'])}`",
        f"- Virtual environment Python: `{_value(venv_info['python_path'])}`",
        f"- Virtual environment site-packages: `{_value(venv_info['site_packages'])}`",
        f"- GEOAI_CACHE_DIR: `{_value(os.environ.get('GEOAI_CACHE_DIR'), '<unset>')}`",
        f"- GEOAI_VENV_DIR: `{_value(os.environ.get('GEOAI_VENV_DIR'), '<unset>')}`",
        "",
        "## GPU Detection",
        "",
        "- NVIDIA GPU detected by `nvidia-smi`: "
        f"`{_yes_no(gpu_detect_info['detected'])}`",
    ]

    if gpu_detect_info["details"]:
        for key, value in gpu_detect_info["details"].items():
            lines.append(f"- NVIDIA {key}: `{_value(value)}`")
    if gpu_detect_info["error"]:
        lines.append(f"- NVIDIA detection error: `{_value(gpu_detect_info['error'])}`")

    lines.extend(["", "## Venv Runtime", ""])
    if venv_runtime.get("error"):
        lines.append(f"- Error: `{_value(venv_runtime['error'])}`")
    else:
        lines.extend(_format_venv_runtime(venv_runtime))

    return "\n".join(lines).rstrip() + "\n"


def _get_plugin_info() -> Dict[str, str]:
    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    metadata_path = os.path.join(plugin_dir, "metadata.txt")
    version = "Unknown"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    version = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"path": plugin_dir, "version": version}


def _get_qgis_info() -> Dict[str, str]:
    try:
        from qgis.core import Qgis

        qgis_version = str(getattr(Qgis, "QGIS_VERSION", "Unknown"))
    except Exception:
        qgis_version = "Unknown"

    return {
        "qgis_version": qgis_version,
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
    }


def _get_system_info() -> Dict[str, str]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
    }


def _get_nvidia_gpu_detection() -> Dict[str, Any]:
    try:
        detected, details = detect_nvidia_gpu()
        return {"detected": bool(detected), "details": details, "error": None}
    except Exception as exc:
        return {"detected": False, "details": {}, "error": str(exc)}


def _get_venv_info() -> Dict[str, Any]:
    return {
        "cache_dir": CACHE_DIR,
        "venv_dir": VENV_DIR,
        "exists": venv_exists(),
        "python_path": get_venv_python_path(),
        "site_packages": get_venv_site_packages(),
    }


def _collect_venv_runtime_info(venv_info: Dict[str, Any]) -> Dict[str, Any]:
    if not venv_info["exists"]:
        return {"error": "Virtual environment does not exist."}

    python_path = venv_info["python_path"]
    if not os.path.exists(python_path):
        return {"error": f"Virtual environment Python not found: {python_path}"}

    env = _get_clean_env_for_venv()
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    subprocess_kwargs = _get_subprocess_kwargs()

    try:
        result = subprocess.run(
            [python_path, "-c", _venv_probe_script()],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            **subprocess_kwargs,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Timed out while collecting venv runtime diagnostics."}
    except Exception as exc:
        return {"error": f"Failed to run venv diagnostics: {exc}"}

    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        return {
            "error": "Venv diagnostics subprocess failed with code {}: {}".format(
                result.returncode, output[:2000]
            )
        }

    try:
        runtime = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "error": "Could not parse venv diagnostics output: {}\n{}".format(
                exc, result.stdout[:2000]
            )
        }

    runtime["packages"] = _collect_package_import_info(
        runtime.get("packages") or [], python_path, env, subprocess_kwargs
    )
    runtime["torch_runtime"] = _collect_torch_runtime_info(
        python_path, env, subprocess_kwargs
    )
    return runtime


def _venv_probe_script() -> str:
    specs_json = json.dumps(PACKAGE_SPECS)
    return textwrap.dedent(f"""
        import importlib.util
        import importlib.metadata as metadata
        import json
        import platform
        import sys

        PACKAGE_SPECS = {specs_json}

        def dist_version(dist_name):
            try:
                return metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                return None
            except Exception as exc:
                return "error: " + str(exc)

        def module_origin(module_name):
            try:
                spec = importlib.util.find_spec(module_name)
            except Exception:
                return None
            if spec is None:
                return None
            return spec.origin

        def package_metadata(dist_name, module_name, label):
            info = {{
                "label": label,
                "dist_name": dist_name,
                "module_name": module_name,
                "dist_version": dist_version(dist_name),
                "module_version": None,
                "module_file": module_origin(module_name),
                "import_ok": None,
                "import_error": None,
            }}
            return info

        payload = {{
            "python_version": sys.version.replace("\\n", " "),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "platform": platform.platform(),
            "packages": [
                package_metadata(dist_name, module_name, label)
                for dist_name, module_name, label in PACKAGE_SPECS
            ],
            "torch_runtime": {{}},
        }}
        print(json.dumps(payload, sort_keys=True))
        """)


def _collect_package_import_info(
    packages: List[Dict[str, Any]],
    python_path: str,
    env: dict,
    subprocess_kwargs: dict,
) -> List[Dict[str, Any]]:
    """Probe package imports one process at a time so native crashes are isolated."""
    updated = []
    for package in packages:
        package = dict(package)
        module_name = package.get("module_name")
        if not module_name:
            package["import_ok"] = False
            package["import_error"] = "No module name configured for import probe."
            updated.append(package)
            continue

        result = _run_probe(
            python_path,
            _package_import_probe_script(module_name),
            env,
            subprocess_kwargs,
            timeout=30,
        )
        if result.get("error"):
            package["import_ok"] = False
            package["import_error"] = result["error"]
        else:
            package.update(result["payload"])
        updated.append(package)
    return updated


def _collect_torch_runtime_info(
    python_path: str,
    env: dict,
    subprocess_kwargs: dict,
) -> Dict[str, Any]:
    """Collect PyTorch accelerator info in its own crash-isolated process."""
    result = _run_probe(
        python_path,
        _torch_runtime_probe_script(),
        env,
        subprocess_kwargs,
        timeout=45,
    )
    if not result.get("error"):
        return result["payload"]

    return {
        "torch_import_ok": False,
        "torch_import_error": result["error"],
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "cuda_device_count": None,
        "cuda_devices": [],
        "cudnn_version": None,
        "mps_available": None,
        "mps_built": None,
    }


def _run_probe(
    python_path: str,
    script: str,
    env: dict,
    subprocess_kwargs: dict,
    timeout: int,
) -> Dict[str, Any]:
    """Run a JSON probe and convert subprocess crashes into reportable errors."""
    try:
        result = subprocess.run(
            [python_path, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            **subprocess_kwargs,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Probe timed out after {timeout} seconds."}
    except Exception as exc:
        return {"error": f"Probe failed to start: {exc}"}

    if result.returncode != 0:
        return {"error": _format_probe_failure(result)}

    try:
        return {"payload": json.loads(result.stdout)}
    except json.JSONDecodeError as exc:
        return {
            "error": "Probe returned invalid JSON: {}\n{}".format(
                exc, (result.stdout or result.stderr)[:2000]
            )
        }


def _package_import_probe_script(module_name: str) -> str:
    module_json = json.dumps(module_name)
    return textwrap.dedent(f"""
        import importlib
        import json
        import traceback

        module_name = {module_json}
        payload = {{
            "import_ok": False,
            "module_version": None,
            "module_file": None,
            "import_error": None,
        }}
        try:
            module = importlib.import_module(module_name)
            payload["import_ok"] = True
            payload["module_version"] = getattr(module, "__version__", None)
            payload["module_file"] = getattr(module, "__file__", None)
        except Exception as exc:
            payload["import_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
        print(json.dumps(payload, sort_keys=True))
        """)


def _torch_runtime_probe_script() -> str:
    return textwrap.dedent("""
        import json
        import traceback

        runtime = {
            "torch_import_ok": False,
            "torch_import_error": None,
            "torch_version": None,
            "cuda_available": None,
            "cuda_version": None,
            "cuda_device_count": None,
            "cuda_devices": [],
            "cudnn_version": None,
            "mps_available": None,
            "mps_built": None,
        }
        try:
            import torch
            runtime["torch_import_ok"] = True
            runtime["torch_version"] = getattr(torch, "__version__", None)
            runtime["cuda_version"] = getattr(torch.version, "cuda", None)
            try:
                runtime["cuda_available"] = bool(torch.cuda.is_available())
                runtime["cuda_device_count"] = int(torch.cuda.device_count())
                for idx in range(runtime["cuda_device_count"]):
                    runtime["cuda_devices"].append(torch.cuda.get_device_name(idx))
            except Exception as exc:
                runtime["cuda_available"] = False
                runtime["cuda_error"] = str(exc)
            try:
                runtime["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                pass
            try:
                mps = getattr(torch.backends, "mps", None)
                if mps is not None:
                    runtime["mps_available"] = bool(mps.is_available())
                    runtime["mps_built"] = bool(mps.is_built())
            except Exception as exc:
                runtime["mps_error"] = str(exc)
        except Exception as exc:
            runtime["torch_import_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
        print(json.dumps(runtime, sort_keys=True))
        """)


def _format_probe_failure(result: subprocess.CompletedProcess) -> str:
    output = (result.stderr or result.stdout or "").strip()
    message = f"Probe subprocess failed with code {result.returncode}"
    crash_reason = _WINDOWS_CRASH_CODES.get(result.returncode)
    if crash_reason:
        message += f" ({crash_reason})"
    if output:
        message += f": {output[:2000]}"
    return message


def _format_venv_runtime(runtime: Dict[str, Any]) -> List[str]:
    torch_runtime = runtime.get("torch_runtime", {})
    lines = [
        f"- Python version: `{_value(runtime.get('python_version'))}`",
        f"- Python executable: `{_value(runtime.get('python_executable'))}`",
        f"- Python prefix: `{_value(runtime.get('python_prefix'))}`",
        f"- Platform: `{_value(runtime.get('platform'))}`",
        "",
        "## CUDA / Accelerator",
        "",
        f"- PyTorch import OK: `{_yes_no(torch_runtime.get('torch_import_ok'))}`",
        f"- PyTorch version: `{_value(torch_runtime.get('torch_version'))}`",
        f"- PyTorch CUDA build: `{_value(torch_runtime.get('cuda_version'))}`",
        f"- CUDA available: `{_yes_no(torch_runtime.get('cuda_available'))}`",
        f"- CUDA device count: `{_value(torch_runtime.get('cuda_device_count'))}`",
    ]

    cuda_devices = torch_runtime.get("cuda_devices") or []
    for index, name in enumerate(cuda_devices):
        lines.append(f"- CUDA device {index}: `{_value(name)}`")

    if torch_runtime.get("cuda_error"):
        lines.append(f"- CUDA error: `{_value(torch_runtime['cuda_error'])}`")
    lines.extend(
        [
            f"- cuDNN version: `{_value(torch_runtime.get('cudnn_version'))}`",
            f"- MPS available: `{_yes_no(torch_runtime.get('mps_available'))}`",
            f"- MPS built: `{_yes_no(torch_runtime.get('mps_built'))}`",
        ]
    )
    if torch_runtime.get("mps_error"):
        lines.append(f"- MPS error: `{_value(torch_runtime['mps_error'])}`")
    if torch_runtime.get("torch_import_error"):
        lines.append(
            f"- PyTorch import error: `{_value(torch_runtime['torch_import_error'])}`"
        )

    lines.extend(["", "## Package Versions", ""])
    lines.extend(_format_packages(runtime.get("packages", [])))
    return lines


def _format_packages(packages: Iterable[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Package | Distribution | Import | Version | Import status |",
        "| --- | --- | --- | --- | --- |",
    ]
    details = []
    for package in packages:
        label = package.get("label") or package.get("dist_name") or "Unknown"
        dist_name = package.get("dist_name") or "Unknown"
        module_name = package.get("module_name") or "Unknown"
        version = _value(
            package.get("dist_version") or package.get("module_version"),
            missing="not installed",
        )
        import_status = "OK" if package.get("import_ok") else "FAILED"
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` |".format(
                _escape_table(label),
                _escape_table(dist_name),
                _escape_table(module_name),
                _escape_table(version),
                import_status,
            )
        )
        detail_lines = []
        if package.get("module_version") and package.get("module_version") != version:
            detail_lines.append(
                f"- module `__version__`: `{_value(package['module_version'])}`"
            )
        if package.get("module_file"):
            detail_lines.append(f"- module file: `{_value(package['module_file'])}`")
        if package.get("import_error"):
            detail_lines.append("- import error:")
            detail_lines.extend(_fenced_block(_value(package["import_error"])))
        if detail_lines:
            details.append("")
            details.append(f"<details><summary>{label} details</summary>")
            details.append("")
            details.extend(detail_lines)
            details.append("")
            details.append("</details>")
    lines.extend(details)
    return lines


def _yes_no(value: Optional[bool]) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if bool(value) else "No"


def _value(value: Any, missing: str = "Unknown") -> str:
    if value is None or value == "":
        return missing
    return _redact_home(str(value))


def _escape_table(value: Any) -> str:
    return _value(value).replace("|", "\\|").replace("\n", " ")


def _fenced_block(value: str) -> List[str]:
    """Render arbitrary text as a Markdown fenced code block.

    The fence length is chosen so that any internal backtick run cannot close
    the block, which keeps multi-line tracebacks and backtick characters from
    breaking the surrounding Markdown rendering.
    """
    text = "" if value is None else str(value)
    longest_run = 0
    current_run = 0
    for char in text:
        if char == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * max(3, longest_run + 1)
    return [fence] + text.splitlines() + [fence]


def _redact_home(value: str) -> str:
    """Replace the current user's home directory with ``~`` in report output."""
    home = os.path.expanduser("~")
    if not home or home == "~":
        return value

    redacted = value
    candidates = {home, os.path.abspath(home)}
    candidates.update(path.replace("\\", "/") for path in list(candidates))

    for candidate in sorted(candidates, key=len, reverse=True):
        if not candidate or candidate == os.path.sep:
            continue
        redacted = re.sub(
            re.escape(candidate) + r"(?=$|[/\\\s`'\"),;:])",
            "~",
            redacted,
        )

    return redacted
