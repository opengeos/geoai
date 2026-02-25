"""Virtual environment manager for GeoAI QGIS Plugin.

Manages the creation, package installation, and verification of a
virtual environment used to isolate GeoAI's Python dependencies
from the QGIS environment.

This script is adapted from the QGIS AI-Segmentation plugin's venv_manager.py. Created by the TerraLabAI team.
Source: https://github.com/TerraLabAI/QGIS_AI-Segmentation/blob/main/src/core/venv_manager.py
"""

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Optional, Tuple

from qgis.core import Qgis, QgsMessageLog

PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
CACHE_DIR = os.path.expanduser("~/.qgis_geoai")
VENV_DIR = os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")

REQUIRED_PACKAGES = [
    ("torch", ">=2.0.0"),
    ("torchvision", ">=0.15.0"),
    ("geoai-py", ""),
    ("segment-geospatial", ""),
    ("sam3", ""),
    ("deepforest", ""),
    ("omniwatermask", ""),
]

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")
CUDA_FLAG_FILE = os.path.join(VENV_DIR, "cuda_installed.txt")

# Bump when install logic changes significantly to force re-install.
_INSTALL_LOGIC_VERSION = "2"

# Bump independently for CUDA-specific install logic changes.
_CUDA_LOGIC_VERSION = "1"

# Minimum NVIDIA driver versions for each CUDA toolkit version.
_CUDA_DRIVER_REQUIREMENTS = {
    "cu128": 570,
    "cu126": 560,
    "cu124": 550,
    "cu121": 530,
}

# Blackwell (sm_120+) requires cu128; everything else works with cu124.
_MIN_COMPUTE_CAP_FOR_CU128 = 12.0

# Cache for detect_nvidia_gpu() — avoids re-running nvidia-smi.
_gpu_detect_cache = None  # type: Optional[Tuple[bool, dict]]


def _log(message: str, level=Qgis.Info):
    """Log a message to the QGIS message log.

    Args:
        message: The message to log.
        level: The log level (default: Qgis.Info).
    """
    QgsMessageLog.logMessage(message, "GeoAI", level=level)


def _log_system_info():
    """Log system information for debugging installation issues."""
    try:
        qgis_version = Qgis.QGIS_VERSION
    except Exception:
        qgis_version = "Unknown"

    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  QGIS: {qgis_version}",
        "=" * 50,
    ]
    for line in info_lines:
        _log(line, Qgis.Info)


def _check_rosetta_warning() -> Optional[str]:
    """Detect if running under Rosetta on macOS ARM.

    Returns:
        Warning message if Rosetta detected, None otherwise.
    """
    if sys.platform != "darwin":
        return None

    machine = platform.machine()
    if machine == "x86_64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Apple" in result.stdout:
                return (
                    "Warning: QGIS is running under Rosetta (x86_64 emulation) "
                    "on Apple Silicon. This may cause compatibility issues. "
                    "Consider using the native ARM64 version of QGIS."
                )
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# CUDA flag persistence
# ---------------------------------------------------------------------------


def _write_cuda_flag(value: str):
    """Persist CUDA install state.

    Args:
        value: One of 'cuda', 'cpu', or 'cuda_fallback'.
    """
    if value == "cuda_fallback":
        content = f"cuda_fallback:{_CUDA_LOGIC_VERSION}"
    else:
        content = value
    try:
        os.makedirs(os.path.dirname(CUDA_FLAG_FILE), exist_ok=True)
        with open(CUDA_FLAG_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except (OSError, IOError) as e:
        _log(f"Failed to write CUDA flag: {e}", Qgis.Warning)


def _read_cuda_flag() -> Optional[str]:
    """Read CUDA install state.

    Returns:
        One of 'cuda', 'cpu', 'cuda_fallback', or None.
    """
    try:
        with open(CUDA_FLAG_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        base = value.split(":")[0]
        if base in ("cuda", "cpu", "cuda_fallback"):
            return base
    except (OSError, IOError):
        pass
    return None


# ---------------------------------------------------------------------------
# Dependency hash tracking
# ---------------------------------------------------------------------------


def _compute_deps_hash() -> str:
    """Compute MD5 hash of REQUIRED_PACKAGES + install logic version.

    Returns:
        Hex digest string.
    """
    data = repr(REQUIRED_PACKAGES).encode("utf-8")
    data += _INSTALL_LOGIC_VERSION.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _read_deps_hash() -> Optional[str]:
    """Read stored deps hash from the venv directory.

    Returns:
        The stored hash string, or None if not found.
    """
    try:
        with open(DEPS_HASH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _write_deps_hash():
    """Write the current deps hash to the venv directory."""
    try:
        os.makedirs(os.path.dirname(DEPS_HASH_FILE), exist_ok=True)
        with open(DEPS_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
    except (OSError, IOError) as e:
        _log(f"Failed to write deps hash: {e}", Qgis.Warning)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_nvidia_gpu() -> Tuple[bool, dict]:
    """Detect if an NVIDIA GPU is present by querying nvidia-smi.

    Results are cached for the lifetime of the QGIS session.

    Returns:
        Tuple of (has_gpu, info_dict). info_dict keys: name, compute_cap,
        driver_version, memory_mb (any key may be missing).
    """
    global _gpu_detect_cache
    if _gpu_detect_cache is not None:
        return _gpu_detect_cache

    try:
        subprocess_kwargs = _get_subprocess_kwargs()
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            **subprocess_kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            best_gpu = {}
            best_compute_cap = -1.0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]

                gpu_info = {}
                if len(parts) >= 1 and parts[0]:
                    gpu_info["name"] = parts[0]
                if len(parts) >= 2 and parts[1]:
                    try:
                        gpu_info["compute_cap"] = float(parts[1])
                    except ValueError:
                        pass
                if len(parts) >= 3 and parts[2]:
                    gpu_info["driver_version"] = parts[2]
                if len(parts) >= 4 and parts[3]:
                    try:
                        gpu_info["memory_mb"] = int(float(parts[3]))
                    except ValueError:
                        pass

                cc = gpu_info.get("compute_cap", 0.0)
                if cc > best_compute_cap:
                    best_compute_cap = cc
                    best_gpu = gpu_info

            if not best_gpu:
                _gpu_detect_cache = (False, {})
                return _gpu_detect_cache

            _log(
                "NVIDIA GPU detected (best of {}): {}".format(len(lines), best_gpu),
                Qgis.Info,
            )
            _gpu_detect_cache = (True, best_gpu)
            return _gpu_detect_cache
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        _log("nvidia-smi timed out", Qgis.Warning)
    except Exception as e:
        _log(f"nvidia-smi check failed: {e}", Qgis.Warning)

    _gpu_detect_cache = (False, {})
    return _gpu_detect_cache


def _select_cuda_index(gpu_info: dict) -> Optional[str]:
    """Choose the correct PyTorch CUDA wheel index based on GPU info.

    Args:
        gpu_info: Dict with GPU information from detect_nvidia_gpu().

    Returns:
        'cu128', 'cu124', or None if driver is too old.
    """
    compute_cap = gpu_info.get("compute_cap")
    gpu_name = gpu_info.get("name", "")

    if compute_cap is not None:
        needs_cu128 = compute_cap >= _MIN_COMPUTE_CAP_FOR_CU128
    else:
        needs_cu128 = "RTX 50" in gpu_name.upper()

    cuda_index = "cu128" if needs_cu128 else "cu124"

    driver_str = gpu_info.get("driver_version", "")
    if driver_str:
        try:
            driver_major = int(driver_str.split(".")[0])
            required = _CUDA_DRIVER_REQUIREMENTS.get(cuda_index, 0)
            if driver_major < required:
                _log(
                    "NVIDIA driver {} too old for {} (needs >= {}), "
                    "will use CPU instead".format(driver_str, cuda_index, required),
                    Qgis.Warning,
                )
                return None
        except (ValueError, IndexError):
            _log(
                f"Could not parse driver version: {driver_str}",
                Qgis.Warning,
            )

    return cuda_index


# ---------------------------------------------------------------------------
# Error detection helpers
# ---------------------------------------------------------------------------

_SSL_ERROR_PATTERNS = [
    "ssl",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
    "SSLError",
    "SSLCertVerificationError",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
]

_NETWORK_ERROR_PATTERNS = [
    "connectionreseterror",
    "connection aborted",
    "connection was forcibly closed",
    "remotedisconnected",
    "connectionerror",
    "newconnectionerror",
    "maxretryerror",
    "protocolerror",
    "readtimeouterror",
    "connecttimeouterror",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
]

# Windows NTSTATUS crash codes
_WINDOWS_CRASH_CODES = {
    3221225477,  # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed
    3221225725,  # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed
    3221225781,  # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed
}


def _is_ssl_error(stderr: str) -> bool:
    """Detect SSL/certificate errors in pip output.

    Args:
        stderr: The error output from pip.

    Returns:
        True if SSL errors detected.
    """
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in _SSL_ERROR_PATTERNS)


def _is_hash_mismatch(output: str) -> bool:
    """Detect pip hash mismatch errors.

    Args:
        output: The pip output to check.

    Returns:
        True if hash mismatch detected.
    """
    output_lower = output.lower()
    return "do not match the hashes" in output_lower or "hash mismatch" in output_lower


def _get_pip_ssl_flags() -> List[str]:
    """Get pip flags to bypass SSL verification for corporate proxies.

    Returns:
        List of pip command-line flags.
    """
    return [
        "--trusted-host",
        "pypi.org",
        "--trusted-host",
        "pypi.python.org",
        "--trusted-host",
        "files.pythonhosted.org",
    ]


def _is_network_error(output: str) -> bool:
    """Detect transient network/connection errors in pip output.

    Args:
        output: The pip output to check.

    Returns:
        True if network errors detected (excluding SSL).
    """
    output_lower = output.lower()
    if _is_ssl_error(output):
        return False
    return any(p in output_lower for p in _NETWORK_ERROR_PATTERNS)


def _is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip output.

    Args:
        stderr: The error output from pip.

    Returns:
        True if antivirus blocking detected.
    """
    stderr_lower = stderr.lower()
    patterns = [
        "access is denied",
        "winerror 5",
        "winerror 225",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
        "blocked by group policy",
        "applocker",
        "blocked by your organization",
    ]
    return any(p in stderr_lower for p in patterns)


def _is_proxy_auth_error(output: str) -> bool:
    """Detect proxy authentication errors (HTTP 407).

    Args:
        output: The pip output to check.

    Returns:
        True if proxy auth error detected.
    """
    output_lower = output.lower()
    patterns = [
        "407 proxy authentication",
        "proxy authentication required",
        "proxyerror",
    ]
    return any(p in output_lower for p in patterns)


def _is_windows_process_crash(returncode: int) -> bool:
    """Detect Windows process crashes.

    Args:
        returncode: The subprocess return code.

    Returns:
        True if the return code indicates a Windows crash.
    """
    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _get_clean_env_for_venv() -> dict:
    """Get a clean environment dict for subprocess calls.

    Strips QGIS-specific variables to prevent interference.

    Returns:
        A clean copy of os.environ.
    """
    env = os.environ.copy()
    for var in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH",
        "QGIS_PLUGINPATH",
        "PROJ_DATA",
        "PROJ_LIB",
        "GDAL_DATA",
        "GDAL_DRIVER_PATH",
    ):
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"

    # Ensure CUDA libraries are discoverable for GPU-accelerated torch.
    # QGIS desktop launchers may not inherit the user's shell profile,
    # so CUDA_PATH and LD_LIBRARY_PATH may be empty.
    _cuda_lib_dirs = []
    cuda_path = env.get("CUDA_PATH", "")
    if cuda_path:
        _cuda_lib_dirs.append(os.path.join(cuda_path, "lib64"))
    for candidate in ("/opt/cuda/lib64", "/usr/local/cuda/lib64"):
        if os.path.isdir(candidate) and candidate not in _cuda_lib_dirs:
            _cuda_lib_dirs.append(candidate)
    if _cuda_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in existing.split(":") if p]
        for d in _cuda_lib_dirs:
            if d not in parts:
                parts.append(d)
        env["LD_LIBRARY_PATH"] = ":".join(parts)

    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)
    return env


def _get_subprocess_kwargs() -> dict:
    """Get platform-specific subprocess kwargs.

    Includes a safe ``cwd`` so that the venv Python never finds the QGIS
    plugin package (also named ``geoai``) via the current working directory.

    Returns:
        Dict with cwd and startupinfo (Windows).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    kwargs = {"cwd": CACHE_DIR}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _get_qgis_proxy_settings() -> Optional[str]:
    """Read proxy configuration from QGIS settings.

    Returns:
        A proxy URL string, or None if not configured.
    """
    try:
        from qgis.core import QgsSettings
        from urllib.parse import quote as url_quote

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None

        port = settings.value("proxy/proxyPort", "", type=str)
        user = settings.value("proxy/proxyUser", "", type=str)
        password = settings.value("proxy/proxyPassword", "", type=str)

        proxy_url = "http://"
        if user:
            proxy_url += url_quote(user, safe="")
            if password:
                proxy_url += ":" + url_quote(password, safe="")
            proxy_url += "@"
        proxy_url += host
        if port:
            proxy_url += f":{port}"
        return proxy_url
    except Exception as e:
        _log(f"Could not read QGIS proxy settings: {e}", Qgis.Warning)
        return None


def _get_pip_proxy_args() -> List[str]:
    """Get pip --proxy argument if QGIS proxy is configured.

    Returns:
        List with --proxy args, or empty list.
    """
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        safe_url = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url
        _log(f"Using QGIS proxy for pip: {safe_url}", Qgis.Info)
        return ["--proxy", proxy_url]
    return []


# ---------------------------------------------------------------------------
# Venv path helpers
# ---------------------------------------------------------------------------


def get_venv_dir() -> str:
    """Get the venv directory path.

    Returns:
        Path to the virtual environment directory.
    """
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    """Get the site-packages directory within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the site-packages directory.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
    else:
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python") and os.path.isdir(
                    os.path.join(lib_dir, entry)
                ):
                    site_packages = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(site_packages):
                        return site_packages

        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return os.path.join(venv_dir, "lib", py_version, "site-packages")


def get_venv_python_path(venv_dir: str = None) -> str:
    """Get the Python executable path within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the venv Python executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    """Get the pip executable path within the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Path to the venv pip executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_dir, "bin", "pip")


def venv_exists(venv_dir: str = None) -> bool:
    """Check if the venv exists and has a Python executable.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        True if the venv Python executable exists.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    python_path = get_venv_python_path(venv_dir)
    return os.path.exists(python_path)


def ensure_venv_packages_available() -> bool:
    """Add the venv site-packages to sys.path so packages can be imported.

    Returns:
        True if packages were made available, False otherwise.
    """
    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        _log(f"Venv site-packages not found: {site_packages}", Qgis.Warning)
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.Info)

    # Fix PROJ database for the venv's pyproj / rasterio / pyogrio.
    # QGIS may set PROJ_LIB to its own PROJ data, but the venv's pyproj
    # bundles its own proj.db and needs PROJ_DATA to point there.
    _fix_proj_data(site_packages)

    # On Windows, register DLL directories for native packages (torch, etc.)
    # so that the OS loader can find their DLLs when importing from the venv
    # inside QGIS's process.  Also patch geoai/__init__.py to guard
    # torch-dependent imports against DLL load failures (WinError 127).
    if sys.platform == "win32":
        _add_windows_dll_directories(site_packages)
        _repair_corrupted_geoai_init(site_packages)
        patch_geoai_init_for_torch_guard(site_packages)

    # Fix stale typing_extensions (QGIS may load old version missing TypeIs)
    if "typing_extensions" in sys.modules:
        try:
            typing_ext = sys.modules["typing_extensions"]
            if not hasattr(typing_ext, "TypeIs"):
                old_ver = getattr(typing_ext, "__version__", "unknown")
                del sys.modules["typing_extensions"]
                import typing_extensions as new_te

                _log(
                    "Reloaded typing_extensions {} -> {} from venv".format(
                        old_ver, new_te.__version__
                    ),
                    Qgis.Info,
                )
        except Exception:
            _log(
                "Failed to reload typing_extensions, torch may fail",
                Qgis.Warning,
            )

    return True


def _add_windows_dll_directories(site_packages: str) -> None:
    """Register DLL search directories for native packages on Windows.

    Torch and other native packages ship DLLs in subdirectories (e.g.
    ``torch/lib/``) that the OS loader doesn't search by default when loading
    from a foreign venv inside QGIS's process.  We add them via
    ``os.add_dll_directory()`` and also prepend to ``PATH`` to cover legacy
    ``LoadLibrary`` calls without search flags.

    Args:
        site_packages: Path to the venv site-packages directory.
    """
    dll_dirs = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "torch", "bin"),
        os.path.join(site_packages, "torchvision"),
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
                _log(f"Added DLL directory: {dll_dir}", Qgis.Info)
            except OSError as exc:
                _log(f"add_dll_directory({dll_dir}) failed: {exc}", Qgis.Warning)
            if dll_dir not in path_parts:
                path_parts.insert(0, dll_dir)

    os.environ["PATH"] = os.pathsep.join(path_parts)


def _repair_corrupted_geoai_init(site_packages: str) -> None:
    """Detect and repair a corrupted geoai/__init__.py in the venv.

    A previous version of the patching logic could produce invalid Python
    (empty ``try:`` blocks).  If the file fails to compile, reinstall
    geoai-py to get a clean copy.

    Args:
        site_packages: Path to venv site-packages directory.
    """
    init_path = os.path.join(site_packages, "geoai", "__init__.py")
    if not os.path.exists(init_path):
        return

    try:
        with open(init_path, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, init_path, "exec")
        # File compiles OK — nothing to repair.
        return
    except SyntaxError as exc:
        _log(
            "geoai/__init__.py has syntax error (line {}), "
            "reinstalling geoai-py to repair...".format(exc.lineno),
            Qgis.Warning,
        )

    # Reinstall geoai-py to get a clean copy.
    python_path = get_venv_python_path()
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    try:
        cmd = [
            python_path,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "geoai-py",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs,
        )
        if result.returncode == 0:
            _log("Repaired geoai-py via reinstall", Qgis.Success)
        else:
            err = result.stderr or result.stdout or ""
            _log(
                "Failed to repair geoai-py: {}".format(err[:200]),
                Qgis.Warning,
            )
    except Exception as exc:
        _log("Exception repairing geoai-py: {}".format(exc), Qgis.Warning)


def patch_geoai_init_for_torch_guard(site_packages: str = None) -> bool:
    """Patch the venv's geoai package for Windows DLL load failures.

    On Windows, torch DLLs may fail to load inside QGIS's Python process
    (WinError 127), raising ``OSError`` instead of ``ImportError``.

    This function applies two transformations:

    1. **Widen existing guards**: ``except ImportError:`` →
       ``except (ImportError, OSError):`` so existing try/except blocks
       also catch DLL load failures.

    2. **Wrap bare imports**: Any torch-dependent ``from .xxx import ...``
       that is NOT already inside a ``try`` block gets wrapped in
       ``try: ... except (ImportError, OSError): pass``.

    Both ``geoai/__init__.py`` and ``geoai/tools/__init__.py`` are patched.
    The result is verified with ``compile()`` before writing — if the
    patched code would have a syntax error, the file is left unchanged.

    This is idempotent — already-patched files are left unchanged.

    Args:
        site_packages: Path to venv site-packages. Uses default if None.

    Returns:
        True if patched (or already patched), False on failure.
    """
    if site_packages is None:
        site_packages = get_venv_site_packages()

    geoai_dir = os.path.join(site_packages, "geoai")

    # Files to patch with the simple except replacement.
    simple_targets = [
        os.path.join(geoai_dir, "__init__.py"),
        os.path.join(geoai_dir, "tools", "__init__.py"),
    ]

    any_patched = False

    for filepath in simple_targets:
        if not os.path.exists(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = content.replace(
                "except ImportError:", "except (ImportError, OSError):"
            )

            if new_content != content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                rel = os.path.relpath(filepath, site_packages)
                _log("Patched {} (except widening)".format(rel), Qgis.Info)
                any_patched = True

        except Exception as exc:
            rel = os.path.relpath(filepath, site_packages)
            _log("Failed to patch {}: {}".format(rel, exc), Qgis.Warning)

    # Wrap bare torch-dependent imports in geoai/__init__.py.
    init_path = os.path.join(geoai_dir, "__init__.py")
    if os.path.exists(init_path):
        try:
            any_patched |= _wrap_bare_imports(init_path)
        except Exception as exc:
            _log("Failed to wrap bare imports: {}".format(exc), Qgis.Warning)

    if not any_patched:
        _log("geoai package already patched, no changes needed", Qgis.Info)

    return True


def _wrap_bare_imports(filepath: str) -> bool:
    """Wrap bare torch-dependent imports in try/except guards.

    Processes the file line-by-line to find import statements that match
    known torch-dependent patterns and are NOT already inside a try block.
    Wraps them in ``try: ... except (ImportError, OSError): pass``.

    The patched content is verified with ``compile()`` before writing.

    Args:
        filepath: Path to the Python file to patch.

    Returns:
        True if the file was modified.
    """
    # Patterns that indicate torch-dependent imports.
    bare_patterns = [
        "from .geoai import",
        "from .dinov3 import",
        "from .timm_train import",
        "from .recognize import",
        "from .timm_segment import",
        "from .timm_regress import",
        "from . import tools",
        "from .tools import",
        "from .object_detect import",
        "from .water import",
        "from .canopy import",
        "from .change_detection import",
        "from .prithvi import",
        "from .moondream import",
        "from .map_widgets import",
        "from .onnx import",
    ]

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    modified = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line starts a bare import matching our patterns.
        matches_pattern = any(stripped.startswith(p) for p in bare_patterns)

        if matches_pattern:
            # Check if already inside a try block (look at preceding
            # non-blank lines for "try:").
            already_guarded = False
            for j in range(len(new_lines) - 1, max(len(new_lines) - 5, -1), -1):
                prev = new_lines[j].strip()
                if prev == "try:":
                    already_guarded = True
                    break
                if prev and not prev.startswith("#"):
                    break  # Non-blank, non-comment line that isn't try:

            if already_guarded:
                new_lines.append(line)
                i += 1
                continue

            # Collect the full import statement (may span multiple lines
            # if it uses parentheses).
            import_lines = [line]
            if "(" in line and ")" not in line:
                i += 1
                while i < len(lines):
                    import_lines.append(lines[i])
                    if ")" in lines[i]:
                        i += 1
                        break
                    i += 1
            else:
                i += 1

            # Determine indentation from the original line.
            indent = line[: len(line) - len(line.lstrip())]

            # Build the wrapped block.
            new_lines.append("{}try:\n".format(indent))
            for imp_line in import_lines:
                # Add 4 spaces of indentation to the import line.
                if imp_line.strip():
                    new_lines.append("{}    {}\n".format(indent, imp_line.strip()))
                else:
                    new_lines.append("\n")
            new_lines.append("{}except (ImportError, OSError):\n".format(indent))
            new_lines.append("{}    pass\n".format(indent))
            modified = True
        else:
            new_lines.append(line)
            i += 1

    if not modified:
        return False

    new_content = "".join(new_lines)

    # Safety check: verify the patched code compiles.
    try:
        compile(new_content, filepath, "exec")
    except SyntaxError as exc:
        _log(
            "Patched {} would have syntax error at line {}, "
            "skipping write".format(filepath, exc.lineno),
            Qgis.Warning,
        )
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    _log(
        "Wrapped bare imports in {}".format(os.path.basename(filepath)),
        Qgis.Info,
    )
    return True


def _fix_proj_data(site_packages: str) -> None:
    """Set PROJ_DATA/PROJ_LIB and GDAL_DATA for the venv's geospatial libraries.

    The venv's pyproj bundles its own PROJ database at
    ``<site-packages>/pyproj/proj_dir/share/proj/``.  QGIS may or may not
    set ``PROJ_LIB`` to its own copy, but the venv's native pyproj
    library needs the matching version. Similarly, rasterio/pyogrio may
    bundle their own GDAL data.

    Args:
        site_packages: Path to the venv's site-packages directory.
    """
    # --- PROJ database ---
    proj_candidates = [
        os.path.join(site_packages, "pyproj", "proj_dir", "share", "proj"),
        os.path.join(site_packages, "rasterio", "proj_data"),
        os.path.join(site_packages, "pyogrio", "proj_data"),
    ]

    for candidate in proj_candidates:
        proj_db = os.path.join(candidate, "proj.db")
        if os.path.exists(proj_db):
            os.environ["PROJ_DATA"] = candidate
            os.environ["PROJ_LIB"] = candidate
            _log(f"Set PROJ_DATA={candidate}", Qgis.Info)
            break
    else:
        _log("No venv proj.db found; PROJ_DATA unchanged", Qgis.Warning)

    # --- GDAL data ---
    gdal_candidates = [
        os.path.join(site_packages, "rasterio", "gdal_data"),
        os.path.join(site_packages, "pyogrio", "gdal_data"),
    ]

    for candidate in gdal_candidates:
        if os.path.isdir(candidate):
            os.environ["GDAL_DATA"] = candidate
            _log(f"Set GDAL_DATA={candidate}", Qgis.Info)
            break


# ---------------------------------------------------------------------------
# System Python resolution
# ---------------------------------------------------------------------------


def _get_qgis_python() -> Optional[str]:
    """Get the path to QGIS's bundled Python on Windows.

    Returns:
        Path to the Python executable, or None if not found.
    """
    if sys.platform != "win32":
        return None

    python_path = os.path.join(sys.prefix, "python.exe")
    if not os.path.exists(python_path):
        python_path = os.path.join(sys.prefix, "python3.exe")

    if not os.path.exists(python_path):
        _log("QGIS bundled Python not found at sys.prefix", Qgis.Warning)
        return None

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            **subprocess_kwargs,
        )
        if result.returncode == 0:
            _log(f"QGIS Python verified: {result.stdout.strip()}", Qgis.Info)
            return python_path
        else:
            _log(
                f"QGIS Python failed verification: {result.stderr}",
                Qgis.Warning,
            )
            return None
    except Exception as e:
        _log(f"QGIS Python verification error: {e}", Qgis.Warning)
        return None


def _get_system_python() -> str:
    """Get the path to the Python executable for creating venvs.

    Uses standalone Python downloaded by python_manager, with fallback
    to QGIS's bundled Python on Windows.

    Returns:
        Path to the Python executable.

    Raises:
        RuntimeError: If no suitable Python is found.
    """
    from .python_manager import get_standalone_python_path, standalone_python_exists

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}", Qgis.Info)
        return python_path

    if sys.platform == "win32":
        qgis_python = _get_qgis_python()
        if qgis_python:
            _log(
                "Standalone Python unavailable, using QGIS Python as fallback",
                Qgis.Warning,
            )
            return qgis_python

    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------


def _cleanup_partial_venv(venv_dir: str):
    """Remove a partially-created venv directory.

    Args:
        venv_dir: Path to the venv directory to clean up.
    """
    if os.path.exists(venv_dir):
        try:
            shutil.rmtree(venv_dir, ignore_errors=True)
            _log(f"Cleaned up partial venv: {venv_dir}", Qgis.Info)
        except Exception:
            _log(f"Could not clean up partial venv: {venv_dir}", Qgis.Warning)


def create_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Create a virtual environment.

    Args:
        venv_dir: Optional directory for the venv. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}", Qgis.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.Info)

    cmd = [system_python, "-m", "venv", venv_dir]

    try:
        env = _get_clean_env_for_venv()
        subprocess_kwargs = _get_subprocess_kwargs()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs,
        )

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.Success)

            pip_path = get_venv_pip_path(venv_dir)
            if not os.path.exists(pip_path):
                _log(
                    "pip not found in venv, bootstrapping with ensurepip...",
                    Qgis.Info,
                )
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [
                    python_in_venv,
                    "-m",
                    "ensurepip",
                    "--upgrade",
                ]
                try:
                    ensurepip_result = subprocess.run(
                        ensurepip_cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env=env,
                        **subprocess_kwargs,
                    )
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.Success)
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout
                        _log(f"ensurepip failed: {err[:200]}", Qgis.Warning)
                        _cleanup_partial_venv(venv_dir)
                        return False, f"Failed to bootstrap pip: {err[:200]}"
                except Exception as e:
                    _log(f"ensurepip exception: {e}", Qgis.Warning)
                    _cleanup_partial_venv(venv_dir)
                    return False, f"Failed to bootstrap pip: {str(e)[:200]}"

            if progress_callback:
                progress_callback(15, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = (
                result.stderr or result.stdout or f"Return code {result.returncode}"
            )
            _log(f"Failed to create venv: {error_msg}", Qgis.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Virtual environment creation timed out", Qgis.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Pip install with progress
# ---------------------------------------------------------------------------


class _PipResult:
    """Lightweight result object compatible with subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _parse_pip_download_line(line: str) -> Optional[str]:
    """Extract a human-readable status from a pip stdout/stderr line.

    Args:
        line: A line from pip output.

    Returns:
        Human-readable download status string, or None.
    """
    m = re.search(r"Downloading\s+(\S+)\s+\(([^)]+)\)", line)
    if not m:
        return None

    raw_name = m.group(1)
    size = m.group(2)

    if "/" in raw_name:
        raw_name = raw_name.rsplit("/", 1)[-1]

    name_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)", raw_name)
    pkg_name = name_match.group(1) if name_match else raw_name

    size_match = re.match(r"([\d.]+)\s*(kB|MB|GB)", size)
    if size_match:
        num = float(size_match.group(1))
        unit = size_match.group(2)
        if unit == "MB" and num >= 1000:
            size = "{:.1f} GB".format(num / 1000)

    return "Downloading {} ({})".format(pkg_name, size)


def _run_pip_install(
    cmd: List[str],
    timeout: int,
    env: dict,
    subprocess_kwargs: dict,
    package_name: str,
    package_index: int,
    total_packages: int,
    progress_start: int,
    progress_end: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    is_cuda: bool = False,
) -> _PipResult:
    """Run a pip install command with real-time progress updates.

    Args:
        cmd: The pip install command to run.
        timeout: Maximum time in seconds.
        env: Environment variables dict.
        subprocess_kwargs: Platform-specific kwargs for subprocess.
        package_name: Name of the package being installed.
        package_index: Index in the package list (0-based).
        total_packages: Total number of packages.
        progress_start: Start percentage for this package's progress range.
        progress_end: End percentage for this package's progress range.
        progress_callback: Optional progress callback.
        cancel_check: Optional cancellation check callback.
        is_cuda: Whether this is a CUDA package install.

    Returns:
        _PipResult with returncode, stdout, and stderr.
    """
    poll_interval = 2

    stdout_fd, stdout_path = tempfile.mkstemp(suffix="_stdout.txt", prefix="pip_")
    stderr_fd, stderr_path = tempfile.mkstemp(suffix="_stderr.txt", prefix="pip_")

    try:
        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    except Exception:
        try:
            os.close(stdout_fd)
        except Exception:
            pass
        try:
            os.close(stderr_fd)
        except Exception:
            pass
        raise

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            **subprocess_kwargs,
        )

        start_time = time.monotonic()
        last_download_status = ""

        while True:
            try:
                process.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass

            elapsed = int(time.monotonic() - start_time)

            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                return _PipResult(-1, "", "Installation cancelled")

            if elapsed >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise subprocess.TimeoutExpired(cmd, timeout)

            # Read last lines to find download progress
            try:
                with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(0, 2)
                    file_size = f.tell()
                    read_from = max(0, file_size - 4096)
                    f.seek(read_from)
                    tail = f.read()
                    lines = tail.strip().split("\n")
                    for line in reversed(lines):
                        parsed = _parse_pip_download_line(line)
                        if parsed:
                            last_download_status = parsed
                            break
            except Exception:
                pass

            if elapsed >= 60:
                elapsed_str = "{}m {}s".format(elapsed // 60, elapsed % 60)
            else:
                elapsed_str = "{}s".format(elapsed)

            if last_download_status:
                msg = "{}... {}".format(last_download_status, elapsed_str)
            elif is_cuda and package_name == "torch":
                msg = "Installing GPU PyTorch (~2.5 GB)... {}".format(elapsed_str)
            elif package_name == "torch":
                msg = "Downloading PyTorch (~600 MB)... {}".format(elapsed_str)
            else:
                msg = "Installing {}... {}".format(package_name, elapsed_str)

            progress_range = progress_end - progress_start
            if timeout > 0:
                fraction = min(elapsed / timeout, 0.9)
            else:
                fraction = 0
            interpolated = progress_start + int(progress_range * fraction)
            interpolated = min(interpolated, progress_end - 1)

            if progress_callback:
                progress_callback(interpolated, msg)

        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        try:
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                full_stdout = f.read()
        except Exception:
            full_stdout = ""

        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                full_stderr = f.read()
        except Exception:
            full_stderr = ""

        return _PipResult(process.returncode, full_stdout, full_stderr)

    except subprocess.TimeoutExpired:
        raise
    except Exception:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        raise
    finally:
        if stdout_file is not None:
            try:
                stdout_file.close()
            except Exception:
                pass
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass
        try:
            os.unlink(stdout_path)
        except Exception:
            pass
        try:
            os.unlink(stderr_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------


def _is_cpu_torch_installed(
    python_path: str, env: dict, subprocess_kwargs: dict
) -> bool:
    """Check if installed torch has no CUDA support (CPU-only build).

    Args:
        python_path: Path to the venv Python.
        env: Environment dict for subprocess.
        subprocess_kwargs: Platform-specific subprocess kwargs.

    Returns:
        True if CPU-only torch is installed.
    """
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch; print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            **subprocess_kwargs,
        )
        if result.returncode == 0:
            return result.stdout.strip() == "None"
    except Exception:
        pass
    return False


def _reinstall_cpu_torch(
    venv_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    """Reinstall CPU-only torch/torchvision after CUDA failure.

    Args:
        venv_dir: Path to the virtual environment.
        progress_callback: Optional progress callback.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    _log("Reinstalling CPU-only torch/torchvision...", Qgis.Warning)
    if progress_callback:
        progress_callback(96, "CUDA failed, reinstalling CPU torch...")

    try:
        subprocess.run(
            [python_path, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **subprocess_kwargs,
        )
    except Exception as e:
        _log(f"torch uninstall error (continuing): {e}", Qgis.Warning)

    for pkg in ("torch>=2.0.0", "torchvision>=0.15.0"):
        try:
            cmd = (
                [
                    python_path,
                    "-m",
                    "pip",
                    "install",
                    "--no-warn-script-location",
                    "--disable-pip-version-check",
                    "--prefer-binary",
                ]
                + _get_pip_ssl_flags()
                + [pkg]
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                **subprocess_kwargs,
            )
            if result.returncode == 0:
                _log(f"Installed {pkg} (CPU)", Qgis.Success)
            else:
                err = result.stderr or result.stdout or ""
                _log(f"Failed to install {pkg} (CPU): {err[:200]}", Qgis.Warning)
        except Exception as e:
            _log(f"Exception installing {pkg} (CPU): {e}", Qgis.Warning)

    if progress_callback:
        progress_callback(98, "CPU torch installed, re-verifying...")


def _verify_cuda_in_venv(venv_dir: str) -> bool:
    """Run a CUDA smoke test inside the venv.

    Args:
        venv_dir: Path to the virtual environment.

    Returns:
        True if CUDA is functional.
    """
    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    cuda_test_code = (
        "import torch; "
        "print('torch=' + torch.__version__); "
        "print('cuda_built=' + str(torch.version.cuda)); "
        "assert torch.cuda.is_available(), 'CUDA not available'; "
        "print('device=' + torch.cuda.get_device_name(0)); "
        "t = torch.zeros(1, device='cuda'); "
        "torch.cuda.synchronize(); "
        "print('CUDA OK')"
    )

    try:
        # Retry once because CUDA initialization can be transiently slow/flaky
        # immediately after installation on some Windows systems.
        for attempt in (1, 2):
            result = subprocess.run(
                [python_path, "-c", cuda_test_code],
                capture_output=True,
                text=True,
                timeout=180 if attempt == 2 else 120,
                env=env,
                **subprocess_kwargs,
            )
            if result.returncode == 0 and "CUDA OK" in result.stdout:
                _log(
                    "CUDA verification passed: {}".format(result.stdout.strip()[:400]),
                    Qgis.Success,
                )
                return True

            out = result.stdout or ""
            err = result.stderr or ""
            _log(
                "CUDA verification attempt {} failed (rc={}).\nstdout: {}\nstderr: {}".format(
                    attempt, result.returncode, out[:400], err[:400]
                ),
                Qgis.Warning,
            )
            if attempt == 1:
                time.sleep(2)
        return False
    except Exception as e:
        _log(f"CUDA verification exception: {e}", Qgis.Warning)
        return False


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False,
) -> Tuple[bool, str]:
    """Install all required packages into the virtual environment.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).
        cancel_check: Optional function returning True to cancel.
        cuda_enabled: Whether to install CUDA-enabled PyTorch.

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    pip_path = get_venv_pip_path(venv_dir)
    _log(f"Installing dependencies using: {pip_path}", Qgis.Info)
    if cuda_enabled:
        _log("CUDA mode enabled - will install GPU-accelerated PyTorch", Qgis.Info)

    _cuda_fell_back = False
    _driver_too_old = False

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_range = 80

    # Weighted progress: torch is heaviest, then geoai-py, segment-geospatial, etc.
    # Order: torch, torchvision, geoai-py, segment-geospatial, sam3, deepforest,
    #         omniwatermask
    if cuda_enabled:
        _weights = [30, 8, 20, 12, 10, 10, 10]
    else:
        _weights = [20, 8, 25, 15, 10, 12, 10]
    weight_total = sum(_weights)

    _cumulative = [0]
    for w in _weights:
        _cumulative.append(_cumulative[-1] + w)

    def _pkg_progress_start(idx):
        return base_progress + int(progress_range * _cumulative[idx] / weight_total)

    def _pkg_progress_end(idx):
        return base_progress + int(progress_range * _cumulative[idx + 1] / weight_total)

    python_path = get_venv_python_path(venv_dir)

    # Detect CPU-only torch already in venv (needs force reinstall for CUDA)
    _force_cuda_reinstall = False
    if cuda_enabled:
        _precheck_env = _get_clean_env_for_venv()
        _precheck_kwargs = _get_subprocess_kwargs()
        if _is_cpu_torch_installed(python_path, _precheck_env, _precheck_kwargs):
            _force_cuda_reinstall = True
            _log(
                "CPU torch detected in venv, CUDA packages will use "
                "--force-reinstall",
                Qgis.Info,
            )

    for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.Warning)
            return False, "Installation cancelled"

        package_spec = f"{package_name}{version_spec}"
        pkg_start = _pkg_progress_start(i)
        pkg_end = _pkg_progress_end(i)

        is_cuda_package = cuda_enabled and package_name in (
            "torch",
            "torchvision",
        )

        if progress_callback:
            if package_name == "torch" and cuda_enabled:
                progress_callback(
                    pkg_start,
                    "Installing GPU dependencies... ({}/{})".format(
                        i + 1, total_packages
                    ),
                )
            elif package_name == "torch":
                progress_callback(
                    pkg_start,
                    "Installing {} (~600MB)... ({}/{})".format(
                        package_name, i + 1, total_packages
                    ),
                )
            else:
                progress_callback(
                    pkg_start,
                    "Installing {}... ({}/{})".format(
                        package_name, i + 1, total_packages
                    ),
                )

        _log(f"[{i + 1}/{total_packages}] Installing {package_spec}...", Qgis.Info)

        pip_args = [
            "install",
            "--upgrade",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--prefer-binary",
        ]
        pip_args.extend(_get_pip_ssl_flags())
        pip_args.extend(_get_pip_proxy_args())
        pip_args.append(package_spec)

        if is_cuda_package:
            _, gpu_info = detect_nvidia_gpu()
            cuda_index = _select_cuda_index(gpu_info)
            if cuda_index is None:
                _log(
                    "Driver too old for CUDA, installing CPU {} instead".format(
                        package_name
                    ),
                    Qgis.Warning,
                )
                is_cuda_package = False
                _driver_too_old = True
            else:
                pip_args.extend(
                    [
                        "--index-url",
                        "https://download.pytorch.org/whl/{}".format(cuda_index),
                        "--no-cache-dir",
                    ]
                )
                _log(
                    "Using CUDA {} index for {}".format(cuda_index, package_name),
                    Qgis.Info,
                )

        env = _get_clean_env_for_venv()
        subprocess_kwargs = _get_subprocess_kwargs()

        # Uninstall CPU torch before CUDA install
        if _force_cuda_reinstall and is_cuda_package:
            _log(
                "Uninstalling CPU {} before CUDA install".format(package_name),
                Qgis.Info,
            )
            try:
                subprocess.run(
                    [python_path, "-m", "pip", "uninstall", "-y", package_name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                    **subprocess_kwargs,
                )
            except Exception as exc:
                _log(
                    f"Failed to uninstall CPU {package_name}: {exc}",
                    Qgis.Warning,
                )

        if is_cuda_package and package_name in ("torch", "torchvision"):
            pkg_timeout = 2400
        elif package_name == "geoai-py":
            pkg_timeout = 1200  # geoai-py has many transitive deps
        else:
            pkg_timeout = 600

        install_failed = False
        install_error_msg = ""
        last_returncode = None

        try:
            base_cmd = [python_path, "-m", "pip"] + pip_args

            result = _run_pip_install(
                cmd=base_cmd,
                timeout=pkg_timeout,
                env=env,
                subprocess_kwargs=subprocess_kwargs,
                package_name=package_name,
                package_index=i,
                total_packages=total_packages,
                progress_start=pkg_start,
                progress_end=pkg_end,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                is_cuda=is_cuda_package,
            )

            if result.returncode == -1 and "cancelled" in (result.stderr or "").lower():
                _log("Installation cancelled by user", Qgis.Warning)
                return False, "Installation cancelled"

            # Retry on hash mismatch with --no-cache-dir
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or ""
                if _is_hash_mismatch(error_output):
                    _log(
                        "Hash mismatch detected, retrying with --no-cache-dir...",
                        Qgis.Warning,
                    )
                    nocache_cmd = base_cmd + ["--no-cache-dir"]
                    result = _run_pip_install(
                        cmd=nocache_cmd,
                        timeout=pkg_timeout,
                        env=env,
                        subprocess_kwargs=subprocess_kwargs,
                        package_name=package_name,
                        package_index=i,
                        total_packages=total_packages,
                        progress_start=pkg_start,
                        progress_end=pkg_end,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        is_cuda=is_cuda_package,
                    )

            # Retry on network errors
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or ""
                if _is_network_error(error_output):
                    for attempt in range(1, 3):
                        _log(
                            "Network error, retrying in 5s "
                            "(attempt {}/2)...".format(attempt),
                            Qgis.Warning,
                        )
                        if progress_callback:
                            progress_callback(
                                pkg_start,
                                "Network error, retrying {}...".format(package_name),
                            )
                        time.sleep(5)
                        if cancel_check and cancel_check():
                            return False, "Installation cancelled"
                        result = _run_pip_install(
                            cmd=base_cmd,
                            timeout=pkg_timeout,
                            env=env,
                            subprocess_kwargs=subprocess_kwargs,
                            package_name=package_name,
                            package_index=i,
                            total_packages=total_packages,
                            progress_start=pkg_start,
                            progress_end=pkg_end,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            is_cuda=is_cuda_package,
                        )
                        if result.returncode == 0:
                            break

            if result.returncode == 0:
                _log(f"Successfully installed {package_spec}", Qgis.Success)
                if progress_callback:
                    progress_callback(pkg_end, f"{package_name} installed")
            else:
                error_msg = (
                    result.stderr or result.stdout or f"Return code {result.returncode}"
                )
                _log(
                    f"Failed to install {package_spec}: {error_msg[:500]}",
                    Qgis.Critical,
                )
                install_failed = True
                install_error_msg = error_msg
                last_returncode = result.returncode

        except subprocess.TimeoutExpired:
            _log(f"Installation of {package_spec} timed out", Qgis.Critical)
            install_failed = True
            install_error_msg = f"Installation of {package_name} timed out"
        except Exception as e:
            _log(
                f"Exception during installation of {package_spec}: {str(e)}",
                Qgis.Critical,
            )
            install_failed = True
            install_error_msg = f"Error installing {package_name}: {str(e)[:200]}"

        # CUDA -> CPU fallback
        if install_failed and is_cuda_package:
            _log(
                "CUDA install of {} failed, falling back to CPU...".format(
                    package_name
                ),
                Qgis.Warning,
            )
            if progress_callback:
                progress_callback(
                    pkg_start,
                    "CUDA failed, installing {} (CPU)...".format(package_name),
                )

            cpu_pip_args = [
                "install",
                "--upgrade",
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--prefer-binary",
            ]
            cpu_pip_args.extend(_get_pip_ssl_flags())
            cpu_pip_args.append(package_spec)
            cpu_cmd = [python_path, "-m", "pip"] + cpu_pip_args
            try:
                cpu_result = _run_pip_install(
                    cmd=cpu_cmd,
                    timeout=600,
                    env=env,
                    subprocess_kwargs=subprocess_kwargs,
                    package_name=package_name,
                    package_index=i,
                    total_packages=total_packages,
                    progress_start=pkg_start,
                    progress_end=pkg_end,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    is_cuda=False,
                )
                if cpu_result.returncode == 0:
                    _log(
                        "Successfully installed {} (CPU)".format(package_spec),
                        Qgis.Success,
                    )
                    if progress_callback:
                        progress_callback(
                            pkg_end, "{} installed (CPU)".format(package_name)
                        )
                    install_failed = False
                    _cuda_fell_back = True
                else:
                    cpu_err = cpu_result.stderr or cpu_result.stdout or ""
                    install_error_msg = (
                        "CUDA and CPU install both failed for {}: {}".format(
                            package_name, cpu_err[:200]
                        )
                    )
            except subprocess.TimeoutExpired:
                install_error_msg = "CUDA and CPU install both timed out for {}".format(
                    package_name
                )
            except Exception as e:
                install_error_msg = (
                    "CUDA and CPU install both failed for {}: {}".format(
                        package_name, str(e)[:200]
                    )
                )

        if install_failed:
            _log(
                "pip error output: {}".format(install_error_msg[:500]),
                Qgis.Critical,
            )

            if _is_ssl_error(install_error_msg):
                return False, "Failed to install {}: SSL certificate error".format(
                    package_name
                )
            if _is_proxy_auth_error(install_error_msg):
                return (
                    False,
                    "Failed to install {}: proxy authentication required".format(
                        package_name
                    ),
                )
            if _is_network_error(install_error_msg):
                return False, "Failed to install {}: network error".format(package_name)
            if _is_antivirus_error(install_error_msg):
                return (
                    False,
                    "Failed to install {}: blocked by antivirus or security policy".format(
                        package_name
                    ),
                )
            if last_returncode is not None and _is_windows_process_crash(
                last_returncode
            ):
                return (
                    False,
                    "Failed to install {}: process crashed (code {})".format(
                        package_name, last_returncode
                    ),
                )

            return False, "Failed to install {}: {}".format(
                package_name, install_error_msg[:200]
            )

    if progress_callback:
        progress_callback(100, "All dependencies installed")

    _log("=" * 50, Qgis.Success)
    _log("All dependencies installed successfully!", Qgis.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.Success)
    _log("=" * 50, Qgis.Success)

    if _driver_too_old:
        return True, "All dependencies installed successfully [DRIVER_TOO_OLD]"
    if _cuda_fell_back:
        return True, "All dependencies installed successfully [CUDA_FALLBACK]"
    return True, "All dependencies installed successfully"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _get_verification_timeout(package_name: str) -> int:
    """Get verification timeout for a package.

    Args:
        package_name: The package name.

    Returns:
        Timeout in seconds.
    """
    if package_name == "torch":
        return 120
    elif package_name in (
        "torchvision",
        "geoai-py",
        "segment-geospatial",
        "sam3",
        "deepforest",
    ):
        return 120
    else:
        return 60


def _get_verification_code(package_name: str) -> str:
    """Get verification code that tests the package works.

    Args:
        package_name: The package name.

    Returns:
        Python code string to verify the package.
    """
    if package_name == "torch":
        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    elif package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    elif package_name == "geoai-py":
        return "import geoai; print(geoai.__version__)"
    elif package_name == "segment-geospatial":
        return "import samgeo; print(samgeo.__version__)"
    elif package_name == "sam3":
        return "import sam3; print('ok')"
    elif package_name == "deepforest":
        return "from deepforest import main; print('ok')"
    elif package_name == "omniwatermask":
        return "import omniwatermask; print('ok')"
    else:
        import_name = package_name.replace("-", "_")
        return f"import {import_name}"


def verify_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Verify all required packages are importable in the venv.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.
        progress_callback: Optional function called with (percent, message).

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    total_packages = len(REQUIRED_PACKAGES)
    for i, (package_name, _) in enumerate(REQUIRED_PACKAGES):
        if progress_callback:
            percent = int((i / total_packages) * 100)
            progress_callback(
                percent,
                f"Verifying {package_name}... ({i + 1}/{total_packages})",
            )

        verify_code = _get_verification_code(package_name)
        cmd = [python_path, "-c", verify_code]
        pkg_timeout = _get_verification_timeout(package_name)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=pkg_timeout,
                env=env,
                **subprocess_kwargs,
            )

            if result.returncode != 0:
                error_detail = (
                    result.stderr[:300] if result.stderr else result.stdout[:300]
                )
                _log(
                    "Package {} verification failed: {}".format(
                        package_name, error_detail
                    ),
                    Qgis.Warning,
                )
                return False, "Package {} is broken: {}".format(
                    package_name, error_detail[:200]
                )

        except subprocess.TimeoutExpired:
            _log(
                "Verification of {} timed out ({}s), retrying...".format(
                    package_name, pkg_timeout
                ),
                Qgis.Info,
            )
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=pkg_timeout,
                    env=env,
                    **subprocess_kwargs,
                )
                if result.returncode != 0:
                    error_detail = (
                        result.stderr[:300] if result.stderr else result.stdout[:300]
                    )
                    return False, "Package {} is broken: {}".format(
                        package_name, error_detail[:200]
                    )
            except subprocess.TimeoutExpired:
                return False, "Verification error: {} (timed out)".format(package_name)
            except Exception as e:
                return False, "Verification error: {} ({})".format(
                    package_name, str(e)[:100]
                )

        except Exception as e:
            _log(
                "Failed to verify {}: {}".format(package_name, str(e)),
                Qgis.Warning,
            )
            return False, "Verification error: {}".format(package_name)

    if progress_callback:
        progress_callback(100, "Verification complete")

    _log("Virtual environment verified successfully", Qgis.Success)
    return True, "Virtual environment ready"


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def cleanup_old_venv_directories() -> List[str]:
    """Remove old venv directories that don't match current Python version.

    Returns:
        List of removed directory paths.
    """
    current_venv_name = f"venv_{PYTHON_VERSION}"
    removed = []

    try:
        if not os.path.exists(CACHE_DIR):
            return removed
        for entry in os.listdir(CACHE_DIR):
            entry_cmp = os.path.normcase(entry)
            current_cmp = os.path.normcase(current_venv_name)
            if (
                entry_cmp.startswith(os.path.normcase("venv_py"))
                and entry_cmp != current_cmp
            ):
                old_path = os.path.join(CACHE_DIR, entry)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path)
                        _log(f"Cleaned up old venv: {old_path}", Qgis.Info)
                        removed.append(old_path)
                    except Exception as e:
                        _log(
                            f"Failed to remove old venv {old_path}: {e}",
                            Qgis.Warning,
                        )
    except Exception as e:
        _log(f"Error scanning for old venvs: {e}", Qgis.Warning)

    return removed


# ---------------------------------------------------------------------------
# Quick check & status
# ---------------------------------------------------------------------------


def _quick_check_packages(venv_dir: str = None) -> Tuple[bool, str]:
    """Fast filesystem check that packages exist in site-packages.

    Does NOT spawn subprocesses -- safe for the main thread.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Tuple of (packages_found, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    site_packages = get_venv_site_packages(venv_dir)
    if not os.path.exists(site_packages):
        return False, "site-packages directory not found"

    package_markers = {
        "torch": "torch",
        "torchvision": "torchvision",
        "geoai": "geoai",
    }

    for package_name, dir_name in package_markers.items():
        pkg_dir = os.path.join(site_packages, dir_name)
        if not os.path.exists(pkg_dir):
            _log(
                "Quick check: {} not found at {}".format(package_name, pkg_dir),
                Qgis.Warning,
            )
            return False, "Package {} not found".format(package_name)

    _log(
        "Quick check: all packages found in {}".format(site_packages),
        Qgis.Info,
    )
    return True, "All packages found"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation.

    Performs a quick filesystem check (no subprocess calls).
    Safe to call from the main thread.

    Returns:
        Tuple of (is_ready, message).
    """
    from .python_manager import get_python_full_version, standalone_python_exists

    if not standalone_python_exists():
        # Also check for QGIS Python fallback on Windows
        if sys.platform == "win32" and venv_exists():
            pass  # venv was created with QGIS Python fallback
        else:
            _log("get_venv_status: standalone Python not found", Qgis.Info)
            return False, "Dependencies not installed"

    if not venv_exists():
        _log(f"get_venv_status: venv not found at {VENV_DIR}", Qgis.Info)
        return False, "Virtual environment not configured"

    is_present, msg = _quick_check_packages()
    if is_present:
        stored_hash = _read_deps_hash()
        current_hash = _compute_deps_hash()
        if stored_hash is not None and stored_hash != current_hash:
            _log(
                "get_venv_status: deps hash mismatch "
                "(stored={}, current={})".format(stored_hash, current_hash),
                Qgis.Warning,
            )
            return False, "Dependencies need updating"
        if stored_hash is None:
            _log(
                "get_venv_status: no deps hash file, writing current hash",
                Qgis.Info,
            )
            _write_deps_hash()
        python_version = get_python_full_version()
        _log("get_venv_status: ready (quick check passed)", Qgis.Success)
        return True, "Ready (Python {})".format(python_version)
    else:
        _log(f"get_venv_status: quick check failed: {msg}", Qgis.Warning)
        return False, "Virtual environment incomplete: {}".format(msg)


def remove_venv(venv_dir: str = None) -> Tuple[bool, str]:
    """Remove the virtual environment.

    Args:
        venv_dir: Optional venv directory path. Uses VENV_DIR if None.

    Returns:
        Tuple of (success, message).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not os.path.exists(venv_dir):
        return True, "Virtual environment does not exist"

    try:
        shutil.rmtree(venv_dir)
        _log(f"Removed virtual environment: {venv_dir}", Qgis.Success)
        return True, "Virtual environment removed"
    except Exception as e:
        _log(f"Failed to remove venv: {e}", Qgis.Warning)
        return False, f"Failed to remove venv: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False,
) -> Tuple[bool, str]:
    """Complete installation: download Python + create venv + install packages.

    Progress breakdown:
    - 0-10%: Download Python standalone (~50MB)
    - 10-15%: Create virtual environment
    - 15-95%: Install packages
    - 95-100%: Verify installation

    Args:
        progress_callback: Optional function called with (percent, message).
        cancel_check: Optional function returning True to cancel.
        cuda_enabled: Whether to install CUDA-enabled PyTorch.

    Returns:
        Tuple of (success, message).
    """
    from .python_manager import (
        download_python_standalone,
        get_python_full_version,
        standalone_python_exists,
    )

    _log_system_info()

    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.Warning)

    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.Info)

    # Step 1: Download Python standalone (0-10%)
    if not standalone_python_exists():
        python_version = get_python_full_version()
        _log(f"Downloading Python {python_version} standalone...", Qgis.Info)

        def python_progress(percent, msg):
            if progress_callback:
                progress_callback(int(percent * 0.10), msg)

        success, msg = download_python_standalone(
            progress_callback=python_progress,
            cancel_check=cancel_check,
        )

        if not success:
            if sys.platform == "win32":
                qgis_python = _get_qgis_python()
                if qgis_python:
                    if sys.version_info < (3, 9):
                        py_ver = "{}.{}.{}".format(
                            sys.version_info.major,
                            sys.version_info.minor,
                            sys.version_info.micro,
                        )
                        return (
                            False,
                            "Python {} is too old. Please upgrade QGIS.".format(py_ver),
                        )
                    _log(
                        "Standalone Python download failed, "
                        "falling back to QGIS Python: {}".format(msg),
                        Qgis.Warning,
                    )
                    if progress_callback:
                        progress_callback(10, "Using QGIS Python (fallback)...")
                else:
                    return False, f"Failed to download Python: {msg}"
            else:
                return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed", Qgis.Info)
        if progress_callback:
            progress_callback(10, "Python standalone ready")

    # Step 2: Create venv (10-15%)
    if venv_exists():
        _log("Virtual environment already exists", Qgis.Info)
        if progress_callback:
            progress_callback(15, "Virtual environment ready")
    else:
        success, msg = create_venv(progress_callback=progress_callback)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies (15-95%)
    def deps_progress(percent, msg):
        if progress_callback:
            mapped = 15 + int((percent - 20) * 80 / 80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check,
        cuda_enabled=cuda_enabled,
    )

    if not success:
        return False, msg

    _driver_too_old = "[DRIVER_TOO_OLD]" in msg
    _cuda_fell_back = "[CUDA_FALLBACK]" in msg

    # Step 4: Verify (95-100%)
    def verify_progress(percent: int, msg: str):
        if progress_callback:
            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(progress_callback=verify_progress)

    if not is_valid and cuda_enabled:
        _log(
            "Verification failed with CUDA torch, "
            "falling back to CPU: {}".format(verify_msg),
            Qgis.Warning,
        )
        _reinstall_cpu_torch(VENV_DIR, progress_callback=progress_callback)
        is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
        _cuda_fell_back = True

    # CUDA smoke test
    _cuda_smoke_failed = False
    if is_valid and cuda_enabled:
        if progress_callback:
            progress_callback(99, "Verifying CUDA functionality...")
        cuda_works = _verify_cuda_in_venv(VENV_DIR)
        if cuda_works and _cuda_fell_back:
            _cuda_fell_back = False
        elif not cuda_works and not _cuda_fell_back:
            _log(
                "CUDA smoke test failed after install. Keeping CUDA torch installed "
                "for debugging/manual verification instead of auto-reinstalling CPU torch.",
                Qgis.Warning,
            )
            _cuda_smoke_failed = True

    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    _write_deps_hash()

    if cuda_enabled and not _cuda_fell_back and not _driver_too_old:
        _write_cuda_flag("cuda")
    elif cuda_enabled and _cuda_fell_back:
        _write_cuda_flag("cuda_fallback")
    else:
        _write_cuda_flag("cpu")

    if progress_callback:
        progress_callback(100, "All dependencies installed and verified")

    if _driver_too_old:
        return True, "Virtual environment ready [DRIVER_TOO_OLD]"
    if _cuda_fell_back:
        return True, "Virtual environment ready [CUDA_FALLBACK]"
    if _cuda_smoke_failed:
        return True, "Virtual environment ready [CUDA_VERIFY_FAILED]"
    return True, "Virtual environment ready"
