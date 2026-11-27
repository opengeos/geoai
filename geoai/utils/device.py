"""Device and environment utilities."""

import logging
import os
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "get_device",
    "empty_cache",
    "install_package",
    "temp_file_path",
]


def install_package(package: Union[str, List[str]]) -> None:
    """Install a Python package.

    Args:
        package (str | list): The package name or a GitHub URL or a list of package names or GitHub URLs.
    """
    import subprocess
    import sys

    if isinstance(package, str):
        packages = [package]
    elif isinstance(package, list):
        packages = package
    else:
        raise ValueError("The package argument must be a string or a list of strings.")

    for pkg in packages:
        if not isinstance(pkg, str) or not pkg.strip():
            raise ValueError("Each package name must be a non-empty string.")
        pkg = pkg.strip()
        if pkg.startswith(("http://", "https://")):
            pkg = f"git+{pkg}"

        # Execute pip install using the active interpreter without shell/whitespace splitting
        cmd = [sys.executable, "-m", "pip", "install", "--", pkg]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            if process.stdout is not None:
                while True:
                    output = process.stdout.readline()
                    if output == b"" and process.poll() is not None:
                        break
                    if output:
                        logger.info(output.decode("utf-8", errors="replace").strip())
            returncode = process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)


def temp_file_path(ext: str) -> str:
    """Returns a temporary file path.

    Args:
        ext (str): The file extension.

    Returns:
        str: The temporary file path.
    """

    import tempfile
    import uuid

    if not ext.startswith("."):
        ext = "." + ext
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{ext}")

    return file_path


def get_device() -> "torch.device":
    """
    Returns the best available device for deep learning in the order:
    CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.

    Environment Variables:
        GEOAI_DEVICE: Optional device override string accepted by
            ``torch.device`` (for example, ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
            ``"mps"``, or ``"xpu"``). When set, the requested device is
            validated for syntax and accelerator availability.

    Returns:
        torch.device: The selected PyTorch device.

    Raises:
        ValueError: If ``GEOAI_DEVICE`` is not a valid PyTorch device string
            or requests an accelerator (``cuda``, ``mps``, or ``xpu``) that is
            not available in the current environment.
    """
    import torch

    env_device = os.environ.get("GEOAI_DEVICE", "").strip()
    if env_device:
        try:
            device = torch.device(env_device)
        except (RuntimeError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid GEOAI_DEVICE value {env_device!r}: {e}") from e

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    f"GEOAI_DEVICE={env_device!r} requested CUDA, "
                    "but CUDA is not available."
                )
            if device.index is not None:
                device_count = torch.cuda.device_count()
                if device.index < 0 or (
                    device_count > 0 and device.index >= device_count
                ):
                    raise ValueError(
                        f"GEOAI_DEVICE={env_device!r} requested CUDA device "
                        f"index {device.index}, but only {device_count} "
                        "CUDA device(s) are available."
                    )
        elif device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if not (mps_backend and mps_backend.is_available()):
                raise ValueError(
                    f"GEOAI_DEVICE={env_device!r} requested MPS, "
                    "but MPS is not available."
                )
        elif device.type == "xpu":
            xpu_mod = getattr(torch, "xpu", None)
            if not (xpu_mod and xpu_mod.is_available()):
                raise ValueError(
                    f"GEOAI_DEVICE={env_device!r} requested XPU, "
                    "but XPU is not available."
                )
            if device.index is not None and hasattr(xpu_mod, "device_count"):
                device_count = xpu_mod.device_count()
                if device.index < 0 or (
                    device_count > 0 and device.index >= device_count
                ):
                    raise ValueError(
                        f"GEOAI_DEVICE={env_device!r} requested XPU device "
                        f"index {device.index}, but only {device_count} "
                        "XPU device(s) are available."
                    )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def empty_cache() -> None:
    """Empty the cache of the current device."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
