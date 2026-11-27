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

    If the ``GEOAI_DEVICE`` environment variable is set, returns that device
    directly (for example, ``"cpu"``, ``"cuda"``, or ``"mps"``).
    """
    import torch

    env_device = os.environ.get("GEOAI_DEVICE", "").strip()
    if env_device:
        return torch.device(env_device)

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
