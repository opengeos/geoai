"""Device and environment utilities."""

import os
from typing import List, Union

import torch

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

    if isinstance(package, str):
        packages = [package]
    elif isinstance(package, list):
        packages = package
    else:
        raise ValueError("The package argument must be a string or a list of strings.")

    for package in packages:
        if package.startswith("https"):
            package = f"git+{package}"

        # Execute pip install command and show output in real-time
        command = f"pip install {package}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.decode("utf-8").strip())

        # Wait for process to complete
        process.wait()


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


def get_device() -> torch.device:
    """
    Returns the best available device for deep learning in the order:
    CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def empty_cache() -> None:
    """Empty the cache of the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
