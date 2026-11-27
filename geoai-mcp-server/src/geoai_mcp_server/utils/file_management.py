"""File management utilities for GeoAI MCP Server.

Provides secure file access within designated workspace directories.
Prevents directory traversal attacks and unauthorized file access.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any
import logging

from ..config import get_config
from .error_handling import FileAccessError, InputValidationError

logger = logging.getLogger("geoai_mcp.file_management")


def _is_path_within_directory(path: Path, directory: Path) -> bool:
    """Check if a path is within a directory (prevents traversal attacks).

    Args:
        path: Path to check
        directory: Directory that should contain the path

    Returns:
        True if path is within directory
    """
    try:
        resolved_path = path.resolve()
        resolved_dir = directory.resolve()
        # Use is_relative_to() for proper path containment check
        # This prevents prefix-matching bypasses (e.g., /safe vs /safe_evil)
        return resolved_path.is_relative_to(resolved_dir)
    except (ValueError, Exception):
        # is_relative_to raises ValueError if paths are not relative
        return False


def validate_input_path(path: str | Path) -> Path:
    """Validate that a path is within the allowed input directory.

    Args:
        path: Path to validate (relative or absolute)

    Returns:
        Validated absolute Path

    Raises:
        FileAccessError: If path is outside input directory or doesn't exist
    """
    config = get_config()
    path = Path(path)

    # If relative, make it relative to input directory
    if not path.is_absolute():
        full_path = config.input_dir / path
    else:
        full_path = path

    full_path = full_path.resolve()

    # Security check: ensure path is within input directory
    if not _is_path_within_directory(full_path, config.input_dir):
        logger.warning(f"Path traversal attempt blocked: {path}")
        raise FileAccessError(
            message=f"Access denied: path is outside the allowed input directory",
            file_path=str(path),
            operation="access",
        )

    # Check existence
    if not full_path.exists():
        raise FileAccessError(
            message=f"File not found: {path}", file_path=str(path), operation="read"
        )

    return full_path


def validate_output_path(path: str | Path, config: Optional[Any] = None) -> Path:
    """Validate that a path is strictly within the allowed output directory.

    Args:
        path: Path to validate (relative or absolute)
        config: Optional configuration object (uses global config if not provided)

    Returns:
        Validated absolute Path

    Raises:
        FileAccessError: If path is outside output directory or equals output directory
    """
    if config is None:
        config = get_config()
    path = Path(path)

    # If relative, make it relative to output directory
    if not path.is_absolute():
        full_path = config.output_dir / path
    else:
        full_path = path

    full_path = full_path.resolve()
    output_dir_resolved = config.output_dir.resolve()

    # Ensure path is strictly within output directory
    if (
        not _is_path_within_directory(full_path, output_dir_resolved)
        or full_path == output_dir_resolved
    ):
        logger.warning(f"Output path validation blocked: {path}")
        raise FileAccessError(
            message="Access denied: path must be strictly inside the allowed output directory",
            file_path=str(path),
            operation="access",
        )

    # Ensure parent directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def validate_output_subdir(subdir: str | Path, config: Optional[Any] = None) -> Path:
    """Validate that a relative subdirectory path stays strictly within output_dir.

    Args:
        subdir: Relative subdirectory path within the output directory.
        config: Optional configuration object (uses global config if not provided).

    Returns:
        Validated absolute Path for the subdirectory.

    Raises:
        InputValidationError: If subdir is empty, absolute, or contains '..'.
        FileAccessError: If the resolved path is outside or equal to output_dir.
    """
    if config is None:
        config = get_config()

    subdir_str = str(subdir)
    if not subdir_str.strip() or "\x00" in subdir_str:
        raise InputValidationError(
            message="Output subdirectory must be a non-empty relative path",
            parameter_name="output_subdir",
            received_value=subdir_str,
            expected="A relative subdirectory path within the output directory",
        )

    subdir_path = Path(subdir_str)
    if subdir_path.is_absolute() or subdir_str.startswith(("/", "\\")):
        raise InputValidationError(
            message="Output subdirectory must be a relative path",
            parameter_name="output_subdir",
            received_value=subdir_str,
            expected="A relative subdirectory path within the output directory",
        )

    if ".." in subdir_path.parts:
        raise InputValidationError(
            message="Output subdirectory cannot contain '..'",
            parameter_name="output_subdir",
            received_value=subdir_str,
            expected="A relative subdirectory path without '..'",
        )

    output_dir_resolved = config.output_dir.resolve()
    full_path = (output_dir_resolved / subdir_path).resolve()

    if (
        not _is_path_within_directory(full_path, output_dir_resolved)
        or full_path == output_dir_resolved
    ):
        logger.warning(f"Output subdirectory validation blocked: {subdir}")
        raise FileAccessError(
            message="Access denied: output subdirectory must be strictly inside the allowed output directory",
            file_path=subdir_str,
            operation="access",
        )

    return full_path


def get_safe_file_path(
    filename: str, source: str = "input", config: Optional[Any] = None
) -> Path:
    """Get a safe path for reading a file from either the input or output directory.

    Args:
        filename: Filename or relative path within the selected workspace directory.
        source: Workspace directory to resolve against ("input" or "output").
        config: Optional configuration object (uses global config if not provided).

    Returns:
        Safe absolute path to the existing file.

    Raises:
        InputValidationError: If source is invalid or filename contains traversal.
        FileAccessError: If the resolved path escapes the directory or does not exist.
    """
    if source not in ("input", "output"):
        raise InputValidationError(
            message=f"Invalid source directory '{source}'. Use 'input' or 'output'.",
            parameter_name="source",
            received_value=source,
            expected="'input' or 'output'",
        )

    if not filename or "\x00" in filename or ".." in Path(filename).parts:
        raise InputValidationError(
            message="Filename cannot be empty or contain '..'",
            parameter_name="filename",
            received_value=filename,
            expected="A relative filename without directory traversal",
        )

    if config is None:
        config = get_config()

    base_dir = (config.input_dir if source == "input" else config.output_dir).resolve()
    full_path = (base_dir / filename).resolve()

    if not _is_path_within_directory(full_path, base_dir):
        logger.warning(f"Path traversal attempt blocked ({source}): {filename}")
        raise FileAccessError(
            message=f"Path is outside allowed {source} directory",
            file_path=filename,
            operation="read",
        )

    if not full_path.exists():
        raise FileAccessError(
            message=f"File not found in {source} directory: {filename}",
            file_path=filename,
            operation="read",
        )

    return full_path


def get_safe_input_path(filename: str, config: Optional[Any] = None) -> Path:
    """Get a safe path for reading an input file.

    Args:
        filename: Filename or relative path within input directory
        config: Optional configuration object (uses global config if not provided)

    Returns:
        Safe absolute path to the file

    Raises:
        InputValidationError: If filename contains path traversal
        FileAccessError: If file doesn't exist
    """
    return get_safe_file_path(filename, source="input", config=config)


def get_safe_output_path(
    filename: str, create_subdirs: bool = True, config: Optional[Any] = None
) -> Path:
    """Get a safe path for writing an output file.

    Args:
        filename: Filename or relative path within output directory
        create_subdirs: Whether to create subdirectories if needed
        config: Optional configuration object (uses global config if not provided)

    Returns:
        Safe absolute path for the file

    Raises:
        InputValidationError: If filename contains path traversal
        FileAccessError: If resolved path is outside or equal to output directory
    """
    # Check for traversal attempts
    if ".." in filename:
        raise InputValidationError(
            message="Filename cannot contain '..'",
            parameter_name="filename",
            received_value=filename,
            expected="A filename without directory traversal",
        )

    if config is None:
        config = get_config()
    output_dir_resolved = config.output_dir.resolve()
    full_path = (output_dir_resolved / filename).resolve()

    # Double-check the path is strictly within output dir
    if (
        not _is_path_within_directory(full_path, output_dir_resolved)
        or full_path == output_dir_resolved
    ):
        raise FileAccessError(
            message="Output path must be strictly inside the allowed directory",
            file_path=filename,
            operation="write",
        )

    if create_subdirs:
        full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def list_input_files(
    base_dir: Optional[Path] = None,
    pattern: str = "*",
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
    max_files: int = 500,
) -> List[Path]:
    """List files in a directory.

    Args:
        base_dir: Directory to search (default: input directory from config)
        pattern: Glob pattern to match (default: all files)
        extensions: Filter by extensions (e.g., ['.tif', '.tiff'])
        recursive: Whether to search subdirectories
        max_files: Maximum number of matching files to return (default: 500)

    Returns:
        List of Path objects for matching files

    Raises:
        InputValidationError: If pattern contains path traversal or '**' wildcards
    """
    # Security: reject patterns that could escape the base directory or cause glob DoS
    if ".." in pattern or pattern.startswith("/") or pattern.startswith("\\"):
        raise InputValidationError(
            message="Pattern cannot contain '..' or absolute path prefixes",
            parameter_name="pattern",
            received_value=pattern,
            expected="A safe glob pattern without directory traversal",
        )
    if "**" in pattern:
        raise InputValidationError(
            message="Pattern cannot contain '**' recursive wildcards",
            parameter_name="pattern",
            received_value=pattern,
            expected="A non-recursive glob pattern without '**'",
        )

    if base_dir is None:
        config = get_config()
        base_dir = config.input_dir

    base_dir = base_dir.resolve()
    if not base_dir.exists():
        return []

    normalized_exts = None
    if extensions:
        normalized_exts = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }

    iterator = base_dir.rglob(pattern) if recursive else base_dir.glob(pattern)

    matched: List[Path] = []
    for candidate in iterator:
        if not _is_path_within_directory(candidate, base_dir):
            continue
        if not candidate.is_file():
            continue
        if (
            normalized_exts is not None
            and candidate.suffix.lower() not in normalized_exts
        ):
            continue
        matched.append(candidate)
        if max_files > 0 and len(matched) >= max_files:
            break

    return sorted(matched)


def generate_output_filename(
    base_name: str,
    operation: str,
    extension: str = ".tif",
    add_timestamp: bool = True,
) -> str:
    """Generate a unique output filename.

    Args:
        base_name: Base name for the file (usually from input)
        operation: Operation name to include (e.g., 'segmentation')
        extension: File extension (including dot)
        add_timestamp: Whether to add timestamp for uniqueness

    Returns:
        Generated filename
    """
    # Remove extension from base name
    base_name = Path(base_name).stem

    # Sanitize base name
    base_name = re.sub(r"[^\w\-]", "_", base_name)

    # Build filename
    parts = [base_name, operation]

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)

    # Ensure extension starts with dot
    if not extension.startswith("."):
        extension = f".{extension}"

    return "_".join(parts) + extension
