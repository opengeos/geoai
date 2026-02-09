"""File management utilities for GeoAI MCP Server.

Provides secure file access within designated workspace directories.
Prevents directory traversal attacks and unauthorized file access.
"""

import os
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
        return str(resolved_path).startswith(str(resolved_dir))
    except Exception:
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
            operation="access"
        )

    # Check existence
    if not full_path.exists():
        raise FileAccessError(
            message=f"File not found: {path}",
            file_path=str(path),
            operation="read"
        )

    return full_path


def validate_output_path(path: str | Path, config: Optional[Any] = None) -> Path:
    """Validate that a path is within the allowed output directory.

    Args:
        path: Path to validate (relative or absolute)
        config: Optional configuration object (uses global config if not provided)

    Returns:
        Validated absolute Path

    Raises:
        FileAccessError: If path is outside output directory
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

    # Security check: ensure path is within output directory
    if not _is_path_within_directory(full_path, config.output_dir):
        logger.warning(f"Path traversal attempt blocked: {path}")
        raise FileAccessError(
            message=f"Access denied: path is outside the allowed output directory",
            file_path=str(path),
            operation="access"
        )

    # Ensure parent directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

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
    # Check for traversal attempts
    if ".." in filename:
        raise InputValidationError(
            message="Filename cannot contain '..'",
            parameter_name="filename",
            received_value=filename,
            expected="A filename without directory traversal"
        )

    if config is None:
        config = get_config()
    
    full_path = (config.input_dir / filename).resolve()
    
    # Security check: ensure path is within input directory
    if not _is_path_within_directory(full_path, config.input_dir):
        logger.warning(f"Path traversal attempt blocked: {filename}")
        raise FileAccessError(
            message="Input path is outside allowed directory",
            file_path=filename,
            operation="read"
        )

    if not full_path.exists():
        raise FileAccessError(
            message=f"File not found in input directory: {filename}",
            file_path=filename,
            operation="read"
        )

    return full_path


def get_safe_output_path(filename: str, create_subdirs: bool = True) -> Path:
    """Get a safe path for writing an output file.

    Args:
        filename: Filename or relative path within output directory
        create_subdirs: Whether to create subdirectories if needed

    Returns:
        Safe absolute path for the file

    Raises:
        InputValidationError: If filename contains path traversal
    """
    # Check for traversal attempts
    if ".." in filename:
        raise InputValidationError(
            message="Filename cannot contain '..'",
            parameter_name="filename",
            received_value=filename,
            expected="A filename without directory traversal"
        )

    config = get_config()
    full_path = (config.output_dir / filename).resolve()

    # Double-check the path is still within output dir
    if not _is_path_within_directory(full_path, config.output_dir):
        raise FileAccessError(
            message="Output path is outside allowed directory",
            file_path=filename,
            operation="write"
        )

    if create_subdirs:
        full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def list_input_files(
    base_dir: Optional[Path] = None,
    pattern: str = "*",
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """List files in a directory.

    Args:
        base_dir: Directory to search (default: input directory from config)
        pattern: Glob pattern to match (default: all files)
        extensions: Filter by extensions (e.g., ['.tif', '.tiff'])
        recursive: Whether to search subdirectories

    Returns:
        List of Path objects for matching files
    """
    if base_dir is None:
        config = get_config()
        base_dir = config.input_dir

    if recursive:
        files = list(base_dir.rglob(pattern))
    else:
        files = list(base_dir.glob(pattern))

    # Filter by extensions if specified
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]

    # Return only files (not directories)
    return sorted([f for f in files if f.is_file()])


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
    base_name = re.sub(r'[^\w\-]', '_', base_name)

    # Build filename
    parts = [base_name, operation]

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)

    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = f'.{extension}'

    return "_".join(parts) + extension


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
