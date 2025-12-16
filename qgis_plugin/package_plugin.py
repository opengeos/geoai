#!/usr/bin/env python3
"""
Package the GeoAI QGIS plugin for upload to the official QGIS plugin repository.

This script creates a zip file with the following characteristics:
- Root folder renamed from 'geoai_plugin' to 'geoai'
- Excludes generated files (ui_*.py, resources_rc.py, etc.)
- Excludes __MACOSX, .git, __pycache__, and other hidden directories
- Excludes .pyc files and other compiled Python files

Usage:
    python package_plugin.py [--output OUTPUT_PATH]

Examples:
    python package_plugin.py
    python package_plugin.py --output /path/to/output/geoai.zip
"""

import argparse
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path


# Patterns to exclude from the zip
EXCLUDE_PATTERNS = [
    # Generated UI files from Qt Designer
    r"^ui_.*\.py$",
    # Generated resource files
    r"^resources_rc\.py$",
    r"^.*_rc\.py$",
    # Compiled Python files
    r"^.*\.pyc$",
    r"^.*\.pyo$",
    # Backup files
    r"^.*\.bak$",
    r"^.*~$",
    # IDE/editor files
    r"^\..*\.swp$",
    r"^.*\.orig$",
]

# Directory names to exclude
EXCLUDE_DIRS = {
    "__pycache__",
    "__MACOSX",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
    "build",
    "dist",
    "node_modules",
    "help",  # Generated help files
}


def should_exclude_file(filename: str) -> bool:
    """Check if a file should be excluded based on its name."""
    # Check against exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, filename):
            return True
    return False


def should_exclude_dir(dirname: str) -> bool:
    """Check if a directory should be excluded."""
    # Exclude hidden directories (starting with .)
    if dirname.startswith("."):
        return True
    # Check against exclude directory names
    if dirname in EXCLUDE_DIRS:
        return True
    # Check for egg-info directories
    if dirname.endswith(".egg-info"):
        return True
    return False


def get_version_from_metadata(plugin_dir: Path) -> str:
    """Extract version from metadata.txt file."""
    metadata_file = plugin_dir / "metadata.txt"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    return "unknown"


def package_plugin(
    source_dir: Path,
    output_path: Path | None = None,
    target_name: str = "geoai",
    include_version: bool = True,
) -> Path:
    """
    Package the QGIS plugin into a zip file.

    Args:
        source_dir: Path to the geoai_plugin directory
        output_path: Optional path for the output zip file
        target_name: Name for the root folder in the zip (default: 'geoai')
        include_version: Whether to include version in the zip filename

    Returns:
        Path to the created zip file
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    # Get version from metadata
    version = get_version_from_metadata(source_dir)

    # Determine output path
    if output_path is None:
        if include_version:
            zip_name = f"{target_name}-{version}.zip"
        else:
            zip_name = f"{target_name}.zip"
        output_path = source_dir.parent / zip_name

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing zip if it exists
    if output_path.exists():
        output_path.unlink()

    print(f"Packaging plugin from: {source_dir}")
    print(f"Output zip file: {output_path}")
    print(f"Root folder name in zip: {target_name}")
    print(f"Plugin version: {version}")
    print()

    files_added = 0
    files_excluded = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Filter out excluded directories (modify dirs in-place)
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]

            for file in files:
                file_path = Path(root) / file

                # Check if file should be excluded
                if should_exclude_file(file):
                    print(f"  Excluding: {file_path.relative_to(source_dir)}")
                    files_excluded += 1
                    continue

                # Skip hidden files
                if file.startswith("."):
                    print(f"  Excluding hidden: {file_path.relative_to(source_dir)}")
                    files_excluded += 1
                    continue

                # Calculate the archive name (rename root folder)
                rel_path = file_path.relative_to(source_dir)
                archive_name = Path(target_name) / rel_path

                # Add file to zip
                zipf.write(file_path, archive_name)
                print(f"  Adding: {archive_name}")
                files_added += 1

    print()
    print(f"Package created successfully!")
    print(f"  Files added: {files_added}")
    print(f"  Files excluded: {files_excluded}")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def verify_zip(zip_path: Path) -> None:
    """Verify the contents of the created zip file."""
    print()
    print("Verifying zip contents:")
    print("-" * 50)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        # Check for any problematic entries
        has_issues = False
        for name in zipf.namelist():
            # Check for unwanted patterns
            basename = os.path.basename(name)
            dirname = os.path.dirname(name)

            if "__pycache__" in name:
                print(f"  WARNING: Found __pycache__: {name}")
                has_issues = True
            if "__MACOSX" in name:
                print(f"  WARNING: Found __MACOSX: {name}")
                has_issues = True
            if ".git" in name.split("/"):
                print(f"  WARNING: Found .git: {name}")
                has_issues = True
            if basename.startswith("ui_") and basename.endswith(".py"):
                print(f"  WARNING: Found ui_*.py: {name}")
                has_issues = True
            if basename == "resources_rc.py":
                print(f"  WARNING: Found resources_rc.py: {name}")
                has_issues = True

        if not has_issues:
            print("  All checks passed!")

        # List contents
        print()
        print("Zip contents:")
        print("-" * 50)
        for name in sorted(zipf.namelist()):
            info = zipf.getinfo(name)
            if not name.endswith("/"):  # Skip directories
                print(f"  {name} ({info.file_size} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Package GeoAI QGIS plugin for repository upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for the zip file (default: geoai-{version}.zip in qgis_plugin folder)",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=None,
        help="Source directory (default: geoai_plugin in the same folder as this script)",
    )
    parser.add_argument(
        "--no-version",
        action="store_true",
        help="Don't include version in the zip filename",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of the created zip",
    )

    args = parser.parse_args()

    # Determine source directory
    script_dir = Path(__file__).parent.resolve()
    source_dir = args.source if args.source else script_dir / "geoai_plugin"

    try:
        # Create the package
        zip_path = package_plugin(
            source_dir=source_dir,
            output_path=args.output,
            target_name="geoai",
            include_version=not args.no_version,
        )

        # Verify the package
        if not args.no_verify:
            verify_zip(zip_path)

        print()
        print("=" * 50)
        print("Plugin packaged successfully!")
        print(f"Upload this file to the QGIS plugin repository:")
        print(f"  {zip_path}")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
