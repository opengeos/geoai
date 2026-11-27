#!/usr/bin/env python3
"""
Cross-platform installation script for GeoAI QGIS Plugin.

Usage:
    python install.py          # Install the plugin
    python install.py --remove # Remove the plugin
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List


def get_qgis_settings_root() -> Path:
    """Get the per-user QGIS settings root for the current platform.

    Returns:
        Path to the directory that holds the ``QGIS3``/``QGIS4`` folders.

    Raises:
        RuntimeError: If the platform is not supported.
    """
    home = Path.home()

    if sys.platform.startswith("linux"):
        return home / ".local/share/QGIS"
    if sys.platform == "darwin":
        return home / "Library/Application Support/QGIS"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "QGIS"
        return home / "AppData/Roaming/QGIS"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_qgis_plugin_dirs(profile: str = "default") -> List[Path]:
    """Get the plugin directories of every installed QGIS major version.

    QGIS 3 keeps its settings under ``QGIS/QGIS3`` and QGIS 4 under
    ``QGIS/QGIS4``. The plugin is installed for each version found; when
    neither exists yet, the QGIS 3 location is used.

    Args:
        profile: QGIS user profile name.

    Returns:
        List of plugin directories.
    """
    root = get_qgis_settings_root()
    suffix = Path("profiles") / profile / "python" / "plugins"
    found = [
        root / version / suffix
        for version in ("QGIS4", "QGIS3")
        if (root / version).is_dir()
    ]
    return found or [root / "QGIS3" / suffix]


def get_qgis_plugin_dir() -> Path:
    """Get the QGIS plugin directory based on the current platform.

    Returns:
        Path to the QGIS plugins directory of the newest installed QGIS.
    """
    return get_qgis_plugin_dirs()[0]


def install_plugin(source_dir: Path, plugin_dir: Path) -> bool:
    """Install the plugin to the QGIS plugins directory.

    Args:
        source_dir: Path to the plugin source directory.
        plugin_dir: Path to the QGIS plugins directory.

    Returns:
        True if installation was successful, False otherwise.
    """
    target_dir = plugin_dir / "geoai"

    # Create plugin directory if it doesn't exist
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing installation
    if target_dir.exists():
        print(f"Removing existing installation: {target_dir}")
        shutil.rmtree(target_dir)

    # Copy plugin
    print(f"Installing GeoAI plugin to: {plugin_dir}")
    shutil.copytree(source_dir, target_dir)

    return True


def remove_plugin(plugin_dir: Path) -> bool:
    """Remove the plugin from the QGIS plugins directory.

    Args:
        plugin_dir: Path to the QGIS plugins directory.

    Returns:
        True if removal was successful, False otherwise.
    """
    target_dir = plugin_dir / "geoai"

    if target_dir.exists():
        print(f"Removing plugin: {target_dir}")
        shutil.rmtree(target_dir)
        print("Plugin removed successfully.")
        return True
    else:
        print("Plugin not found. Nothing to remove.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install or remove GeoAI QGIS Plugin")
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the plugin instead of installing",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="QGIS user profile name (default: default)",
    )
    parser.add_argument(
        "--plugin-dir",
        type=str,
        default=None,
        help="Custom QGIS plugin directory (optional)",
    )
    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    source_dir = script_dir / "geoai"

    if not source_dir.exists():
        print(f"Error: Plugin source directory not found: {source_dir}")
        sys.exit(1)

    # Get plugin directories
    if args.plugin_dir:
        plugin_dirs = [Path(args.plugin_dir)]
    else:
        try:
            plugin_dirs = get_qgis_plugin_dirs(args.profile)
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Please specify the plugin directory with --plugin-dir")
            sys.exit(1)

    print(f"Platform: {sys.platform}")
    for plugin_dir in plugin_dirs:
        print(f"Plugin directory: {plugin_dir}")
    print()

    if args.remove:
        results = [remove_plugin(plugin_dir) for plugin_dir in plugin_dirs]
        success = any(results)
    else:
        success = all(
            install_plugin(source_dir, plugin_dir) for plugin_dir in plugin_dirs
        )

        if success:
            print()
            print("=" * 60)
            print("Installation complete!")
            print("=" * 60)
            print()
            print("To use the plugin:")
            print("  1. Restart QGIS")
            print("  2. Go to Plugins -> Manage and Install Plugins...")
            print("  3. Enable 'GeoAI'")
            print()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
