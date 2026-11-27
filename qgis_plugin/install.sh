#!/bin/bash
# Installation script for GeoAI QGIS Plugin
#
# Installs the plugin for every QGIS major version found (QGIS 3 keeps its
# settings under QGIS/QGIS3, QGIS 4 under QGIS/QGIS4). Set QGIS_PROFILE to
# target a profile other than "default".

PROFILE="${QGIS_PROFILE:-default}"

# Detect the QGIS settings root based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    QGIS_ROOT="$HOME/.local/share/QGIS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    QGIS_ROOT="$HOME/Library/Application Support/QGIS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    QGIS_ROOT="$APPDATA/QGIS"
else
    echo "Unknown OS type: $OSTYPE"
    echo "Please manually copy the geoai folder to your QGIS plugins directory."
    exit 1
fi

PLUGIN_DIRS=()
for version in QGIS4 QGIS3; do
    if [ -d "$QGIS_ROOT/$version" ]; then
        PLUGIN_DIRS+=("$QGIS_ROOT/$version/profiles/$PROFILE/python/plugins")
    fi
done
if [ ${#PLUGIN_DIRS[@]} -eq 0 ]; then
    PLUGIN_DIRS=("$QGIS_ROOT/QGIS3/profiles/$PROFILE/python/plugins")
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for PLUGIN_DIR in "${PLUGIN_DIRS[@]}"; do
    # Create plugin directory if it doesn't exist
    mkdir -p "$PLUGIN_DIR"

    # Remove existing installation
    if [ -d "$PLUGIN_DIR/geoai" ]; then
        echo "Removing existing installation from $PLUGIN_DIR..."
        rm -rf "$PLUGIN_DIR/geoai"
    fi

    # Copy plugin
    echo "Installing GeoAI plugin to: $PLUGIN_DIR"
    cp -r "$SCRIPT_DIR/geoai" "$PLUGIN_DIR/"
done

echo ""
echo "Installation complete!"
echo ""
echo "To use the plugin:"
echo "1. Restart QGIS"
echo "2. Go to Plugins -> Manage and Install Plugins..."
echo "3. Enable 'GeoAI'"
echo ""
