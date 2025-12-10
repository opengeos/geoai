#!/bin/bash
# Installation script for GeoAI QGIS Plugin

# Detect QGIS plugin directory based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLUGIN_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLUGIN_DIR="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLUGIN_DIR="$APPDATA/QGIS/QGIS3/profiles/default/python/plugins"
else
    echo "Unknown OS type: $OSTYPE"
    echo "Please manually copy the geoai_plugin folder to your QGIS plugins directory."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create plugin directory if it doesn't exist
mkdir -p "$PLUGIN_DIR"

# Remove existing installation
if [ -d "$PLUGIN_DIR/geoai_plugin" ]; then
    echo "Removing existing installation..."
    rm -rf "$PLUGIN_DIR/geoai_plugin"
fi

# Copy plugin
echo "Installing GeoAI plugin to: $PLUGIN_DIR"
cp -r "$SCRIPT_DIR/geoai_plugin" "$PLUGIN_DIR/"

echo ""
echo "Installation complete!"
echo ""
echo "To use the plugin:"
echo "1. Restart QGIS"
echo "2. Go to Plugins -> Manage and Install Plugins..."
echo "3. Enable 'GeoAI'"
echo ""
echo "Make sure you have the required Python packages installed:"
echo "  pip install geoai-py torch torchvision"
