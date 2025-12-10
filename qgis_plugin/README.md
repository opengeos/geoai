# GeoAI Plugin for QGIS

A QGIS plugin providing AI-powered geospatial analysis tools from the [geoai](https://github.com/opengeos/geoai) package.

## Features

The plugin provides **dockable panels** that can be attached to the left or right side of QGIS.

### Moondream Vision-Language Model Panel
- **Caption**: Generate descriptions of geospatial imagery (short, normal, or long)
- **Query**: Ask questions about images using natural language
- **Detect**: Detect and locate objects with bounding boxes
- **Point**: Locate specific objects with point markers

### Segmentation Panel (Combined Training & Inference)
- **Tab 1 - Create Training Data**: Export GeoTIFF tiles from raster and vector data
- **Tab 2 - Train Models**: Train custom segmentation models (U-Net, DeepLabV3+, FPN, etc.)
- **Tab 3 - Run Inference**: Apply trained models to new imagery and vectorize results

## Requirements

- QGIS 3.28 or later
- Python 3.10+
- PyTorch (with CUDA support for GPU acceleration)
- geoai-py package

## Installation

### Option 1: Install from Plugin Manager (Recommended)

1. Open QGIS
2. Go to `Plugins` → `Manage and Install Plugins...`
3. Search for "GeoAI"
4. Click "Install Plugin"

### Option 2: Install Using Script (Recommended for Developers)

1. Clone or download this repository
2. Run the installation script:

   **Linux/macOS:**
   ```bash
   python install.py
   ```

   **Windows:**
   ```cmd
   python install.py
   ```

3. Restart QGIS
4. Enable the plugin in `Plugins` → `Manage and Install Plugins...`

To remove the plugin:
```bash
python install.py --remove
```

### Option 3: Manual Installation

1. Copy the `geoai_plugin` folder to your QGIS plugins directory:
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS
3. Enable the plugin in `Plugins` → `Manage and Install Plugins...`

### Install Dependencies

Before using the plugin, install the required Python packages in your QGIS Python environment:

```bash
# Activate your QGIS Python environment (e.g., conda environment)
conda activate geo

# Install geoai and dependencies
pip install geoai-py torch torchvision
```

## Usage

### Moondream Vision-Language Model

1. Click `GeoAI` → `Moondream Vision-Language Model`
2. Load a Moondream model (default: vikhyatk/moondream2)
3. Select a raster layer or browse for an image file
4. Choose a mode:
   - **Caption**: Generate a description of the image
   - **Query**: Ask a question about the image
   - **Detect**: Detect objects by type (e.g., "building", "car")
   - **Point**: Locate specific objects
5. Click "Run Analysis"
6. Results are displayed and optionally added to the map

### Segmentation Panel (Create Data, Train, Inference)

1. Click `GeoAI` → `Segmentation` to open the Segmentation panel.
2. Use the tabs at the top of the panel to switch between:
   - **Create Training Data**: Select input raster and vector labels, configure tile size and stride, and export tiles to a directory.
   - **Train Model**: Select the images and labels directories, choose model architecture (U-Net, DeepLabV3+, etc.), configure training parameters, and start training.
   - **Run Inference**: Select input raster layer or file, specify the trained model path, configure inference parameters, run inference, and optionally vectorize the results.
## Supported Model Architectures

- U-Net
- U-Net++
- DeepLabV3
- DeepLabV3+
- FPN (Feature Pyramid Network)
- PSPNet
- LinkNet
- MANet
- SegFormer

## Supported Encoders

- ResNet (34, 50, 101, 152)
- EfficientNet (b0-b4)
- MobileNetV2
- VGG (16, 19)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

- [GeoAI Documentation](https://opengeoai.org)
- [GitHub Repository](https://github.com/opengeos/geoai)
- [Report Issues](https://github.com/opengeos/geoai/issues)
