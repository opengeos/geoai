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

### SamGeo Panel (Segment Anything Model)
- **Model Tab**: Load SAM models (SAM1, SAM2, or SAM3) with configurable backend and device settings
- **Text Tab**: Segment objects using text prompts (e.g., "tree", "building", "road")
- **Interactive Tab**: Segment using point prompts (foreground/background) or box prompts drawn on the map
- **Batch Tab**: Process multiple points interactively or from vector files/layers
- **Output Tab**: Save results as raster (GeoTIFF) or vector (GeoPackage, Shapefile) with optional regularization (orthogonalize polygons, filter by minimum area)

### GPU Memory Management
- **Clear GPU Memory**: Release GPU memory and clear CUDA cache for all loaded models

## Requirements

- QGIS 3.28 or later
- Python 3.10+
- PyTorch (with CUDA support for GPU acceleration)
- geoai-py package
- samgeo package (for SamGeo panel)

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
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge qgis segment-geospatial geoai
```

Some SamGeo dependencies are only available on PyPI. Run the following command to install all dependencies:

```bash
pip install -U "segment-geospatial[samgeo3]"
```

It is a bit tricky to install SAM 3 on Windows. Run the following commands on Windows to install SamGeo:

```bash
conda create -n geo python=3.12
conda activate geo
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge qgis segment-geospatial geoai
pip install "segment-geospatial[samgeo3]" triton-windows
```

## Usage

### Moondream Vision-Language Model

1. Click the **Moondream** button in the GeoAI toolbar (or `GeoAI` menu → `Moondream VLM`)
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

1. Click the **Segmentation** button in the GeoAI toolbar (or `GeoAI` menu → `Segmentation`)
2. Use the tabs at the top of the panel to switch between:
   - **Create Training Data**: Select input raster and vector labels, configure tile size and stride, and export tiles to a directory.
   - **Train Model**: Select the images and labels directories, choose model architecture (U-Net, DeepLabV3+, etc.), configure training parameters, and start training.
   - **Run Inference**: Select input raster layer or file, specify the trained model path, configure inference parameters, run inference, and optionally vectorize the results.

### SamGeo Panel (Segment Anything Model)

1. Click the **SamGeo** button in the GeoAI toolbar (or `GeoAI` menu → `SamGeo`)
2. In the **Model** tab:
   - Select the SAM model version (SamGeo3/SAM3, SamGeo2/SAM2, or SamGeo/SAM1)
   - Configure backend (meta or transformers) and device (auto, cuda, cpu)
   - Click "Load Model" to initialize the model
   - Select a raster layer or browse for an image file and click "Set Image"
3. Choose a segmentation method:
   - **Text Tab**: Enter text prompts describing objects to segment (e.g., "tree, building")
   - **Interactive Tab**:
     - Click "Add Foreground Points" or "Add Background Points" and click on the map
     - Or click "Draw Box" and drag a rectangle on the map
     - Click "Segment by Points" or "Segment by Box"
   - **Batch Tab**: Add multiple points interactively or load from a vector file/layer
4. In the **Output** tab:
   - Select output format (Raster GeoTIFF, Vector GeoPackage, or Vector Shapefile)
   - For vector output, optionally enable regularization:
     - Check "Regularize polygons (orthogonalize)"
     - Set Epsilon (simplification tolerance) and Min Area (filter small polygons)
   - Click "Save Masks" to export results

### Clear GPU Memory

Click the **GPU** button in the GeoAI toolbar to release GPU memory from all loaded models (Moondream, SamGeo, etc.) and clear CUDA cache.

## Supported Model Architectures (Segmentation)

- U-Net
- U-Net++
- DeepLabV3
- DeepLabV3+
- FPN (Feature Pyramid Network)
- PSPNet
- LinkNet
- MANet
- SegFormer

## Supported Encoders (Segmentation)

- ResNet (34, 50, 101, 152)
- EfficientNet (b0-b4)
- MobileNetV2
- VGG (16, 19)

## Supported SAM Models (SamGeo)

- **SamGeo3 (SAM3)**: Latest version with text prompts, point prompts, and box prompts
- **SamGeo2 (SAM2)**: Improved version with better performance
- **SamGeo (SAM1)**: Original Segment Anything Model

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

- [GeoAI Documentation](https://opengeoai.org)
- [SamGeo Documentation](https://samgeo.gishub.org)
- [GitHub Repository](https://github.com/opengeos/geoai)
- [Report Issues](https://github.com/opengeos/geoai/issues)
