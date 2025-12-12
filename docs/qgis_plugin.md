# QGIS Plugin for GeoAI

A QGIS plugin providing AI-powered geospatial analysis tools from the [geoai](https://github.com/opengeos/geoai) package. The plugin bundles several model-driven workflows (Moondream VLM, segmentation training/inference, SamGeo) into dockable panels so you can keep QGIS as the main workspace while experimenting with GeoAI.

## Quick Start

1. Create a fresh conda environment (`conda create -n geo python=3.12`) and install QGIS plus GeoAI dependencies as described below.
2. Install the GeoAI QGIS plugin either from the script (`python install.py`) or by copying the `qgis_plugin` folder into your QGIS profile.
3. Restart QGIS, enable the plugin under `Plugins` → `Manage and Install Plugins...`, and open one of the panels from the GeoAI toolbar to begin experimenting with sample datasets.

[![demo](https://github.com/user-attachments/assets/557feb58-ca58-4e27-800f-f3e8a8d3d362)](https://github.com/user-attachments/assets/557feb58-ca58-4e27-800f-f3e8a8d3d362)

## Features

Each tool lives inside a dockable panel that can be attached to either side of the QGIS interface, so you can keep layers, maps, and models visible simultaneously.

### Moondream Vision-Language Model Panel

-   **Caption**: Generate descriptions of geospatial imagery (short, normal, or long)
-   **Query**: Ask questions about images using natural language
-   **Detect**: Detect and locate objects with bounding boxes
-   **Point**: Locate specific objects with point markers

### Segmentation Panel (Combined Training & Inference)

-   **Tab 1 - Create Training Data**: Export GeoTIFF tiles from raster and vector data
-   **Tab 2 - Train Models**: Train custom segmentation models (U-Net, DeepLabV3+, FPN, etc.)
-   **Tab 3 - Run Inference**: Apply trained models to new imagery and vectorize results. Vector outputs can optionally be smoothed or simplified for immediate use in GIS workflows.

### SamGeo Panel (Segment Anything Model)

-   **Model Tab**: Load SAM models (SAM1, SAM2, or SAM3) with configurable backend and device settings
-   **Text Tab**: Segment objects using text prompts (e.g., "tree", "building", "road")
-   **Interactive Tab**: Segment using point prompts (foreground/background) or box prompts drawn on the map
-   **Batch Tab**: Process multiple points interactively or from vector files/layers
-   **Output Tab**: Save results as raster (GeoTIFF) or vector (GeoPackage, Shapefile) with optional regularization (orthogonalize polygons, filter by minimum area)

### GPU Memory Management

-   **Clear GPU Memory**: Release GPU memory and clear CUDA cache for all loaded models

## Requirements

-   QGIS 3.28 or later
-   Python 3.10+
-   PyTorch (with CUDA support for GPU acceleration)
-   geoai-py package
-   samgeo package (for SamGeo panel)

## Installation

### Install QGIS and Python Packages

Before using the plugin, create a new conda environment dedicated to QGIS and the Python dependencies. Please do not try to reuse an existing QGIS install—mismatched Python packages almost always break the plugin.

It is recommended to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to manage conda environments. Once Miniconda is installed, open the Terminal or Anaconda Prompt and run the following commands to create a new conda environment and install QGIS and Python packages:

```bash
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge qgis segment-geospatial geoai
```

Some SamGeo dependencies are only available on PyPI. Run the following command to install the remaining dependencies:

```bash
pip install -U "segment-geospatial[samgeo3]"
```

It is a bit trickier to install SAM 3 on Windows. Use the following commands to prepare the environment:

```bash
conda create -n geo python=3.12
conda activate geo
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge qgis segment-geospatial geoai
pip install "segment-geospatial[samgeo3]" triton-windows
```

### Install QGIS Plugin

### Option 1: Install from Plugin Manager (Coming soon)

1. Open QGIS
2. Go to `Plugins` → `Manage and Install Plugins...`
3. Search for "GeoAI"
4. Click "Install Plugin"

### Option 2: Install Using Script

1. Clone or download this repository

    ```bash
    git clone https://github.com/opengeos/geoai.git
    ```

2. Run the installation script (from the repo root it will link/copy files to your QGIS profile):

    ```bash
    cd qgis_plugin
    python install.py
    ```

3. Restart QGIS so it picks up the new plugin files.
4. Enable the plugin in `Plugins` → `Manage and Install Plugins...` → Toggle on the `GeoAI` plugin.

To remove the plugin:

```bash
python install.py --remove
```

### Option 3: Manual Installation

1. Copy the `qgis_plugin` folder to your QGIS plugins directory:
    - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
    - **Windows**: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
    - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS
3. Enable the plugin in `Plugins` → `Manage and Install Plugins...` → Toggle on the `GeoAI` plugin. Revisit this dialog any time you update the plugin files.

## Usage

### Moondream Vision-Language Model

Sample dataset: [parking_lot.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/parking_lot.tif)

Steps:

1. Click the **Moondream** button in the GeoAI toolbar (or `GeoAI` menu → `Moondream VLM`)
2. Load a Moondream model (default: vikhyatk/moondream2)
3. Select a raster layer or browse for an image file
4. Choose a mode:
    - **Caption**: Generate a description of the image
    - **Query**: Ask a question about the image
    - **Detect**: Detect objects by type (e.g., "building", "car")
    - **Point**: Locate specific objects
5. Click "Run"
6. Results are displayed and optionally added to the map. You can drag the panel to any side of QGIS to keep it out of the way while browsing results. Save the output table or vector layer if you want to reuse detections later.

    ![moondream](https://github.com/user-attachments/assets/bb800a04-b7c4-4fdd-a628-a48842d7eac5)

### Segmentation Panel (Create Data, Train, Inference)

Sample datasets:

-   [naip_train_buildings.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_train_buildings.tif)
-   [naip_test_buildings.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_test_buildings.tif)
-   [naip_test_buildings.geojson](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_test_buildings.geojson)

Steps:

1. Download the sample datasets (links above) or prepare your own imagery/vector labels. Store them in a folder that is accessible to the conda environment.
2. Click the **Segmentation** button in the GeoAI toolbar (or `GeoAI` menu → `Segmentation`)
3. Use the tabs at the top of the panel to switch between:

    - **Create Training Data**: Select input raster and vector labels, configure tile size and stride, and export tiles to a directory.
    - **Train Model**: Select the images and labels directories, choose model architecture (U-Net, DeepLabV3+, etc.), configure training parameters, and start training.
    - **Run Inference**: Select input raster layer or file, specify the trained model path, configure inference parameters, run inference, and optionally vectorize the results.

    ![data](https://github.com/user-attachments/assets/121fcfa8-6f9b-4413-9419-af666698c053)

    ![train](https://github.com/user-attachments/assets/dfeefb86-ebf7-467c-a5ff-794cde80a7cb)

    ![inference](https://github.com/user-attachments/assets/f0945c01-0fcb-4607-9226-4a3b2bcb05e1)

### SamGeo Panel (Segment Anything Model)

Sample dataset: [uc_berkeley.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif)

Steps:

1. Click the **SamGeo** button in the GeoAI toolbar (or `GeoAI` menu → `SamGeo`)
2. In the **Model** tab:

    - Select the SAM model version (SamGeo3/SAM3, SamGeo2/SAM2, or SamGeo/SAM1)
    - Configure backend (meta or transformers) and device (auto, cuda, cpu)
    - Click "Load Model" to initialize the model
    - Select a raster layer or browse for an image file and click "Set Image"

    ![](https://github.com/user-attachments/assets/600b0879-f851-4423-b668-cb9e8df28425)

3. Choose a segmentation method:

    - **Text Tab**: Enter text prompts describing objects to segment (e.g., "tree, building")

        ![](https://github.com/user-attachments/assets/da2c17fc-4633-488d-ba44-00f1cd97555c)

    - **Interactive Tab**:

        - Click "Add Foreground Points" or "Add Background Points" and click on the map
        - Or click "Draw Box" and drag a rectangle on the map
        - Click "Segment by Points" or "Segment by Box"

        ![](https://github.com/user-attachments/assets/6730737d-62fc-438a-bff5-cffb685d391e)

    - **Batch Tab**: Add multiple points interactively or load from a vector file/layer

        ![](https://github.com/user-attachments/assets/104ec741-44cc-404a-9213-36cf78456171)

4. In the **Output** tab:

    - Select output format (Raster GeoTIFF, Vector GeoPackage, or Vector Shapefile)
    - For vector output, optionally enable regularization:
        - Check "Regularize polygons (orthogonalize)"
        - Set Epsilon (simplification tolerance) and Min Area (filter small polygons)
    - Click "Save Masks" to export results

    ![](https://github.com/user-attachments/assets/5c80cc57-3870-4a20-bb74-73e394ef22a6)

### Clear GPU Memory

Click the **GPU** button in the GeoAI toolbar to release GPU memory from all loaded models (Moondream, SamGeo, etc.) and clear CUDA cache. Use this frequently when switching between large models to prevent out-of-memory errors.

![](https://github.com/user-attachments/assets/76c9dd8a-581c-4975-9ecb-4bfe301447bd)

## Supported Model Architectures (Segmentation)

The QGIS plugin supports any models supported by [Pytorch Segmentation Models](https://smp.readthedocs.io/en/latest/models.html), including:

-   U-Net
-   U-Net++
-   DeepLabV3
-   DeepLabV3+
-   FPN (Feature Pyramid Network)
-   PSPNet
-   LinkNet
-   MANet
-   PAN
-   UperNet
-   SegFormer
-   DPT

## Supported Encoders (Segmentation)

-   ResNet (34, 50, 101, 152)
-   EfficientNet (b0-b4)
-   MobileNetV2
-   VGG (16, 19)

## Supported SAM Models (SamGeo)

-   **SamGeo3 (SAM3)**: Latest version with text prompts, point prompts, and box prompts
-   **SamGeo2 (SAM2)**: Improved version with better performance
-   **SamGeo (SAM1)**: Original Segment Anything Model

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

-   [GeoAI Documentation](https://opengeoai.org)
-   [SamGeo Documentation](https://samgeo.gishub.org)
-   [GitHub Repository](https://github.com/opengeos/geoai)
-   [Report Issues](https://github.com/opengeos/geoai/issues)
