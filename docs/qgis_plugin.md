# QGIS Plugin for GeoAI

A QGIS plugin that brings the [geoai](https://github.com/opengeos/geoai) models into dockable panels (Moondream VLM, segmentation training/inference, SamGeo) so you can keep QGIS as your main workspace while experimenting with GeoAI.

## Quick Start

-   Create a fresh conda env (`conda create -n geo python=3.12`) and install QGIS + deps (see below).
-   Install the plugin (`python install.py`) from this repo.
-   Restart QGIS → `Plugins` → `Manage and Install Plugins...` → enable `GeoAI`.
-   Open a GeoAI toolbar panel and try the sample datasets below.

## Video Tutorials

Check out this [short video demo](https://youtu.be/Esr_e6_P1is) and [full video tutorial](https://youtu.be/8-OhlqeoyiY) on how to use the GeoAI plugin in QGIS.

[![demo](https://github.com/user-attachments/assets/5aabc3d3-efd1-4011-ab31-2b3f11aab3ed)](https://youtu.be/8-OhlqeoyiY)

## Requirements

-   QGIS 3.28 or later
-   Python 3.10+ (conda recommended)
-   PyTorch (CUDA if you want GPU acceleration)
-   `geoai` and `samgeo` packages

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

## Installation

### 1) Set up the environment

#### Installation on Linux/macOS

Use a clean conda env dedicated to QGIS—mixing with an existing QGIS install often breaks dependencies.

```bash
conda create -n geo python=3.12
conda activate geo
```

Install core geospatial deps first:

```bash
conda install -c conda-forge --strict-channel-priority gdal rasterio libnetcdf netcdf4
python -c "import rasterio; print('rasterio import successful')"
```

Install GeoAI:

```bash
conda install -c conda-forge geoai
python -c "import geoai; print('geoai import successful')"
```

Install QGIS:

```bash
conda install -c conda-forge qgis
```

Install SamGeo extras (PyPI is required for some parts):

```bash
pip install -U "segment-geospatial[samgeo3]" sam3
python -c "import samgeo; print('samgeo import successful')"
```

#### Installation on Windows

Windows installation requires some additional steps compared to Linux/macOS. Choose the appropriate section based on whether you have an NVIDIA GPU or want CPU-only installation.

**Prerequisites (Required for all Windows users):**

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) if you haven't already.
2. Open **Anaconda Prompt** (not PowerShell or CMD) for all installation commands.
3. For GPU users: Ensure you have the latest [NVIDIA GPU drivers](https://www.nvidia.com/Download/index.aspx) installed.

##### Option A: Windows with NVIDIA GPU (CUDA)

This option provides the best performance using your NVIDIA GPU for model inference and training.

**Step 1: Create and activate the conda environment**

```bash
conda create -n geo python=3.12 -y
conda activate geo
```

**Step 2: Install PyTorch with CUDA support**

First, check your NVIDIA driver version to determine the compatible CUDA version:

```bash
nvidia-smi
```

Look for the "CUDA Version" in the output. Then install the appropriate PyTorch version:

For CUDA 12.4 (recommended for newer drivers):

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

For CUDA 12.1 (for older drivers):

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Step 3: Verify PyTorch GPU installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see `CUDA available: True` and your GPU name. If not, see the troubleshooting section below.

**Step 4: Install QGIS and core dependencies**

```bash
conda install -c conda-forge qgis -y
```

**Step 5: Install GeoAI**

```bash
conda install -c conda-forge geoai -y
python -c "import geoai; print('geoai import successful')"
```

**Step 6: Install SamGeo with SAM3 support**

```bash
pip install -U triton-windows
pip install -U "segment-geospatial[samgeo3]"
pip install -U sam3
python -c "import samgeo; print('samgeo import successful')"
```

##### Option B: Windows CPU-Only (No GPU)

Use this option if you don't have an NVIDIA GPU or want a simpler installation.

**Step 1: Create and activate the conda environment**

```bash
conda create -n geo python=3.12 -y
conda activate geo
```

**Step 2: Install PyTorch (CPU version)**

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

**Step 3: Verify PyTorch installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print('PyTorch CPU installation successful')"
```

**Step 4: Install QGIS and core dependencies**

```bash
conda install -c conda-forge qgis -y
```

**Step 5: Install GeoAI**

```bash
conda install -c conda-forge geoai -y
python -c "import geoai; print('geoai import successful')"
```

**Step 6: Install SamGeo (without SAM3)**

```bash
pip install segment-geospatial
python -c "import samgeo; print('samgeo import successful')"
```

##### Windows Troubleshooting

**Common Issue 1: CUDA not detected after PyTorch installation**

If `torch.cuda.is_available()` returns `False`:

1. Verify NVIDIA drivers are installed: Run `nvidia-smi` in command prompt
2. Ensure you installed the CUDA-enabled PyTorch (not CPU version)
3. Try reinstalling PyTorch:

```bash
conda uninstall pytorch torchvision -y
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

**Common Issue 2: DLL load failed or missing dependencies**

If you see errors like `DLL load failed` or `ImportError`:

1. Install Microsoft Visual C++ Redistributable:
    - Download and install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Restart your computer after installation

**Common Issue 3: Triton installation fails**

Triton is required for SAM3 on Windows. If `pip install triton-windows` fails:

1. Ensure you're using Python 3.12 (not 3.13+)
2. Try installing from conda-forge:

```bash
pip install triton-windows --no-cache-dir
```

If Triton still doesn't work, you can skip SAM3 and use SAM1/SAM2 instead.

**Common Issue 4: Permission errors during installation**

Run Anaconda Prompt as Administrator, or try:

```bash
pip install --user <package-name>
```

**Common Issue 5: QGIS fails to start or shows import errors**

Make sure you launch QGIS from the activated conda environment:

```bash
conda activate geo
qgis
```

Do NOT use the QGIS shortcut from the Start Menu—it won't have access to the conda packages.

**Common Issue 6: Out of memory errors**

If you run out of GPU memory:

1. Use the **GPU** button in the GeoAI toolbar to clear memory
2. Close other GPU-intensive applications
3. Use smaller batch sizes in training/inference settings
4. Switch to CPU mode in the plugin settings for smaller tasks

##### Video Tutorial

You can follow this [video tutorial](https://youtu.be/a-Ns9peiuu8) to install the GeoAI QGIS Plugin on Windows:

[![windows](https://github.com/user-attachments/assets/8d89d535-1d66-45d2-a6c0-171416c259c9)](https://youtu.be/a-Ns9peiuu8)

#### Request access to SAM 3

To use SAM 3, you will need to request access by filling out this form on Hugging Face at <https://huggingface.co/facebook/sam3>. Once your request has been approved, run the following command in the terminal to authenticate:

```bash
hf auth login
```

### 2) Install the QGIS plugin

Option A — use QGIS Plugin Manager (recommended):

GeoAI is available as an experimental plugin in the official [QGIS plugin repository](https://plugins.qgis.org/plugins/geoai). To install:

1. Launch QGIS: `conda run qgis`
2. Go to `Plugins` → `Manage and Install Plugins...` → `Settings` tab → check `Show also Experimental Plugins` → Click on `Reload all Repositories` button.
3. Switch to the `All` tab, search for `GeoAI`, select it, and click `Install Experimental Plugin`.

![](https://github.com/user-attachments/assets/c15c4b7e-1e8c-45de-9127-671c7c2c85a9)

![](https://github.com/user-attachments/assets/ea101a82-5df7-4947-99b8-d4a83a1598ed)

Option B — use the helper script:

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai/qgis_plugin
python install.py
```

This links/copies the plugin into your active QGIS profile. Re-run after pulling updates. Remove with:

```bash
python install.py --remove
```

Option C — manual copy:

-   Copy the `qgis_plugin` folder to your QGIS plugins directory:
    -   Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
    -   Windows: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
    -   macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

### 3) Enable in QGIS

Launch QGIS: `conda run qgis`

QGIS → `Plugins` → `Manage and Install Plugins...` → enable `GeoAI`. After updates, toggle the plugin off/on or restart QGIS to reload.

![](https://github.com/user-attachments/assets/1b6dab14-311d-4f62-85aa-1faed73ead5b)

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

-   [naip_rgb_train.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_rgb_train.tif)
-   [naip_test.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_test.tif)
-   [naip_train_buildings.geojson](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_train_buildings.geojson)

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

Sample dataset:

-   [uc_berkeley.tif](https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif)
-   [wa_building_image.tif](https://github.com/opengeos/datasets/releases/download/places/wa_building_image.tif)
-   [wa_building_centroids.geojson](https://github.com/opengeos/datasets/releases/download/places/wa_building_centroids.geojson)
-   [wa_building_bboxes.geojson](https://github.com/opengeos/datasets/releases/download/places/wa_building_bboxes.geojson)

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

### Plugin Update Checker

Go to `GeoAI` menu → `Check for Updates...` to see if a newer version of the GeoAI plugin is available. Click on the `Check for Updates` button to fetch the latest version info from GitHub. If an update is found, click the `Download and Install Update` button to download and install the latest version automatically. Restart QGIS to apply the update.

![](https://github.com/user-attachments/assets/cc0dfd38-9b41-4735-9af0-c49b7aa71b72)

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

## Troubleshooting

-   Plugin missing after install: confirm the plugin folder exists in your QGIS profile path and that you restarted QGIS.
-   GDAL/rasterio errors: verify you launched QGIS from the conda env (`conda activate geo` then `qgis`) so it picks up the same Python libs.
-   CUDA OOM: use the **GPU** button to clear cache, lower batch sizes, or switch to CPU for smaller runs.
-   Model download failures: check network/firewall, then retry loading models from the panel.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

-   [GeoAI Documentation](https://opengeoai.org)
-   [SamGeo Documentation](https://samgeo.gishub.org)
-   [GitHub Repository](https://github.com/opengeos/geoai)
-   [Report Issues](https://github.com/opengeos/geoai/issues)
