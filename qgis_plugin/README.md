# QGIS Plugin for GeoAI

A QGIS plugin that brings [GeoAI](https://github.com/opengeos/geoai) models into dockable panels (Tree Segmentation, Water Segmentation, Change Detection, Moondream VLM, Segment Anything, Semantic Segmentation, Instance Segmentation, and AI Processing Monitor) so you can keep QGIS as your main workspace while running AI-powered geospatial workflows. It supports **QGIS 3.28+ and QGIS 4.0 (PyQt6)** on Linux, Windows, and macOS.

## Quick Start

- Install the GeoAI plugin from the QGIS Plugin Manager (or download [geoai.zip](https://qgis.gishub.org/plugins/geoai.zip)).
- Enable the GeoAI plugin in QGIS.
- Open an AI tool panel (such as **Semantic Segmentation**, **Segment Anything**, or **Moondream**); the **Built-in Dependency Installer** will guide you through one-click environment setup.
- Try the sample datasets below.

## Video Tutorials

### Installation Tutorial

You can follow this [video tutorial](https://youtu.be/L90id_ESQME) to install the GeoAI QGIS Plugin on Linux and Windows:

[![installation](https://github.com/user-attachments/assets/3cda92c0-4de9-4cad-81c8-ddb3c790be91)](https://youtu.be/L90id_ESQME)

### Usage Tutorial

Check out this [short video demo](https://youtu.be/Esr_e6_P1is) and [full video tutorial](https://youtu.be/8-OhlqeoyiY) on how to use the GeoAI plugin in QGIS.

[![demo](https://github.com/user-attachments/assets/5aabc3d3-efd1-4011-ab31-2b3f11aab3ed)](https://youtu.be/8-OhlqeoyiY)

## Requirements

- QGIS 3.28 or later, including **QGIS 4.0 (PyQt6)** dual compatibility
- Python 3.12+ (Built-in Dependency Installer recommended; Pixi optional for custom CUDA setups)
- PyTorch (CUDA for NVIDIA GPUs, Apple MPS for Apple Silicon, or CPU fallback)
- `geoai-py`, `segment-geospatial` (`samgeo`), `deepforest`, `omniwatermask`, and `torchange` packages (installed automatically by the Built-in Dependency Installer)

## Features

Each tool lives inside a dockable panel that can be attached to either side of the QGIS interface, so you can keep layers, maps, and models visible simultaneously.

### Tree Segmentation Panel (DeepForest)

- **Model Tab**: Load pretrained [DeepForest](https://deepforest.readthedocs.io) models from Hugging Face for tree crown, bird, livestock, nest, and dead tree detection
- **Predict Tab**: Run predictions on single images or large geospatial tiles with configurable patch size, overlap, IoU threshold, and score filtering
- **Output Tab**: Save detection results as georeferenced vector (GeoPackage, Shapefile, GeoJSON) or raster (GeoTIFF), and export training data in PASCAL_VOC, COCO, or YOLO format

### Water Segmentation Panel (OmniWaterMask)

- **Water Body Detection**: Automated water body segmentation from satellite and aerial imagery using [OmniWaterMask](https://github.com/DPIRD-DMA/OmniWaterMask)
- **Band Configuration**: Configurable band ordering for multispectral imagery (RGBN, BGRN, etc.)
- **Patch-based Inference & OSM Integration**: Sliding-window inference with optional OpenStreetMap water, building, and road refinement
- **Output Options**: Save masks as GeoTIFF or smoothed/filtered vector polygons (GeoPackage, Shapefile)

### Change Detection Panel (AnyChange)

- **Zero-Shot Change Detection**: Detect changes between two co-registered raster layers using [AnyChange](https://github.com/Z-Zheng/pytorch-change-models) built on SAM
- **Threshold Control**: Set the change angle manually or enable automatic thresholding
- **Output Options**: Save change masks as GeoTIFF and optional change polygons (GeoPackage, GeoJSON, Shapefile)

### Moondream Vision-Language Model Panel

- **Caption**: Generate descriptions of geospatial imagery (short, normal, or long)
- **Query**: Ask questions about images using natural language
- **Detect**: Detect and locate objects with bounding boxes
- **Point**: Locate specific objects with point markers

### Segment Anything Panel (SamGeo)

- **Model Tab**: Load SAM models (SAM3, SAM2, or SAM1) with configurable backend and device settings, plus built-in Hugging Face token login for gated SAM 3 access
- **Text Tab**: Segment objects using text prompts (e.g., "tree", "building", "road"), with optional tiled processing for large GeoTIFFs
- **Interactive Tab**: Segment using point prompts (foreground/background) or box prompts drawn on the map
- **Batch Tab**: Process multiple points interactively or from vector files/layers
- **Output Tab**: Save results as raster (GeoTIFF) or vector (GeoPackage, Shapefile) with optional polygon regularization and `score`, `source`, and `prompt` attributes

### Semantic Segmentation Panel (Combined Training & Inference)

- **Tab 1 - Create Training Data**: Export GeoTIFF tiles from raster and vector data
- **Tab 2 - Train Models**: Train custom segmentation models (U-Net, DeepLabV3+, FPN, SegFormer, etc.)
- **Tab 3 - Run Inference**: Apply trained models to new imagery and vectorize results with optional smoothing or simplification

### Instance Segmentation Panel (Mask R-CNN)

- **Tab 1 - Create Training Data**: Export GeoTIFF tiles with instance-level masks in PASCAL_VOC format, with configurable tile size, stride, and buffer radius
- **Tab 2 - Train Models**: Train Mask R-CNN models with configurable channels, classes, batch size, epochs, learning rate, and validation split
- **Tab 3 - Run Inference**: Apply trained Mask R-CNN models to new imagery and vectorize results with optional smoothing

### AI Processing Monitor

Open **GeoAI → AI Processing Monitor** to observe supported Semantic Segmentation tasks while they run. The monitor covers tile export, training, inference, vectorization, and vector smoothing started from the Semantic Segmentation panel.

- **Lifecycle States**: Tracks `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, and `CANCELLED` without blocking the QGIS map canvas.
- **Timing and Progress**: Displays monotonic elapsed time, start and end wall-clock timestamps, epoch-based percentages for training, and indeterminate progress bars for message-only operations (never fabricating percentages).
- **Error Diagnostics and GPU Metrics**: Retains concise error categories (missing input, missing dependency, GPU/CUDA, output writing, processing) alongside full tracebacks, shows PyTorch CUDA memory metrics when accessible from the QGIS process, and lets you clear finished entries with **Clear finished** while retaining active tasks.
- **Safety Notes**: The monitor never clears CUDA cache automatically. **Clear GPU Memory** remains a separate manual action and should not be used while a GPU task is actively running.

### Isolated Managed Subprocess Architecture

Starting with v1.9.2, AI panels run model loading, inference, and training-tile export inside isolated subprocesses launched from the plugin's managed virtual environment (`~/.qgis_geoai/`) rather than importing PyTorch into the QGIS process. This prevents PyTorch and OpenMP runtime conflicts (such as `OMP: Error #15` aborts in conda-forge QGIS on macOS) and keeps the QGIS UI responsive.

### GPU Memory Management

- **Clear GPU Memory**: Release GPU memory and clear CUDA cache for loaded models

## Installation

There are two ways to set up the GeoAI plugin dependencies:

- **Option A - Built-in Dependency Installer (recommended)**: One-click setup that creates an isolated Python environment at `~/.qgis_geoai/` (or `GEOAI_CACHE_DIR`), detects NVIDIA CUDA or Apple MPS, and installs all required packages using `uv`.
- **Option B - Pixi Environment (advanced)**: Manually manage a Pixi environment with explicit PyTorch and CUDA pins.

### Built-in Dependency Installer

When you first open any GeoAI panel, the plugin detects whether the required AI packages are installed. If not, it shows a one-click setup dialog that:

- Detects your GPU (NVIDIA CUDA or Apple MPS) and configures PyTorch accordingly
- Creates an isolated virtual environment at `~/.qgis_geoai/`
- Installs PyTorch, `geoai-py`, `segment-geospatial`, `deepforest`, `omniwatermask`, `torchange`, and `transformers`
- Shows real-time installation progress

To store the managed environment outside your home directory, set `GEOAI_CACHE_DIR` before starting QGIS:

```bash
export GEOAI_CACHE_DIR=/path/to/writable/directory
```

### Pixi Environment (Optional)

If you prefer a manually managed Pixi environment, install [Pixi](https://pixi.prefix.dev/latest), run `pixi init geo && cd geo`, and configure `pixi.toml` with Python 3.12, QGIS `3.44.*`, `geoai`, `segment-geospatial`, and `sam3`. See the full [QGIS Plugin Documentation](https://opengeoai.org/qgis_plugin/#1-set-up-the-environment-with-pixi) for CUDA 12.x, CUDA 13.x, and CPU `pixi.toml` templates and DeepForest setup.

### Install the QGIS Plugin

Option A: use QGIS Plugin Manager (recommended):

1. Launch QGIS (QGIS 3.28+ or QGIS 4.0)
2. Go to `Plugins` → `Manage and Install Plugins...`
3. Switch to the `All` tab, search for `GeoAI`, select it, and click `Install Plugin`

Option B: use the helper script:

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai/qgis_plugin
python install.py
```

This copies the plugin into the `default` profile of every QGIS major version whose settings directory exists (`QGIS4` and `QGIS3`; use `--profile NAME` for another profile). If QGIS 4 has not created its settings directory yet, launch QGIS 4 once first or specify its plugin directory with `--plugin-dir` (if neither settings directory exists, installation falls back to the QGIS 3 path). Remove with `python install.py --remove` (pass the same `--profile NAME` or `--plugin-dir DIR` option if you used one during installation).

Option C: manual copy:

- Copy the `qgis_plugin/geoai` folder to your QGIS plugins directory:
    - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
    - Windows: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins/`
    - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
    - QGIS 4 uses `QGIS4` instead of `QGIS3` in these paths.

## Hugging Face Authentication (SAM 3)

SAM 3 is hosted in a gated Hugging Face repository (<https://huggingface.co/facebook/sam3>). After your access request is approved, create a Read (or gated-repo-enabled Fine-grained) token at <https://huggingface.co/settings/tokens> and authenticate using one of these methods:

- **In-panel login (built-in installer)**: Open **Segment Anything → Model tab → Hugging Face**, paste your token, and click **Log in**.
- **CLI (Pixi environment)**: Run `pixi run hf auth login` from your Pixi project folder.
- **Environment variable**: Set `HF_TOKEN` in the environment that launches QGIS.

SAM 3 currently requires an NVIDIA GPU with CUDA support; use SAM 1 or SAM 2 on CPU-only systems. See [Hugging Face Authentication (SAM 3)](https://opengeoai.org/qgis_plugin/#hugging-face-authentication-sam-3) for complete instructions.

## Utilities and Maintenance

### Plugin Update Checker

Go to `GeoAI` menu → `Check for Updates...` to check GitHub for newer plugin releases and install updates directly from QGIS.

### Diagnostics Report

Go to `GeoAI` menu → `Generate Diagnostics Report...` to create a Markdown report ready to paste into a GitHub issue. It includes the plugin version, QGIS/Python runtime, operating system, managed virtual environment path (with home paths redacted), CUDA/GPU details, and versions/import status for key packages (`geoai-py`, PyTorch, SAM 3, Transformers, `segment-geospatial`).

## Supported Model Architectures (Segmentation)

Any architecture supported by [PyTorch Segmentation Models](https://smp.readthedocs.io/en/latest/models.html), including U-Net, U-Net++, DeepLabV3, DeepLabV3+, FPN, PSPNet, LinkNet, MANet, PAN, UperNet, SegFormer, and DPT, with ResNet, EfficientNet, MobileNetV2, and VGG encoders.

## Troubleshooting

- Include the Markdown from `GeoAI` menu → `Generate Diagnostics Report...` when reporting an issue on GitHub.
- Plugin missing after install: confirm `geoai` exists in your QGIS3 or QGIS4 profile plugins folder and restart QGIS.
- CUDA out-of-memory: use the **GPU** toolbar button to clear cache, lower batch sizes, or switch to CPU.
- SAM 3 gated-repo error: verify your access approval on <https://huggingface.co/facebook/sam3> and sign in via the Segment Anything panel's **Hugging Face** group.

### Windows Installation Issues

If the plugin or dependencies fail to install on Windows when using the QGIS Plugin Manager, try a clean reinstall with the quick installer:

1. Uninstall the GeoAI plugin completely from QGIS.
2. Close QGIS.
3. Open PowerShell and run:

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://qgis.gishub.org/install.ps1 | iex"
    ```

4. Restart QGIS.
5. Open `Plugins` → `Manage and Install Plugins...`, enable **GeoAI**, then use the plugin's built-in dependency installer.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

- [Full QGIS Plugin Guide](https://opengeoai.org/qgis_plugin/)
- [GeoAI Documentation](https://opengeoai.org)
- [SamGeo Documentation](https://samgeo.gishub.org)
- [GitHub Repository](https://github.com/opengeos/geoai)
- [Report Issues](https://github.com/opengeos/geoai/issues)
