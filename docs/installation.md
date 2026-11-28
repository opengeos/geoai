# Installation

This guide covers various methods for installing GeoAI on different platforms with different package managers.

## ✅ Prerequisites

GeoAI requires:

-   Python 3.12 or above
-   The required dependencies will be installed automatically

## 🚀 Recommended Installation Methods

### 🐍 Using pip

The simplest way to install the latest stable release of GeoAI is via pip:

```bash
pip install geoai-py
```

To install GeoAI with optional dependencies for additional features:

```bash
pip install "geoai-py[extra]"
```

You can also install one or more task-specific extras (for example, `pip install "geoai-py[agents,onnx]"`). See [Optional Dependencies](#optional-dependencies-extras) below for the full list.

### 🐍 Using uv

To install the latest stable release of GeoAI with [uv](https://docs.astral.sh/uv), a faster alternative to pip:

```bash
uv pip install geoai-py
```

Or with optional extras:

```bash
uv pip install "geoai-py[extra]"
```

### 🐼 Using conda

For Anaconda/Miniconda users, we recommend installing GeoAI via conda-forge, which handles dependencies like GDAL more elegantly:

```bash
conda install -c conda-forge geoai
```

### 🦡 Using mamba

Mamba provides faster dependency resolution compared to conda. This is especially useful for large packages like GeoAI:

```bash
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge mamba
mamba install -c conda-forge geoai
```

## 📦 Optional Dependencies (Extras)

GeoAI defines 10 optional dependency groups in `pyproject.toml` so you only install heavy or specialized packages when you need them:

| Extra | Packages | Capabilities |
| ----- | -------- | ------------ |
| `extra` | `torchange`, `lightly-train`, `multiclean`, `omnicloudmask`, `omniwatermask`, `quackosm` | Change detection, self-supervised pretraining, mask cleaning, cloud/water masking, and OSM downloads |
| `agents` | `strands-agents`, `strands-agents-tools`, `strands-agents[ollama]`, `strands-agents[anthropic]`, `strands-agents[openai]` | AI geospatial agents (`GeoAgent`, `STACAgent`, `CatalogAgent`, `DataTools`) |
| `onnx` | `onnx>=1.21.0`, `onnxruntime` | ONNX model export and inference (`geoai.onnx`, `ONNXInferenceStep`) |
| `rfdetr` | `rfdetr` | RF-DETR object detection and segmentation (`geoai.rfdetr`) |
| `terratorch` | `terratorch>=1.0` | Geospatial foundation models via TerraTorch and Prithvi (`geoai.prithvi`, `geoai.foundation_models`) |
| `sr` | `opensr-model` | Satellite imagery super-resolution (`geoai.tools.sr`) |
| `vllm` | `vllm` | High-throughput vision-language model inference (`geoai.vllm_geo`) |
| `networks` | `neatnet`, `osmnx` | Road network simplification and spatial graph analysis (`geoai.networks`) |
| `osd` | `opticallyshallowdeep` | Optically shallow and deep water classification (`geoai.osd`) |
| `building` | `buildingregulariser` | Building footprint regularization and orthogonalization (`geoai.orthogonalization`) |

Install one or more extras by combining them in quotes:

```bash
pip install "geoai-py[extra]"
pip install "geoai-py[agents,onnx]"
```

## 🔧 Advanced Installation Options

### 🖥️ GPU Support

To enable GPU acceleration for deep learning models (requires NVIDIA GPU):

```bash
mamba install -c conda-forge geoai "pytorch=*=cuda*"
```

This will install the appropriate PyTorch version with CUDA support.

If you run into issues with the ipympl package, you can install it using the following command:

```bash
mamba install -c conda-forge geoai "pytorch=*=cuda*" jupyterlab ipympl
```

If you encounter issues with the sqlite package, you can update it using the following command:

```bash
mamba update -c conda-forge sqlite
```

### Notes for Windows Users

If you use mamba to install geoai, you may not have the latest version of torchgeo, which may cause issues when importing geoai. To fix this, you can install the latest version of torchgeo using the following command:

```bash
pip install -U torchgeo
```

### 👩‍💻 Development Installation

For contributing to GeoAI development, install directly from the source repository:

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai
pip install -e .
```

The `-e` flag installs the package in development mode, allowing you to modify the code and immediately see the effects.

### 📦 Installing from GitHub

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/opengeos/geoai.git
```

For a specific branch:

```bash
pip install git+https://github.com/opengeos/geoai.git@branch-name
```

## ✓ Verifying Installation

To verify your installation, run:

```python
import geoai
print(geoai.__version__)
```

## ⚙️ Runtime Environment Variables

GeoAI recognizes the following environment variables at runtime:

| Variable | Values | Description |
| -------- | ------ | ----------- |
| `GEOAI_DEVICE` | PyTorch device string (e.g. `cpu`, `cuda`, `cuda:0`, `mps`, `xpu`) | Overrides automatic device selection in `geoai.utils.get_device()`. Raises a `ValueError` if the value is invalid or the requested accelerator is unavailable. |
| `GEOAI_ALLOW_UNSAFE_TORCH_LOAD` | `1`, `true`, `yes` | Allows loading legacy PyTorch checkpoints with `weights_only=False` when safe `weights_only=True` deserialization fails. Only enable this for checkpoints from trusted sources. |
| `GEOAI_DISABLE_PROGRESS` | `1`, `true`, `yes`, `on` | Globally disables `tqdm` progress bars across GeoAI operations (useful in CI, batch logs, and headless scripts). |

Example:

```bash
export GEOAI_DEVICE=cuda:0
export GEOAI_DISABLE_PROGRESS=1
```

## ⚠️ Troubleshooting

If you encounter installation problems:

1. Search for similar issues in our [GitHub Issues](https://github.com/opengeos/geoai/issues)
2. Ask for help in our [GitHub Discussions](https://github.com/opengeos/geoai/discussions)

## 🔄 Upgrading

To upgrade GeoAI to the latest version:

```bash
pip install -U geoai-py
```

Or with conda:

```bash
conda update -c conda-forge geoai
```
