# Installation

This guide covers various methods for installing GeoAI on different platforms with different package managers.

## ✅ Prerequisites

GeoAI requires:

-   Python 3.9 or above
-   The required dependencies will be installed automatically

## 🚀 Recommended Installation Methods

### 🐍 Using pip

The simplest way to install the latest stable release of GeoAI is via pip:

```bash
pip install geoai-py
```

To install GeoAI with all optional dependencies for additional features:

```bash
pip install "geoai-py[all]"
```

### 🐼 Using conda

For Anaconda/Miniconda users, we recommend installing GeoAI via conda-forge, which handles dependencies like GDAL more elegantly:

```bash
conda install -c conda-forge geoai
```

Consider creating a dedicated environment for your GeoAI projects:

```bash
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge geoai
```

### 🦡 Using mamba

Mamba provides faster dependency resolution compared to conda:

```bash
mamba install -c conda-forge geoai
```

## 🔧 Advanced Installation Options

### 🖥️ GPU Support

To enable GPU acceleration for deep learning models (requires NVIDIA GPU):

```bash
mamba install -c conda-forge geoai "pytorch=*=cuda*"
```

This will install the appropriate PyTorch version with CUDA support.

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

## ⚠️ Troubleshooting

If you encounter installation problems:

1. Check the [FAQ section](https://geoai.gishub.org/faq) of our documentation
2. Search for similar issues in our [GitHub Issues](https://github.com/opengeos/geoai/issues)
3. Ask for help in our [GitHub Discussions](https://github.com/opengeos/geoai/discussions)

## 🔄 Upgrading

To upgrade GeoAI to the latest version:

```bash
pip install -U geoai-py
```

Or with conda:

```bash
conda update -c conda-forge geoai
```
