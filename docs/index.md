# GeoAI: Artificial Intelligence for Geospatial Data

[![image](https://img.shields.io/pypi/v/geoai-py.svg)](https://pypi.python.org/pypi/geoai-py)
[![image](https://static.pepy.tech/badge/geoai-py)](https://pepy.tech/project/geoai-py)
[![image](https://img.shields.io/conda/vn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Recipe](https://img.shields.io/badge/recipe-geoai-green.svg)](https://github.com/conda-forge/geoai-py-feedstock)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/badge/YouTube-Tutorials-red)](https://tinyurl.com/GeoAI-Tutorials)

[![logo](https://raw.githubusercontent.com/opengeos/geoai/master/docs/assets/logo_rect.png)](https://github.com/opengeos/geoai/blob/master/docs/assets/logo.png)

**A powerful Python package for integrating artificial intelligence with geospatial data analysis and visualization**

## 📖 Introduction

[GeoAI](https://opengeoai.org) is a comprehensive Python package designed to bridge artificial intelligence (AI) and geospatial data analysis, providing researchers and practitioners with intuitive tools for applying machine learning techniques to geographic data. The package offers a unified framework for processing satellite imagery, aerial photographs, and vector data using state-of-the-art deep learning models. GeoAI integrates popular AI frameworks including [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers), [PyTorch Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch), and specialized geospatial libraries like [torchange](https://github.com/Z-Zheng/pytorch-change-models), enabling users to perform complex geospatial analyses with minimal code.

The package provides five core capabilities:

1. Interactive and programmatic search and download of remote sensing imagery and geospatial data.
2. Automated dataset preparation with image chips and label generation.
3. Model training for tasks such as classification, detection, and segmentation.
4. Inference pipelines for applying models to new geospatial datasets.
5. Interactive visualization through integration with [Leafmap](https://github.com/opengeos/leafmap/) and [MapLibre](https://github.com/eoda-dev/py-maplibregl).

GeoAI addresses the growing demand for accessible AI tools in geospatial research by providing high-level APIs that abstract complex machine learning workflows while maintaining flexibility for advanced users. The package supports multiple data formats (GeoTIFF, JPEG2000,GeoJSON, Shapefile, GeoPackage) and includes automatic device management for GPU acceleration when available. With over 10 modules and extensive notebook examples, GeoAI serves as both a research tool and educational resource for the geospatial AI community.

## 📝 Statement of Need

The integration of artificial intelligence with geospatial data analysis has become increasingly critical across numerous scientific disciplines, from environmental monitoring and urban planning to disaster response and climate research. However, applying AI techniques to geospatial data presents unique challenges including data preprocessing complexities, specialized model architectures, and the need for domain-specific knowledge in both machine learning and geographic information systems.

Existing solutions often require researchers to navigate fragmented ecosystems of tools, combining general-purpose machine learning libraries with specialized geospatial packages, leading to steep learning curves and reproducibility challenges. While packages like TorchGeo and TerraTorch provide excellent foundational tools for geospatial deep learning, there remains a gap for comprehensive, high-level interfaces that can democratize access to advanced AI techniques for the broader geospatial community.

GeoAI addresses this need by providing a unified, user-friendly interface that abstracts the complexity of integrating multiple AI frameworks with geospatial data processing workflows. It lowers barriers for: (1) geospatial researchers who need accessible AI workflows without deep ML expertise; (2) AI practitioners who want streamlined geospatial preprocessing and domain-specific datasets; and (3) educators seeking reproducible examples and teaching-ready workflows.

The package's design philosophy emphasizes simplicity without sacrificing functionality, enabling users to perform sophisticated analyses such as building footprint extraction from satellite imagery, land cover classification, and change detection with just a few lines of code. By integrating cutting-edge AI models and providing seamless access to major geospatial data sources, GeoAI significantly lowers the barrier to entry for geospatial AI applications while maintaining the flexibility needed for advanced research applications.

## Citations

If you find GeoAI useful in your research, please consider citing the following paper to support my work. Thank you for your support.

-   Wu, Q. (2025). GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization. _Journal of Open Source Software_, 9025. [https://doi.org/10.21105/joss.09025](https://github.com/openjournals/joss-papers/blob/joss.09025/joss.09025/10.21105.joss.09025.pdf) (Under Review)

## 🚀 Key Features

### 📊 Advanced Geospatial Data Visualization

-   Interactive multi-layer visualization of vector and raster data stored locally or in cloud storage
-   Customizable styling and symbology
-   Time-series data visualization capabilities

### 🛠️ Data Preparation & Processing

-   Streamlined access to satellite and aerial imagery from providers like Sentinel, Landsat, NAIP, and other open datasets
-   Tools for downloading, mosaicking, and preprocessing remote sensing data
-   Automated generation of training datasets with image chips and corresponding labels
-   Vector-to-raster and raster-to-vector conversion utilities optimized for AI workflows
-   Data augmentation techniques specific to geospatial data
-   Support for integrating Overture Maps data and other open datasets for training and validation

### 🖼️ Image Segmentation

-   Integration with [PyTorch Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch) for automatic feature extraction
-   Specialized segmentation algorithms optimized for satellite and aerial imagery
-   Streamlined workflows for segmenting buildings, water bodies, wetlands,solar panels, etc.
-   Export capabilities to standard geospatial formats (GeoJSON, Shapefile, GeoPackage, GeoParquet)

### 🔍 Image Classification

-   Pre-trained models for land cover and land use classification
-   Transfer learning utilities for fine-tuning models with your own data
-   Multi-temporal classification support for change detection
-   Accuracy assessment and validation tools

### 🌍 Additional Capabilities

-   Change detection with AI-enhanced feature extraction
-   Object detection in aerial and satellite imagery
-   Georeferencing utilities for AI model outputs

## 📦 Installation

### Using pip

```bash
pip install geoai-py
```

### Using conda

```bash
conda install -c conda-forge geoai
```

### Using mamba

```bash
mamba install -c conda-forge geoai
```

## 📋 Documentation

Comprehensive documentation is available at [https://opengeoai.org](https://opengeoai.org), including:

-   Detailed API reference
-   Tutorials and example notebooks
-   Contributing guide

## 📺 Video Tutorials

### GeoAI Made Easy: Learn the Python Package Step-by-Step (Beginner Friendly)

[![intro](https://github.com/user-attachments/assets/7e60ce05-573d-4d0d-9876-5289b87e5136)](https://youtu.be/VIl29Rca6zE&list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

### GeoAI Workshop: Unlocking the Power of GeoAI with Python

[![cover](https://github.com/user-attachments/assets/1c14e651-65b9-41ae-b42d-3ad028b3eeb8)](https://youtu.be/jdK-cleFUkc&list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

### GeoAI Tutorials Playlist

[![cover](https://github.com/user-attachments/assets/3cde9547-ab62-4d70-b23a-3e5ed27c7407)](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE)

## 🤝 Contributing

We welcome contributions of all kinds! See our [contributing guide](https://opengeoai.org/contributing) for ways to get started.

## 📄 License

GeoAI is free and open source software, licensed under the MIT License.

## Acknowledgments

We gratefully acknowledge the support of the following organizations:

-   [NASA](https://www.nasa.gov): This research is partially supported by the National Aeronautics and Space Administration (NASA) through Grant No. 80NSSC22K1742, awarded under the [Open Source Tools, Frameworks, and Libraries Program](https://bit.ly/3RVBRcQ).
-   [AmericaView](https://americaview.org): This work is also partially supported by the U.S. Geological Survey through Grant/Cooperative Agreement No. G23AP00683 (GY23-GY27) in collaboration with AmericaView.
