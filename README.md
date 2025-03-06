# GeoAI: Artificial Intelligence for Geospatial Data

[![image](https://img.shields.io/pypi/v/geoai-py.svg)](https://pypi.python.org/pypi/geoai-py)
[![image](https://static.pepy.tech/badge/geoai-py)](https://pepy.tech/project/geoai-py)
[![image](https://img.shields.io/conda/vn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoai.svg)](https://anaconda.org/conda-forge/geoai)
[![Conda Recipe](https://img.shields.io/badge/recipe-geoai-green.svg)](https://github.com/giswqs/geoai-py-feedstock)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/badge/YouTube-Tutorials-red)](https://bit.ly/GeoAI-Tutorials)

**A powerful Python package for integrating Artificial Intelligence with geospatial data analysis and visualization**

GeoAI bridges the gap between AI and geospatial analysis, providing tools for processing, analyzing, and visualizing geospatial data using advanced machine learning techniques. Whether you're working with satellite imagery, LiDAR point clouds, or vector data, GeoAI offers intuitive interfaces to apply cutting-edge AI models.

-   📖 **Documentation:** [https://geoai.gishub.org](https://geoai.gishub.org)
-   💬 **Community:** [GitHub Discussions](https://github.com/opengeos/geoai/discussions)
-   🐛 **Issue Tracker:** [GitHub Issues](https://github.com/opengeos/geoai/issues)

## 🚀 Key Features

❗ **Important notes:** The GeoAI package is under active development and new features are being added regularly. Not all features listed below are available in the current release. If you have a feature request or would like to contribute, please let us know!

### 📊 Advanced Geospatial Data Visualization

-   Interactive multi-layer visualization of vector, raster, and point cloud data
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

-   Integration with Meta's Segment Anything Model (SAM) for automatic feature extraction
-   Specialized segmentation algorithms optimized for satellite and aerial imagery
-   Streamlined workflows for segmenting buildings, roads, vegetation, and water bodies
-   Export capabilities to standard geospatial formats (GeoJSON, Shapefile, GeoPackage, GeoParquet)

### 🔍 Image Classification

-   Pre-trained models for land cover and land use classification
-   Transfer learning utilities for fine-tuning models with your own data
-   Multi-temporal classification support for change detection
-   Accuracy assessment and validation tools

### 🌍 Additional Capabilities

-   Terrain analysis with AI-enhanced feature extraction
-   Point cloud classification and segmentation
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

Comprehensive documentation is available at [https://geoai.gishub.org](https://geoai.gishub.org), including:

-   Detailed API reference
-   Tutorials and example notebooks
-   Explanation of algorithms and models
-   Best practices for geospatial AI

## 📺 Video Tutorials

Check out our [YouTube channel](https://bit.ly/GeoAI-Tutorials) for video tutorials on using GeoAI for geospatial data analysis and visualization.

[![cover](https://github.com/user-attachments/assets/3cde9547-ab62-4d70-b23a-3e5ed27c7407)](https://bit.ly/GeoAI-Tutorials)

## 🤝 Contributing

We welcome contributions of all kinds! See our [contributing guide](https://geoai.gishub.org/contributing) for ways to get started.

## 📄 License

GeoAI is free and open source software, licensed under the MIT License.
