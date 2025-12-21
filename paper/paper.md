---
title: "GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization"
tags:
    - Python
    - geospatial
    - artificial intelligence
    - deep learning
    - Jupyter
    - visualization

authors:
    - name: Qiusheng Wu
      orcid: 0000-0001-5437-4073
      affiliation: "1"
affiliations:
    - name: Department of Geography & Sustainability, University of Tennessee, Knoxville, TN 37996, United States
      index: 1
date: 12 September 2025
bibliography: paper.bib
---

# Summary

GeoAI is a comprehensive Python package designed to bridge artificial intelligence (AI) and geospatial data analysis, providing researchers and practitioners with intuitive tools for applying machine learning techniques to geographic data. The package offers a unified framework for processing satellite imagery, aerial photographs, and vector data using state-of-the-art deep learning models. GeoAI integrates popular AI frameworks including PyTorch [@Paszke2019], Transformers [@Wolf2019], PyTorch Segmentation Models [@Iakubovskii2019], and specialized geospatial libraries like torchange [@Zheng2024], enabling users to perform complex geospatial analyses with minimal code.

The package provides five core capabilities:

1. Interactive and programmatic search and download of remote sensing imagery and geospatial data.
2. Automated dataset preparation with image chips and label generation.
3. Model training for tasks such as classification, detection, and segmentation.
4. Inference pipelines for applying models to new geospatial datasets.
5. Interactive visualization through integration with Leafmap [@Wu2021] and MapLibre.
6. Seamless QGIS integration via a dedicated GeoAI plugin, enabling users to run AI-powered geospatial workflows directly within the QGIS desktop environment, without writing code.

GeoAI addresses the growing demand for accessible AI tools in geospatial research by providing high-level APIs that abstract complex machine learning workflows while maintaining flexibility for advanced users. The package supports multiple data formats (GeoTIFF, JPEG2000,GeoJSON, Shapefile, GeoPackage) and includes automatic device management for GPU acceleration when available. With over 10 modules and extensive notebook examples, GeoAI serves as both a research tool and educational resource for the geospatial AI community.

# Statement of Need

The integration of artificial intelligence with geospatial data analysis has become increasingly critical across numerous scientific disciplines, from environmental monitoring and urban planning to disaster response and climate research [@Li2022; @Mai2024]. However, applying AI techniques to geospatial data presents unique challenges including data preprocessing complexities, specialized model architectures, and the need for domain-specific knowledge in both machine learning and geographic information systems [@Zhu2017; @Ma2019].

Existing solutions often require researchers to navigate fragmented ecosystems of tools, combining general-purpose machine learning libraries with specialized geospatial packages, leading to steep learning curves and reproducibility challenges. While packages like TorchGeo [@Stewart2022], TerraTorch [@Gomes2025], and SRAI [@Gramacki2023] provide excellent foundational tools for geospatial deep learning, there remains a gap for comprehensive, high-level interfaces that can democratize access to advanced AI techniques for the broader geospatial community.

GeoAI addresses this need by providing a unified, user-friendly interface that abstracts the complexity of integrating multiple AI frameworks with geospatial data processing workflows. It lowers barriers for: (1) geospatial researchers who need accessible AI workflows without deep ML expertise; (2) AI practitioners who want streamlined geospatial preprocessing and domain-specific datasets; and (3) educators seeking reproducible examples and teaching-ready workflows.

The package's design philosophy emphasizes simplicity without sacrificing functionality, enabling users to perform sophisticated analyses such as building footprint extraction from satellite imagery, land cover classification, and change detection with just a few lines of code. By integrating cutting-edge AI models and providing seamless access to major geospatial data sources, GeoAI significantly lowers the barrier to entry for geospatial AI applications while maintaining the flexibility needed for advanced research applications.

# Acknowledgements

We gratefully acknowledge the support of the National Aeronautics and Space Administration (NASA) through Grant No. 80NSSC22K1742, awarded under the Open Source Tools, Frameworks, and Libraries Program. Additional support was provided by the U.S. Geological Survey through Grant/Cooperative Agreement No. G23AP00683 (GY23-GY27) in collaboration with AmericaView. We also thank the broader open-source geospatial community for their contributions and feedback during the development of this package.

# References
