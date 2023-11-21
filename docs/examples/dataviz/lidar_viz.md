---
jupytext:
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.15.2
---

[![image](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/opengeos/geoai/blob/main/examples/dataviz/lidar_viz.ipynb)
[![image](https://img.shields.io/badge/Open-Planetary%20Computer-black?style=flat&logo=microsoft)](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/user-redirect/git-pull?repo=https://github.com/opengeos/geoai&urlpath=lab/tree/geoai/examples/dataviz/lidar_viz.ipynb&branch=main)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/opengeos/geoai/blob/main/examples/dataviz/lidar_viz.ipynb)

# Visualizing LiDAR Data with Leafmap

```{code-cell}
# %pip install leafmap[lidar] open3d
```

```{code-cell}
import leafmap
```

Download a [sample LiDAR dataset](https://drive.google.com/file/d/1H_X1190vL63BoFYa_cVBDxtIa8rG-Usb/view?usp=sharing) from Google Drive. The zip file is 52.1 MB and the uncompressed LAS file is 109 MB.

```{code-cell}
url = 'https://open.gishub.org/data/lidar/madison.zip'
filename = 'madison.las'
```

```{code-cell}
leafmap.download_file(url, 'madison.zip', unzip=True)
```

Read the LiDAR data

```{code-cell}
las = leafmap.read_lidar(filename)
```

The LAS header.

```{code-cell}
las.header
```

The number of points.

```{code-cell}
las.header.point_count
```

The list of features.

```{code-cell}
list(las.point_format.dimension_names)
```

Inspect data.

```{code-cell}
las.X
```

```{code-cell}
las.Y
```

```{code-cell}
las.Z
```

```{code-cell}
las.intensity
```

Visualize LiDAR data using the [pyvista](https://github.com/pyvista/pyvista) backend.

```{code-cell}
leafmap.view_lidar(filename, cmap='terrain', backend='pyvista')
```

![](https://i.imgur.com/xezcgMP.gif)

+++

Visualize LiDAR data using the [ipygany](https://github.com/QuantStack/ipygany) backend.

```{code-cell}
leafmap.view_lidar(filename, backend='ipygany', background='white')
```

![](https://i.imgur.com/MyMWW4I.gif)

+++

Visualize LiDAR data using the [panel](https://github.com/holoviz/panel) backend.

```{code-cell}
leafmap.view_lidar(filename, cmap='terrain', backend='panel', background='white')
```

![](https://i.imgur.com/XQGWbJk.gif)

+++

Visualize LiDAR data using the [open3d](http://www.open3d.org) backend.

```{code-cell}
leafmap.view_lidar(filename, backend='open3d')
```

![](https://i.imgur.com/rL85fbl.gif)
