"""The utils module contains common functions and classes used by the other modules."""

# Standard Library
import glob
import json
import math
import os
import subprocess
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-Party Libraries
import cv2
import geopandas as gpd
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import rioxarray as rxr
import torch
import torchgeo
import xarray as xr
from PIL import Image
from rasterio import features
from rasterio.plot import show
from rasterio.windows import Window
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.transforms import RandomRotation
from tqdm import tqdm


def view_raster(
    source: str,
    indexes: Optional[int] = None,
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Optional[float] = None,
    attribution: Optional[str] = None,
    layer_name: Optional[str] = "Raster",
    layer_index: Optional[int] = None,
    zoom_to_layer: Optional[bool] = True,
    visible: Optional[bool] = True,
    opacity: Optional[float] = 1.0,
    array_args: Optional[Dict] = None,
    client_args: Optional[Dict] = {"cors_all": False},
    basemap: Optional[str] = "OpenStreetMap",
    basemap_args: Optional[Dict] = None,
    backend: Optional[str] = "folium",
    **kwargs: Any,
) -> Any:
    """
    Visualize a raster using leafmap.

    Args:
        source (str): The source of the raster.
        indexes (Optional[int], optional): The band indexes to visualize. Defaults to None.
        colormap (Optional[str], optional): The colormap to apply. Defaults to None.
        vmin (Optional[float], optional): The minimum value for colormap scaling. Defaults to None.
        vmax (Optional[float], optional): The maximum value for colormap scaling. Defaults to None.
        nodata (Optional[float], optional): The nodata value. Defaults to None.
        attribution (Optional[str], optional): The attribution for the raster. Defaults to None.
        layer_name (Optional[str], optional): The name of the layer. Defaults to "Raster".
        layer_index (Optional[int], optional): The index of the layer. Defaults to None.
        zoom_to_layer (Optional[bool], optional): Whether to zoom to the layer. Defaults to True.
        visible (Optional[bool], optional): Whether the layer is visible. Defaults to True.
        opacity (Optional[float], optional): The opacity of the layer. Defaults to 1.0.
        array_args (Optional[Dict], optional): Additional arguments for array processing. Defaults to {}.
        client_args (Optional[Dict], optional): Additional arguments for the client. Defaults to {"cors_all": False}.
        basemap (Optional[str], optional): The basemap to use. Defaults to "OpenStreetMap".
        basemap_args (Optional[Dict], optional): Additional arguments for the basemap. Defaults to None.
        backend (Optional[str], optional): The backend to use. Defaults to "folium".
        **kwargs (Any): Additional keyword arguments.

    Returns:
        leafmap.Map: The map object with the raster layer added.
    """

    if backend == "folium":
        import leafmap.foliumap as leafmap
    else:
        import leafmap.leafmap as leafmap

    if basemap_args is None:
        basemap_args = {}

    if array_args is None:
        array_args = {}

    m = leafmap.Map()

    if isinstance(basemap, str):
        if basemap.lower().endswith(".tif"):
            if basemap.lower().startswith("http"):
                if "name" not in basemap_args:
                    basemap_args["name"] = "Basemap"
                m.add_cog_layer(basemap, **basemap_args)
            else:
                if "layer_name" not in basemap_args:
                    basemap_args["layer_name"] = "Basemap"
                m.add_raster(basemap, **basemap_args)
    else:
        m.add_basemap(basemap, **basemap_args)

    if isinstance(source, dict):
        source = dict_to_image(source)

    if isinstance(source, str) and source.startswith("http"):
        if backend == "folium":

            m.add_geotiff(
                url=source,
                name=layer_name,
                opacity=opacity,
                attribution=attribution,
                fit_bounds=zoom_to_layer,
                palette=colormap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            m.add_layer_control()
            m.add_opacity_control()

        else:
            if indexes is not None:
                kwargs["bidx"] = indexes
            if colormap is not None:
                kwargs["colormap_name"] = colormap
            if attribution is None:
                attribution = "TiTiler"

            m.add_cog_layer(
                source,
                name=layer_name,
                opacity=opacity,
                attribution=attribution,
                zoom_to_layer=zoom_to_layer,
                **kwargs,
            )
    else:
        m.add_raster(
            source=source,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            layer_index=layer_index,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            opacity=opacity,
            array_args=array_args,
            client_args=client_args,
            **kwargs,
        )
    return m


def view_image(
    image: Union[np.ndarray, torch.Tensor],
    transpose: bool = False,
    bdx: Optional[int] = None,
    clip_percentiles: Optional[Tuple[float, float]] = (2, 98),
    gamma: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 5),
    axis_off: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Visualize an image using matplotlib.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The image to visualize.
        transpose (bool, optional): Whether to transpose the image. Defaults to False.
        bdx (Optional[int], optional): The band index to visualize. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 5).
        axis_off (bool, optional): Whether to turn off the axis. Defaults to True.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        **kwargs (Any): Additional keyword arguments for plt.imshow().

    Returns:
        None
    """

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    elif isinstance(image, str):
        image = rasterio.open(image).read().transpose(1, 2, 0)

    ax = plt.figure(figsize=figsize)

    if transpose:
        image = image.transpose(1, 2, 0)

    if bdx is not None:
        image = image[:, :, bdx]

    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, 0:3]

    if clip_percentiles is not None:
        p_low, p_high = clip_percentiles
        lower = np.percentile(image, p_low)
        upper = np.percentile(image, p_high)
        image = np.clip((image - lower) / (upper - lower), 0, 1)

    if gamma is not None:
        image = np.power(image, gamma)

    plt.imshow(image, **kwargs)
    if axis_off:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def plot_images(
    images: Iterable[torch.Tensor],
    axs: Iterable[plt.Axes],
    chnls: List[int] = [2, 1, 0],
    bright: float = 1.0,
) -> None:
    """
    Plot a list of images.

    Args:
        images (Iterable[torch.Tensor]): The images to plot.
        axs (Iterable[plt.Axes]): The axes to plot the images on.
        chnls (List[int], optional): The channels to use for RGB. Defaults to [2, 1, 0].
        bright (float, optional): The brightness factor. Defaults to 1.0.

    Returns:
        None
    """
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis("off")


def plot_masks(
    masks: Iterable[torch.Tensor], axs: Iterable[plt.Axes], cmap: str = "Blues"
) -> None:
    """
    Plot a list of masks.

    Args:
        masks (Iterable[torch.Tensor]): The masks to plot.
        axs (Iterable[plt.Axes]): The axes to plot the masks on.
        cmap (str, optional): The colormap to use. Defaults to "Blues".

    Returns:
        None
    """
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap=cmap)
        ax.axis("off")


def plot_batch(
    batch: Dict[str, Any],
    bright: float = 1.0,
    cols: int = 4,
    width: int = 5,
    chnls: List[int] = [2, 1, 0],
    cmap: str = "Blues",
) -> None:
    """
    Plot a batch of images and masks. This function is adapted from the plot_batch()
    function in the torchgeo library at
    https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html
    Credit to the torchgeo developers for the original implementation.

    Args:
        batch (Dict[str, Any]): The batch containing images and masks.
        bright (float, optional): The brightness factor. Defaults to 1.0.
        cols (int, optional): The number of columns in the plot grid. Defaults to 4.
        width (int, optional): The width of each plot. Defaults to 5.
        chnls (List[int], optional): The channels to use for RGB. Defaults to [2, 1, 0].
        cmap (str, optional): The colormap to use for masks. Defaults to "Blues".

    Returns:
        None
    """

    try:
        from torchgeo.datasets import unbind_samples
    except ImportError as e:
        raise ImportError(
            "Your torchgeo version is too old. Please upgrade to the latest version using 'pip install -U torchgeo'."
        )

    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ("image" in batch) and ("mask" in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ("image" in batch) and ("mask" in batch):
        # plot the images on the even axis
        plot_images(
            images=map(lambda x: x["image"], samples),
            axs=axs.reshape(-1)[::2],
            chnls=chnls,
            bright=bright,
        )

        # plot the masks on the odd axis
        plot_masks(masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1)[1::2])

    else:
        if "image" in batch:
            plot_images(
                images=map(lambda x: x["image"], samples),
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )

        elif "mask" in batch:
            plot_masks(
                masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1), cmap=cmap
            )


def calc_stats(dataset, divide_by: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the statistics (mean and std) for the entire dataset.

    This function is adapted from the plot_batch() function in the torchgeo library at
    https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html.
    Credit to the torchgeo developers for the original implementation.

    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.

    Args:
        dataset (RasterDataset): The dataset to calculate statistics for.
        divide_by (float, optional): The value to divide the image data by. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard deviation for each band.
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object
        for item in dataset.index.intersection(dataset.index.bounds, objects=True)
    ]

    # Resetting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rasterio.open(file).read() / divide_by  # type: ignore
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with length n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


def calc_iou(
    ground_truth: Union[str, np.ndarray, torch.Tensor],
    prediction: Union[str, np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    smooth: float = 1e-6,
    band: int = 1,
) -> Union[float, np.ndarray]:
    """
    Calculate Intersection over Union (IoU) between ground truth and prediction masks.

    This function computes the IoU metric for segmentation tasks. It supports both
    binary and multi-class segmentation, and can handle numpy arrays, PyTorch tensors,
    or file paths to raster files.

    Args:
        ground_truth (Union[str, np.ndarray, torch.Tensor]): Ground truth segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            For binary segmentation: shape (H, W) with values {0, 1}.
            For multi-class segmentation: shape (H, W) with class indices.
        prediction (Union[str, np.ndarray, torch.Tensor]): Predicted segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            Should have the same shape and format as ground_truth.
        num_classes (Optional[int], optional): Number of classes for multi-class segmentation.
            If None, assumes binary segmentation. Defaults to None.
        ignore_index (Optional[int], optional): Class index to ignore in computation.
            Useful for ignoring background or unlabeled pixels. Defaults to None.
        smooth (float, optional): Smoothing factor to avoid division by zero.
            Defaults to 1e-6.
        band (int, optional): Band index to read from raster file (1-based indexing).
            Only used when input is a file path. Defaults to 1.

    Returns:
        Union[float, np.ndarray]: For binary segmentation, returns a single float IoU score.
            For multi-class segmentation, returns an array of IoU scores for each class.

    Examples:
        >>> # Binary segmentation with arrays
        >>> gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> iou = calc_iou(gt, pred)
        >>> print(f"IoU: {iou:.4f}")
        IoU: 0.8333

        >>> # Multi-class segmentation
        >>> gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        >>> iou = calc_iou(gt, pred, num_classes=3)
        >>> print(f"IoU per class: {iou}")
        IoU per class: [0.8333 0.5000 0.5000]

        >>> # Using PyTorch tensors
        >>> gt_tensor = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred_tensor = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> iou = calc_iou(gt_tensor, pred_tensor)
        >>> print(f"IoU: {iou:.4f}")
        IoU: 0.8333

        >>> # Using raster file paths
        >>> iou = calc_iou("ground_truth.tif", "prediction.tif", num_classes=3)
        >>> print(f"Mean IoU: {np.nanmean(iou):.4f}")
        Mean IoU: 0.7500
    """
    # Load from file if string path is provided
    if isinstance(ground_truth, str):
        with rasterio.open(ground_truth) as src:
            ground_truth = src.read(band)
    if isinstance(prediction, str):
        with rasterio.open(prediction) as src:
            prediction = src.read(band)

    # Convert to numpy if torch tensor
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    # Ensure inputs have the same shape
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: ground_truth {ground_truth.shape} vs prediction {prediction.shape}"
        )

    # Binary segmentation
    if num_classes is None:
        ground_truth = ground_truth.astype(bool)
        prediction = prediction.astype(bool)

        intersection = np.logical_and(ground_truth, prediction).sum()
        union = np.logical_or(ground_truth, prediction).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = (intersection + smooth) / (union + smooth)
        return float(iou)

    # Multi-class segmentation
    else:
        iou_per_class = []

        for class_idx in range(num_classes):
            # Handle ignored class by appending np.nan
            if ignore_index is not None and class_idx == ignore_index:
                iou_per_class.append(np.nan)
                continue

            # Create binary masks for current class
            gt_class = (ground_truth == class_idx).astype(bool)
            pred_class = (prediction == class_idx).astype(bool)

            intersection = np.logical_and(gt_class, pred_class).sum()
            union = np.logical_or(gt_class, pred_class).sum()

            if union == 0:
                # If class is not present in both gt and pred
                iou_per_class.append(np.nan)
            else:
                iou_per_class.append((intersection + smooth) / (union + smooth))

        return np.array(iou_per_class)


def calc_f1_score(
    ground_truth: Union[str, np.ndarray, torch.Tensor],
    prediction: Union[str, np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    smooth: float = 1e-6,
    band: int = 1,
) -> Union[float, np.ndarray]:
    """
    Calculate F1 score between ground truth and prediction masks.

    The F1 score is the harmonic mean of precision and recall, computed as:
    F1 = 2 * (precision * recall) / (precision + recall)
    where precision = TP / (TP + FP) and recall = TP / (TP + FN).

    This function supports both binary and multi-class segmentation, and can handle
    numpy arrays, PyTorch tensors, or file paths to raster files.

    Args:
        ground_truth (Union[str, np.ndarray, torch.Tensor]): Ground truth segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            For binary segmentation: shape (H, W) with values {0, 1}.
            For multi-class segmentation: shape (H, W) with class indices.
        prediction (Union[str, np.ndarray, torch.Tensor]): Predicted segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            Should have the same shape and format as ground_truth.
        num_classes (Optional[int], optional): Number of classes for multi-class segmentation.
            If None, assumes binary segmentation. Defaults to None.
        ignore_index (Optional[int], optional): Class index to ignore in computation.
            Useful for ignoring background or unlabeled pixels. Defaults to None.
        smooth (float, optional): Smoothing factor to avoid division by zero.
            Defaults to 1e-6.
        band (int, optional): Band index to read from raster file (1-based indexing).
            Only used when input is a file path. Defaults to 1.

    Returns:
        Union[float, np.ndarray]: For binary segmentation, returns a single float F1 score.
            For multi-class segmentation, returns an array of F1 scores for each class.

    Examples:
        >>> # Binary segmentation with arrays
        >>> gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> f1 = calc_f1_score(gt, pred)
        >>> print(f"F1 Score: {f1:.4f}")
        F1 Score: 0.8571

        >>> # Multi-class segmentation
        >>> gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        >>> f1 = calc_f1_score(gt, pred, num_classes=3)
        >>> print(f"F1 Score per class: {f1}")
        F1 Score per class: [0.8571 0.6667 0.6667]

        >>> # Using PyTorch tensors
        >>> gt_tensor = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred_tensor = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> f1 = calc_f1_score(gt_tensor, pred_tensor)
        >>> print(f"F1 Score: {f1:.4f}")
        F1 Score: 0.8571

        >>> # Using raster file paths
        >>> f1 = calc_f1_score("ground_truth.tif", "prediction.tif", num_classes=3)
        >>> print(f"Mean F1: {np.nanmean(f1):.4f}")
        Mean F1: 0.7302
    """
    # Load from file if string path is provided
    if isinstance(ground_truth, str):
        with rasterio.open(ground_truth) as src:
            ground_truth = src.read(band)
    if isinstance(prediction, str):
        with rasterio.open(prediction) as src:
            prediction = src.read(band)

    # Convert to numpy if torch tensor
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    # Ensure inputs have the same shape
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: ground_truth {ground_truth.shape} vs prediction {prediction.shape}"
        )

    # Binary segmentation
    if num_classes is None:
        ground_truth = ground_truth.astype(bool)
        prediction = prediction.astype(bool)

        # Calculate True Positives, False Positives, False Negatives
        tp = np.logical_and(ground_truth, prediction).sum()
        fp = np.logical_and(~ground_truth, prediction).sum()
        fn = np.logical_and(ground_truth, ~prediction).sum()

        # Calculate precision and recall
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + smooth)
        return float(f1)

    # Multi-class segmentation
    else:
        f1_per_class = []

        for class_idx in range(num_classes):
            # Mark ignored class with np.nan
            if ignore_index is not None and class_idx == ignore_index:
                f1_per_class.append(np.nan)
                continue

            # Create binary masks for current class
            gt_class = (ground_truth == class_idx).astype(bool)
            pred_class = (prediction == class_idx).astype(bool)

            # Calculate True Positives, False Positives, False Negatives
            tp = np.logical_and(gt_class, pred_class).sum()
            fp = np.logical_and(~gt_class, pred_class).sum()
            fn = np.logical_and(gt_class, ~pred_class).sum()

            # Calculate precision and recall
            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)

            # Calculate F1 score
            if tp + fp + fn == 0:
                # If class is not present in both gt and pred
                f1_per_class.append(np.nan)
            else:
                f1 = 2 * (precision * recall) / (precision + recall + smooth)
                f1_per_class.append(f1)

        return np.array(f1_per_class)


def calc_segmentation_metrics(
    ground_truth: Union[str, np.ndarray, torch.Tensor],
    prediction: Union[str, np.ndarray, torch.Tensor],
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    smooth: float = 1e-6,
    metrics: List[str] = ["iou", "f1"],
    band: int = 1,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate multiple segmentation metrics between ground truth and prediction masks.

    This is a convenient wrapper function that computes multiple metrics at once,
    including IoU (Intersection over Union) and F1 score. It supports both binary
    and multi-class segmentation, and can handle numpy arrays, PyTorch tensors,
    or file paths to raster files.

    Args:
        ground_truth (Union[str, np.ndarray, torch.Tensor]): Ground truth segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            For binary segmentation: shape (H, W) with values {0, 1}.
            For multi-class segmentation: shape (H, W) with class indices.
        prediction (Union[str, np.ndarray, torch.Tensor]): Predicted segmentation mask.
            Can be a file path (str) to a raster file, numpy array, or PyTorch tensor.
            Should have the same shape and format as ground_truth.
        num_classes (Optional[int], optional): Number of classes for multi-class segmentation.
            If None, assumes binary segmentation. Defaults to None.
        ignore_index (Optional[int], optional): Class index to ignore in computation.
            Useful for ignoring background or unlabeled pixels. Defaults to None.
        smooth (float, optional): Smoothing factor to avoid division by zero.
            Defaults to 1e-6.
        metrics (List[str], optional): List of metrics to calculate.
            Options: "iou", "f1". Defaults to ["iou", "f1"].
        band (int, optional): Band index to read from raster file (1-based indexing).
            Only used when input is a file path. Defaults to 1.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Dictionary containing the computed metrics.
            Keys are metric names ("iou", "f1"), values are the metric scores.
            For binary segmentation, values are floats.
            For multi-class segmentation, values are numpy arrays with per-class scores.
            Also includes "mean_iou" and "mean_f1" for multi-class segmentation
            (mean computed over valid classes, ignoring NaN values).

    Examples:
        >>> # Binary segmentation with arrays
        >>> gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> metrics = calc_segmentation_metrics(gt, pred)
        >>> print(f"IoU: {metrics['iou']:.4f}, F1: {metrics['f1']:.4f}")
        IoU: 0.8333, F1: 0.8571

        >>> # Multi-class segmentation
        >>> gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        >>> pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        >>> metrics = calc_segmentation_metrics(gt, pred, num_classes=3)
        >>> print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        >>> print(f"Mean F1: {metrics['mean_f1']:.4f}")
        >>> print(f"Per-class IoU: {metrics['iou']}")
        Mean IoU: 0.6111
        Mean F1: 0.7302
        Per-class IoU: [0.8333 0.5000 0.5000]

        >>> # Calculate only IoU
        >>> metrics = calc_segmentation_metrics(gt, pred, num_classes=3, metrics=["iou"])
        >>> print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        Mean IoU: 0.6111

        >>> # Using PyTorch tensors
        >>> gt_tensor = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> pred_tensor = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        >>> metrics = calc_segmentation_metrics(gt_tensor, pred_tensor)
        >>> print(f"IoU: {metrics['iou']:.4f}, F1: {metrics['f1']:.4f}")
        IoU: 0.8333, F1: 0.8571

        >>> # Using raster file paths
        >>> metrics = calc_segmentation_metrics("ground_truth.tif", "prediction.tif", num_classes=3)
        >>> print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        >>> print(f"Mean F1: {metrics['mean_f1']:.4f}")
        Mean IoU: 0.6111
        Mean F1: 0.7302
    """
    results = {}

    # Calculate IoU if requested
    if "iou" in metrics:
        iou = calc_iou(
            ground_truth,
            prediction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=smooth,
            band=band,
        )
        results["iou"] = iou

        # Add mean IoU for multi-class
        if num_classes is not None and isinstance(iou, np.ndarray):
            # Calculate mean ignoring NaN values
            valid_ious = iou[~np.isnan(iou)]
            results["mean_iou"] = (
                float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
            )

    # Calculate F1 score if requested
    if "f1" in metrics:
        f1 = calc_f1_score(
            ground_truth,
            prediction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=smooth,
            band=band,
        )
        results["f1"] = f1

        # Add mean F1 for multi-class
        if num_classes is not None and isinstance(f1, np.ndarray):
            # Calculate mean ignoring NaN values
            valid_f1s = f1[~np.isnan(f1)]
            results["mean_f1"] = (
                float(np.mean(valid_f1s)) if len(valid_f1s) > 0 else 0.0
            )

    return results


def dict_to_rioxarray(data_dict: Dict) -> xr.DataArray:
    """Convert a dictionary to a xarray DataArray. The dictionary should contain the
    following keys: "crs", "bounds", and "image". It can be generated from a TorchGeo
    dataset sampler.

    Args:
        data_dict (Dict): The dictionary containing the data.

    Returns:
        xr.DataArray: The xarray DataArray.
    """

    from collections import namedtuple

    from affine import Affine

    BoundingBox = namedtuple("BoundingBox", ["minx", "maxx", "miny", "maxy"])

    # Extract components from the dictionary
    crs = data_dict["crs"]
    bounds = data_dict["bounds"]
    image_tensor = data_dict["image"]

    if hasattr(bounds, "left"):
        bounds = BoundingBox(bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Convert tensor to numpy array if needed
    if hasattr(image_tensor, "numpy"):
        # For PyTorch tensors
        image_array = image_tensor.numpy()
    else:
        # If it's already a numpy array or similar
        image_array = np.array(image_tensor)

    # Calculate pixel resolution
    width = image_array.shape[2]  # Width is the size of the last dimension
    height = image_array.shape[1]  # Height is the size of the middle dimension

    res_x = (bounds.maxx - bounds.minx) / width
    res_y = (bounds.maxy - bounds.miny) / height

    # Create the transform matrix
    transform = Affine(res_x, 0.0, bounds.minx, 0.0, -res_y, bounds.maxy)

    # Create dimensions
    x_coords = np.linspace(bounds.minx + res_x / 2, bounds.maxx - res_x / 2, width)
    y_coords = np.linspace(bounds.maxy - res_y / 2, bounds.miny + res_y / 2, height)

    # If time dimension exists in the bounds
    if hasattr(bounds, "mint") and hasattr(bounds, "maxt"):
        # Create a single time value or range if needed
        t_coords = [
            bounds.mint
        ]  # Or np.linspace(bounds.mint, bounds.maxt, num_time_steps)

        # Create DataArray with time dimension
        dims = (
            ("band", "y", "x")
            if image_array.shape[0] <= 10
            else ("time", "band", "y", "x")
        )

        if dims[0] == "band":
            # For multi-band single time
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "band": np.arange(1, image_array.shape[0] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
        else:
            # For multi-time multi-band
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "time": t_coords,
                    "band": np.arange(1, image_array.shape[1] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
    else:
        # Create DataArray without time dimension
        da = xr.DataArray(
            image_array,
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(1, image_array.shape[0] + 1),
                "y": y_coords,
                "x": x_coords,
            },
        )

    # Set spatial attributes
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)

    return da


def dict_to_image(
    data_dict: Dict[str, Any], output: Optional[str] = None, **kwargs: Any
) -> Union[str, Any]:
    """Convert a dictionary containing spatial data to a rasterio dataset or save it to
    a file. The dictionary should contain the following keys: "crs", "bounds", and "image".
    It can be generated from a TorchGeo dataset sampler.

    This function transforms a dictionary with CRS, bounding box, and image data
    into a rasterio DatasetReader using leafmap's array_to_image utility after
    first converting to a rioxarray DataArray.

    Args:
        data_dict: A dictionary containing:
            - 'crs': A pyproj CRS object
            - 'bounds': A BoundingBox object with minx, maxx, miny, maxy attributes
              and optionally mint, maxt for temporal bounds
            - 'image': A tensor or array-like object with image data
        output: Optional path to save the image to a file. If not provided, the image
            will be returned as a rasterio DatasetReader object.
        **kwargs: Additional keyword arguments to pass to leafmap.array_to_image.
            Common options include:
            - colormap: str, name of the colormap (e.g., 'viridis', 'terrain')
            - vmin: float, minimum value for colormap scaling
            - vmax: float, maximum value for colormap scaling

    Returns:
        A rasterio DatasetReader object that can be used for visualization or
        further processing.

    Examples:
        >>> image = dict_to_image(
        ...     {'crs': CRS.from_epsg(26911), 'bounds': bbox, 'image': tensor},
        ...     colormap='terrain'
        ... )
        >>> fig, ax = plt.subplots(figsize=(10, 10))
        >>> show(image, ax=ax)
    """
    da = dict_to_rioxarray(data_dict)

    if output is not None:
        out_dir = os.path.abspath(os.path.dirname(output))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        da.rio.to_raster(output)
        return output
    else:
        image = leafmap.array_to_image(da, **kwargs)
        return image


def view_vector(
    vector_data: Union[str, gpd.GeoDataFrame],
    column: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    legend: bool = True,
    basemap: bool = False,
    basemap_type: str = "streets",
    alpha: float = 0.7,
    edge_color: str = "black",
    classification: str = "quantiles",
    n_classes: int = 5,
    highlight_index: Optional[int] = None,
    highlight_color: str = "red",
    scheme: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Any:
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, and background basemaps.

    Args:
        vector_data (geopandas.GeoDataFrame): The vector dataset to visualize.
        column (str, optional): Column to use for choropleth mapping. If None,
            a single color will be used. Defaults to None.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap to use for
            choropleth mapping. Defaults to "viridis".
        figsize (tuple, optional): Figure size as (width, height) in inches.
            Defaults to (10, 10).
        title (str, optional): Title for the plot. Defaults to None.
        legend (bool, optional): Whether to display a legend. Defaults to True.
        basemap (bool, optional): Whether to add a web basemap. Requires contextily.
            Defaults to False.
        basemap_type (str, optional): Type of basemap to use. Options: 'streets', 'satellite'.
            Defaults to 'streets'.
        alpha (float, optional): Transparency of the vector features, between 0-1.
            Defaults to 0.7.
        edge_color (str, optional): Color for feature edges. Defaults to "black".
        classification (str, optional): Classification method for choropleth maps.
            Options: "quantiles", "equal_interval", "natural_breaks".
            Defaults to "quantiles".
        n_classes (int, optional): Number of classes for choropleth maps.
            Defaults to 5.
        highlight_index (list, optional): List of indices to highlight.
            Defaults to None.
        highlight_color (str, optional): Color to use for highlighted features.
            Defaults to "red".
        scheme (str, optional): MapClassify classification scheme. Overrides
            classification parameter if provided. Defaults to None.
        save_path (str, optional): Path to save the figure. If None, the figure
            is not saved. Defaults to None.
        dpi (int, optional): DPI for saved figure. Defaults to 300.

    Returns:
        matplotlib.axes.Axes: The Axes object containing the plot.

    Examples:
        >>> import geopandas as gpd
        >>> cities = gpd.read_file("cities.shp")
        >>> view_vector(cities, "population", cmap="Reds", basemap=True)

        >>> roads = gpd.read_file("roads.shp")
        >>> view_vector(roads, "type", basemap=True, figsize=(12, 8))
    """
    import contextily as ctx

    if isinstance(vector_data, str):
        vector_data = gpd.read_file(vector_data)

    # Check if input is a GeoDataFrame
    if not isinstance(vector_data, gpd.GeoDataFrame):
        raise TypeError("Input data must be a GeoDataFrame")

    # Make a copy to avoid changing the original data
    gdf = vector_data.copy()

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Determine geometry type
    geom_type = gdf.geometry.iloc[0].geom_type

    # Plotting parameters
    plot_kwargs = {"alpha": alpha, "ax": ax}

    # Set up keyword arguments based on geometry type
    if "Point" in geom_type:
        plot_kwargs["markersize"] = 50
        plot_kwargs["edgecolor"] = edge_color
    elif "Line" in geom_type:
        plot_kwargs["linewidth"] = 1
    elif "Polygon" in geom_type:
        plot_kwargs["edgecolor"] = edge_color

    # Classification options
    if column is not None:
        if scheme is not None:
            # Use mapclassify scheme if provided
            plot_kwargs["scheme"] = scheme
        else:
            # Use classification parameter
            if classification == "quantiles":
                plot_kwargs["scheme"] = "quantiles"
            elif classification == "equal_interval":
                plot_kwargs["scheme"] = "equal_interval"
            elif classification == "natural_breaks":
                plot_kwargs["scheme"] = "fisher_jenks"

        plot_kwargs["k"] = n_classes
        plot_kwargs["cmap"] = cmap
        plot_kwargs["column"] = column
        plot_kwargs["legend"] = legend

    # Plot the main data
    gdf.plot(**plot_kwargs)

    # Highlight specific features if requested
    if highlight_index is not None:
        gdf.iloc[highlight_index].plot(
            ax=ax, color=highlight_color, edgecolor="black", linewidth=2, zorder=5
        )

    if basemap:
        try:
            basemap_options = {
                "streets": ctx.providers.OpenStreetMap.Mapnik,
                "satellite": ctx.providers.Esri.WorldImagery,
            }
            ctx.add_basemap(ax, crs=gdf.crs, source=basemap_options[basemap_type])
        except Exception as e:
            print(f"Could not add basemap: {e}")

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)

    # Remove axes if not needed
    ax.set_axis_off()

    # Adjust layout
    plt.tight_layout()

    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return ax


def view_vector_interactive(
    vector_data: Union[str, gpd.GeoDataFrame],
    layer_name: str = "Vector",
    tiles_args: Optional[Dict] = None,
    opacity: float = 0.7,
    **kwargs: Any,
) -> Any:
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, and background basemaps.

    Args:
        vector_data (geopandas.GeoDataFrame): The vector dataset to visualize.
        layer_name (str, optional): The name of the layer. Defaults to "Vector".
        tiles_args (dict, optional): Additional arguments for the localtileserver client.
            get_folium_tile_layer function. Defaults to None.
        opacity (float, optional): The opacity of the layer. Defaults to 0.7.
        **kwargs: Additional keyword arguments to pass to GeoDataFrame.explore() function.
            See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html

    Returns:
        folium.Map: The map object with the vector data added.

    Examples:
        >>> import geopandas as gpd
        >>> cities = gpd.read_file("cities.shp")
        >>> view_vector_interactive(cities)

        >>> roads = gpd.read_file("roads.shp")
        >>> view_vector_interactive(roads, figsize=(12, 8))
    """

    from leafmap.foliumap import Map
    from localtileserver import TileClient, get_folium_tile_layer

    google_tiles = {
        "Roadmap": {
            "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Maps",
        },
        "Satellite": {
            "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
        "Terrain": {
            "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Terrain",
        },
        "Hybrid": {
            "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Hybrid",
        },
    }

    # Make it compatible with binder and JupyterHub
    if os.environ.get("JUPYTERHUB_SERVICE_PREFIX") is not None:
        os.environ["LOCALTILESERVER_CLIENT_PREFIX"] = (
            f"{os.environ['JUPYTERHUB_SERVICE_PREFIX'].lstrip('/')}/proxy/{{port}}"
        )

    basemap_layer_name = None
    raster_layer = None

    m = Map()

    if "tiles" in kwargs and isinstance(kwargs["tiles"], str):
        if kwargs["tiles"].title() in google_tiles:
            basemap_layer_name = google_tiles[kwargs["tiles"].title()]["name"]
            kwargs["tiles"] = google_tiles[kwargs["tiles"].title()]["url"]
            kwargs["attr"] = "Google"
        elif kwargs["tiles"].lower().endswith(".tif"):
            if tiles_args is None:
                tiles_args = {}
            if kwargs["tiles"].lower().startswith("http"):
                basemap_layer_name = "Remote Raster"
                m.add_geotiff(kwargs["tiles"], name=basemap_layer_name, **tiles_args)
            else:
                basemap_layer_name = "Local Raster"
                client = TileClient(kwargs["tiles"])
                raster_layer = get_folium_tile_layer(client, **tiles_args)
                m.add_tile_layer(
                    raster_layer.tiles,
                    name=basemap_layer_name,
                    attribution="localtileserver",
                    **tiles_args,
                )

    if "max_zoom" not in kwargs:
        kwargs["max_zoom"] = 30

    if isinstance(vector_data, str):
        if vector_data.endswith(".parquet"):
            vector_data = gpd.read_parquet(vector_data)
        else:
            vector_data = gpd.read_file(vector_data)

    # Check if input is a GeoDataFrame
    if not isinstance(vector_data, gpd.GeoDataFrame):
        raise TypeError("Input data must be a GeoDataFrame")

    if "column" in kwargs:
        if "legend_position" not in kwargs:
            kwargs["legend_position"] = "bottomleft"
        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
        m.add_data(vector_data, layer_name=layer_name, opacity=opacity, **kwargs)

    else:
        m.add_gdf(vector_data, layer_name=layer_name, opacity=opacity, **kwargs)

    m.add_layer_control()
    m.add_opacity_control()

    return m


def regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    angle_tolerance: float = 10,
    simplify_tolerance: float = 0.5,
    orthogonalize: bool = True,
    preserve_topology: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Regularizes building footprint polygons with multiple techniques beyond minimum
    rotated rectangles.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints
        angle_tolerance: Degrees within which angles will be regularized to 90/180 degrees
        simplify_tolerance: Distance tolerance for Douglas-Peucker simplification
        orthogonalize: Whether to enforce orthogonal angles in the final polygons
        preserve_topology: Whether to preserve topology during simplification

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely import wkt
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, shape

    regularized_buildings = []

    # Check if we're dealing with a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    for building in geom_objects:
        # Handle potential string representations of geometries
        if isinstance(building, str):
            try:
                # Try to parse as WKT
                building = wkt.loads(building)
            except Exception:
                print(f"Failed to parse geometry string: {building[:30]}...")
                continue

        # Ensure we have a valid geometry
        if not hasattr(building, "simplify"):
            print(f"Invalid geometry type: {type(building)}")
            continue

        # Step 1: Simplify to remove noise and small vertices
        simplified = building.simplify(
            simplify_tolerance, preserve_topology=preserve_topology
        )

        if orthogonalize:
            # Make sure we have a valid polygon with an exterior
            if not hasattr(simplified, "exterior") or simplified.exterior is None:
                print(f"Simplified geometry has no exterior: {simplified}")
                regularized_buildings.append(building)  # Use original instead
                continue

            # Step 2: Get the dominant angle to rotate building
            coords = np.array(simplified.exterior.coords)

            # Make sure we have enough coordinates for angle calculation
            if len(coords) < 3:
                print(f"Not enough coordinates for angle calculation: {len(coords)}")
                regularized_buildings.append(building)  # Use original instead
                continue

            segments = np.diff(coords, axis=0)
            angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

            # Find most common angle classes (0, 90, 180, 270 degrees)
            binned_angles = np.round(angles / 90) * 90
            dominant_angle = np.bincount(binned_angles.astype(int) % 180).argmax()

            # Step 3: Rotate to align with axes, regularize, then rotate back
            rotated = rotate(simplified, -dominant_angle, origin="centroid")

            # Step 4: Rectify coordinates to enforce right angles
            ext_coords = np.array(rotated.exterior.coords)
            rect_coords = []

            # Regularize each vertex to create orthogonal corners
            for i in range(len(ext_coords) - 1):
                rect_coords.append(ext_coords[i])

                # Check if we need to add a right-angle vertex
                angle = (
                    np.arctan2(
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 1]
                        - ext_coords[i, 1],
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 0]
                        - ext_coords[i, 0],
                    )
                    * 180
                    / np.pi
                )

                if abs(angle % 90) > angle_tolerance and abs(angle % 90) < (
                    90 - angle_tolerance
                ):
                    # Add intermediate point to create right angle
                    rect_coords.append(
                        [
                            ext_coords[(i + 1) % (len(ext_coords) - 1), 0],
                            ext_coords[i, 1],
                        ]
                    )

            # Close the polygon by adding the first point again
            rect_coords.append(rect_coords[0])

            # Create regularized polygon and rotate back
            regularized = Polygon(rect_coords)
            final_building = rotate(regularized, dominant_angle, origin="centroid")
        else:
            final_building = simplified

        regularized_buildings.append(final_building)

    # If input was a GeoDataFrame, return a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(
            geometry=regularized_buildings, crs=building_polygons.crs
        )
    else:
        return regularized_buildings


def hybrid_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    A comprehensive hybrid approach to building footprint regularization.

    Applies different strategies based on building characteristics.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Use minimum_rotated_rectangle instead of oriented_envelope
    try:
        from shapely.minimum_rotated_rectangle import minimum_rotated_rectangle
    except ImportError:
        # For older Shapely versions
        def minimum_rotated_rectangle(geom):
            """Calculate the minimum rotated rectangle for a geometry"""
            # For older Shapely versions, implement a simple version
            return geom.minimum_rotated_rectangle

    # Determine input type for correct return
    is_gdf = isinstance(building_polygons, gpd.GeoDataFrame)

    # Extract geometries if GeoDataFrame
    if is_gdf:
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # 1. Analyze building characteristics
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Calculate shape complexity metrics
        complexity = building.length / (4 * np.sqrt(building.area))

        # Calculate dominant angle
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        segment_angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Weight angles by segment length
        hist, bins = np.histogram(
            segment_angles % 180, bins=36, range=(0, 180), weights=segment_lengths
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        dominant_angle = bin_centers[np.argmax(hist)]

        # Check if building is close to orthogonal
        is_orthogonal = min(dominant_angle % 45, 45 - (dominant_angle % 45)) < 5

        # 2. Apply appropriate regularization strategy
        if complexity > 1.5:
            # Complex buildings: use minimum rotated rectangle
            result = minimum_rotated_rectangle(building)
        elif is_orthogonal:
            # Near-orthogonal buildings: orthogonalize in place
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create orthogonal hull in rotated space
            bounds = rotated.bounds
            ortho_hull = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            result = rotate(ortho_hull, dominant_angle, origin="centroid")
        else:
            # Diagonal buildings: use custom approach for diagonal buildings
            # Rotate to align with axes
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Simplify in rotated space
            simplified = rotated.simplify(0.3, preserve_topology=True)

            # Get the bounds in rotated space
            bounds = simplified.bounds
            min_x, min_y, max_x, max_y = bounds

            # Create a rectangular hull in rotated space
            rect_poly = Polygon(
                [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            )

            # Rotate back to original orientation
            result = rotate(rect_poly, dominant_angle, origin="centroid")

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def adaptive_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    simplify_tolerance: float = 0.5,
    area_threshold: float = 0.9,
    preserve_shape: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Adaptively regularizes building footprints based on their characteristics.

    This approach determines the best regularization method for each building.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons
        simplify_tolerance: Distance tolerance for simplification
        area_threshold: Minimum acceptable area ratio
        preserve_shape: Whether to preserve overall shape for complex buildings

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Analyze the overall dataset to set appropriate parameters
    if is_gdf := isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # Skip invalid geometries
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Measure building complexity
        complexity = building.length / (4 * np.sqrt(building.area))

        # Determine if the building has a clear principal direction
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Normalize angles to 0-180 range and get histogram
        norm_angles = angles % 180
        hist, bins = np.histogram(
            norm_angles, bins=18, range=(0, 180), weights=segment_lengths
        )

        # Calculate direction clarity (ratio of longest direction to total)
        direction_clarity = np.max(hist) / np.sum(hist) if np.sum(hist) > 0 else 0

        # Choose regularization method based on building characteristics
        if complexity < 1.2 and direction_clarity > 0.5:
            # Simple building with clear direction: use rotated rectangle
            bin_max = np.argmax(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            dominant_angle = bin_centers[bin_max]

            # Rotate to align with coordinate system
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create bounding box in rotated space
            bounds = rotated.bounds
            rect = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            # Rotate back
            result = rotate(rect, dominant_angle, origin="centroid")

            # Quality check
            if (
                result.area / building.area < area_threshold
                or result.area / building.area > (1.0 / area_threshold)
            ):
                # Too much area change, use simplified original
                result = building.simplify(simplify_tolerance, preserve_topology=True)

        else:
            # Complex building or no clear direction: preserve shape
            if preserve_shape:
                # Simplify with topology preservation
                result = building.simplify(simplify_tolerance, preserve_topology=True)
            else:
                # Fall back to convex hull for very complex shapes
                result = building.convex_hull

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def install_package(package: Union[str, List[str]]) -> None:
    """Install a Python package.

    Args:
        package (str | list): The package name or a GitHub URL or a list of package names or GitHub URLs.
    """
    import subprocess

    if isinstance(package, str):
        packages = [package]
    elif isinstance(package, list):
        packages = package
    else:
        raise ValueError("The package argument must be a string or a list of strings.")

    for package in packages:
        if package.startswith("https"):
            package = f"git+{package}"

        # Execute pip install command and show output in real-time
        command = f"pip install {package}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.decode("utf-8").strip())

        # Wait for process to complete
        process.wait()


def create_split_map(
    left_layer: Optional[str] = "TERRAIN",
    right_layer: Optional[str] = "OpenTopoMap",
    left_args: Optional[dict] = None,
    right_args: Optional[dict] = None,
    left_array_args: Optional[dict] = None,
    right_array_args: Optional[dict] = None,
    zoom_control: Optional[bool] = True,
    fullscreen_control: Optional[bool] = True,
    layer_control: Optional[bool] = True,
    add_close_button: Optional[bool] = False,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    left_position: Optional[str] = "bottomleft",
    right_position: Optional[str] = "bottomright",
    widget_layout: Optional[dict] = None,
    draggable: Optional[bool] = True,
    center: Optional[List[float]] = [20, 0],
    zoom: Optional[int] = 2,
    height: Optional[int] = "600px",
    basemap: Optional[str] = None,
    basemap_args: Optional[Dict] = None,
    m: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Adds split map.

    Args:
        left_layer (str, optional): The left tile layer. Can be a local file path, HTTP URL, or a basemap name. Defaults to 'TERRAIN'.
        right_layer (str, optional): The right tile layer. Can be a local file path, HTTP URL, or a basemap name. Defaults to 'OpenTopoMap'.
        left_args (dict, optional): The arguments for the left tile layer. Defaults to {}.
        right_args (dict, optional): The arguments for the right tile layer. Defaults to {}.
        left_array_args (dict, optional): The arguments for array_to_image for the left layer. Defaults to {}.
        right_array_args (dict, optional): The arguments for array_to_image for the right layer. Defaults to {}.
        zoom_control (bool, optional): Whether to add zoom control. Defaults to True.
        fullscreen_control (bool, optional): Whether to add fullscreen control. Defaults to True.
        layer_control (bool, optional): Whether to add layer control. Defaults to True.
        add_close_button (bool, optional): Whether to add a close button. Defaults to False.
        left_label (str, optional): The label for the left layer. Defaults to None.
        right_label (str, optional): The label for the right layer. Defaults to None.
        left_position (str, optional): The position for the left label. Defaults to "bottomleft".
        right_position (str, optional): The position for the right label. Defaults to "bottomright".
        widget_layout (dict, optional): The layout for the widget. Defaults to None.
        draggable (bool, optional): Whether the split map is draggable. Defaults to True.
    """

    if left_args is None:
        left_args = {}

    if right_args is None:
        right_args = {}

    if left_array_args is None:
        left_array_args = {}

    if right_array_args is None:
        right_array_args = {}

    if basemap_args is None:
        basemap_args = {}

    if m is None:
        m = leafmap.Map(center=center, zoom=zoom, height=height, **kwargs)
        m.clear_layers()
    if isinstance(basemap, str):
        if basemap.endswith(".tif"):
            if basemap.startswith("http"):
                m.add_cog_layer(basemap, name="Basemap", **basemap_args)
            else:
                m.add_raster(basemap, layer_name="Basemap", **basemap_args)
        else:
            m.add_basemap(basemap)
    m.split_map(
        left_layer=left_layer,
        right_layer=right_layer,
        left_args=left_args,
        right_args=right_args,
        left_array_args=left_array_args,
        right_array_args=right_array_args,
        zoom_control=zoom_control,
        fullscreen_control=fullscreen_control,
        layer_control=layer_control,
        add_close_button=add_close_button,
        left_label=left_label,
        right_label=right_label,
        left_position=left_position,
        right_position=right_position,
        widget_layout=widget_layout,
        draggable=draggable,
    )

    return m


def download_file(
    url: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    unzip: bool = True,
) -> str:
    """
    Download a file from a given URL with a progress bar.
    Optionally unzip the file if it's a ZIP archive.

    Args:
        url (str): The URL of the file to download.
        output_path (str, optional): The path where the downloaded file will be saved.
            If not provided, the filename from the URL will be used.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
        unzip (bool, optional): Whether to unzip the file if it is a ZIP archive.

    Returns:
        str: The path to the downloaded file or the extracted directory.
    """

    import zipfile

    from tqdm import tqdm

    if output_path is None:
        output_path = os.path.basename(url)

    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists: {output_path}")
    else:
        # Download the file with a progress bar
        response = requests.get(url, stream=True, timeout=50)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(output_path, "wb") as file,
            tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    # If the file is a ZIP archive and unzip is True
    if unzip and zipfile.is_zipfile(output_path):
        extract_dir = os.path.splitext(output_path)[0]
        if not os.path.exists(extract_dir) or overwrite:
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        return extract_dir

    return output_path


def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """Display basic information about a raster dataset.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        dict: Dictionary containing the basic information about the raster
    """
    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Get basic metadata
        info = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else "No CRS defined",
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": (src.transform[0], -src.transform[4]),
            "nodata": src.nodata,
        }

        # Calculate statistics for each band
        stats = []
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)
            band_stats = {
                "band": i,
                "min": float(band.min()),
                "max": float(band.max()),
                "mean": float(band.mean()),
                "std": float(band.std()),
            }
            stats.append(band_stats)

        info["band_stats"] = stats

    return info


def get_raster_stats(raster_path: str, divide_by: float = 1.0) -> Dict[str, Any]:
    """Calculate statistics for each band in a raster dataset.

    This function computes min, max, mean, and standard deviation values
    for each band in the provided raster, returning results in a dictionary
    with lists for each statistic type.

    Args:
        raster_path (str): Path to the raster file
        divide_by (float, optional): Value to divide pixel values by.
            Defaults to 1.0, which keeps the original pixel

    Returns:
        dict: Dictionary containing lists of statistics with keys:
            - 'min': List of minimum values for each band
            - 'max': List of maximum values for each band
            - 'mean': List of mean values for each band
            - 'std': List of standard deviation values for each band
    """
    # Initialize the results dictionary with empty lists
    stats = {"min": [], "max": [], "mean": [], "std": []}

    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Calculate statistics for each band
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)

            # Append statistics for this band to each list
            stats["min"].append(float(band.min()) / divide_by)
            stats["max"].append(float(band.max()) / divide_by)
            stats["mean"].append(float(band.mean()) / divide_by)
            stats["std"].append(float(band.std()) / divide_by)

    return stats


def print_raster_info(
    raster_path: str, show_preview: bool = True, figsize: Tuple[int, int] = (10, 8)
) -> Optional[Dict[str, Any]]:
    """Print formatted information about a raster dataset and optionally show a preview.

    Args:
        raster_path (str): Path to the raster file
        show_preview (bool, optional): Whether to display a visual preview of the raster.
            Defaults to True.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        dict: Dictionary containing raster information if successful, None otherwise
    """
    try:
        info = get_raster_info(raster_path)

        # Print basic information
        print(f"===== RASTER INFORMATION: {raster_path} =====")
        print(f"Driver: {info['driver']}")
        print(f"Dimensions: {info['width']} x {info['height']} pixels")
        print(f"Number of bands: {info['count']}")
        print(f"Data type: {info['dtype']}")
        print(f"Coordinate Reference System: {info['crs']}")
        print(f"Georeferenced Bounds: {info['bounds']}")
        print(f"Pixel Resolution: {info['resolution'][0]}, {info['resolution'][1]}")
        print(f"NoData Value: {info['nodata']}")

        # Print band statistics
        print("\n----- Band Statistics -----")
        for band_stat in info["band_stats"]:
            print(f"Band {band_stat['band']}:")
            print(f"  Min: {band_stat['min']:.2f}")
            print(f"  Max: {band_stat['max']:.2f}")
            print(f"  Mean: {band_stat['mean']:.2f}")
            print(f"  Std Dev: {band_stat['std']:.2f}")

        # Show a preview if requested
        if show_preview:
            with rasterio.open(raster_path) as src:
                # For multi-band images, show RGB composite or first band
                if src.count >= 3:
                    # Try to show RGB composite
                    rgb = np.dstack([src.read(i) for i in range(1, 4)])
                    plt.figure(figsize=figsize)
                    plt.imshow(rgb)
                    plt.title(f"RGB Preview: {raster_path}")
                else:
                    # Show first band for single-band images
                    plt.figure(figsize=figsize)
                    show(
                        src.read(1),
                        cmap="viridis",
                        title=f"Band 1 Preview: {raster_path}",
                    )
                    plt.colorbar(label="Pixel Value")
                plt.show()

    except Exception as e:
        print(f"Error reading raster: {str(e)}")


def get_raster_info_gdal(raster_path: str) -> Optional[Dict[str, Any]]:
    """Get basic information about a raster dataset using GDAL.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        dict: Dictionary containing the basic information about the raster,
            or None if the file cannot be opened
    """

    from osgeo import gdal

    # Open the dataset
    ds = gdal.Open(raster_path)
    if ds is None:
        print(f"Error: Could not open {raster_path}")
        return None

    # Get basic information
    info = {
        "driver": ds.GetDriver().ShortName,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "count": ds.RasterCount,
        "projection": ds.GetProjection(),
        "geotransform": ds.GetGeoTransform(),
    }

    # Calculate resolution
    gt = ds.GetGeoTransform()
    if gt:
        info["resolution"] = (abs(gt[1]), abs(gt[5]))
        info["origin"] = (gt[0], gt[3])

    # Get band information
    bands_info = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        stats = band.GetStatistics(True, True)
        band_info = {
            "band": i,
            "datatype": gdal.GetDataTypeName(band.DataType),
            "min": stats[0],
            "max": stats[1],
            "mean": stats[2],
            "std": stats[3],
            "nodata": band.GetNoDataValue(),
        }
        bands_info.append(band_info)

    info["bands"] = bands_info

    # Close the dataset
    ds = None

    return info


def get_vector_info(vector_path: str) -> Optional[Dict[str, Any]]:
    """Display basic information about a vector dataset using GeoPandas.

    Args:
        vector_path (str): Path to the vector file

    Returns:
        dict: Dictionary containing the basic information about the vector dataset
    """
    # Open the vector dataset
    gdf = (
        gpd.read_parquet(vector_path)
        if vector_path.endswith(".parquet")
        else gpd.read_file(vector_path)
    )

    # Get basic metadata
    info = {
        "file_path": vector_path,
        "driver": os.path.splitext(vector_path)[1][1:].upper(),  # Format from extension
        "feature_count": len(gdf),
        "crs": str(gdf.crs),
        "geometry_type": str(gdf.geom_type.value_counts().to_dict()),
        "attribute_count": len(gdf.columns) - 1,  # Subtract the geometry column
        "attribute_names": list(gdf.columns[gdf.columns != "geometry"]),
        "bounds": gdf.total_bounds.tolist(),
    }

    # Add statistics about numeric attributes
    numeric_columns = gdf.select_dtypes(include=["number"]).columns
    attribute_stats = {}
    for col in numeric_columns:
        if col != "geometry":
            attribute_stats[col] = {
                "min": gdf[col].min(),
                "max": gdf[col].max(),
                "mean": gdf[col].mean(),
                "std": gdf[col].std(),
                "null_count": gdf[col].isna().sum(),
            }

    info["attribute_stats"] = attribute_stats

    return info


def print_vector_info(
    vector_path: str, show_preview: bool = True, figsize: Tuple[int, int] = (10, 8)
) -> Optional[Dict[str, Any]]:
    """Print formatted information about a vector dataset and optionally show a preview.

    Args:
        vector_path (str): Path to the vector file
        show_preview (bool, optional): Whether to display a visual preview of the vector data.
            Defaults to True.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        dict: Dictionary containing vector information if successful, None otherwise
    """
    try:
        info = get_vector_info(vector_path)

        # Print basic information
        print(f"===== VECTOR INFORMATION: {vector_path} =====")
        print(f"Driver: {info['driver']}")
        print(f"Feature count: {info['feature_count']}")
        print(f"Geometry types: {info['geometry_type']}")
        print(f"Coordinate Reference System: {info['crs']}")
        print(f"Bounds: {info['bounds']}")
        print(f"Number of attributes: {info['attribute_count']}")
        print(f"Attribute names: {', '.join(info['attribute_names'])}")

        # Print attribute statistics
        if info["attribute_stats"]:
            print("\n----- Attribute Statistics -----")
            for attr, stats in info["attribute_stats"].items():
                print(f"Attribute: {attr}")
                for stat_name, stat_value in stats.items():
                    print(
                        f"  {stat_name}: {stat_value:.4f}"
                        if isinstance(stat_value, float)
                        else f"  {stat_name}: {stat_value}"
                    )

        # Show a preview if requested
        if show_preview:
            gdf = (
                gpd.read_parquet(vector_path)
                if vector_path.endswith(".parquet")
                else gpd.read_file(vector_path)
            )
            fig, ax = plt.subplots(figsize=figsize)
            gdf.plot(ax=ax, cmap="viridis")
            ax.set_title(f"Preview: {vector_path}")
            plt.tight_layout()
            plt.show()

            # # Show a sample of the attribute table
            # if not gdf.empty:
            #     print("\n----- Sample of attribute table (first 5 rows) -----")
            #     print(gdf.head().to_string())

    except Exception as e:
        print(f"Error reading vector data: {str(e)}")


# Alternative implementation using OGR directly
def get_vector_info_ogr(vector_path: str) -> Optional[Dict[str, Any]]:
    """Get basic information about a vector dataset using OGR.

    Args:
        vector_path (str): Path to the vector file

    Returns:
        dict: Dictionary containing the basic information about the vector dataset,
            or None if the file cannot be opened
    """
    from osgeo import ogr

    # Register all OGR drivers
    ogr.RegisterAll()

    # Open the dataset
    ds = ogr.Open(vector_path)
    if ds is None:
        print(f"Error: Could not open {vector_path}")
        return None

    # Basic dataset information
    info = {
        "file_path": vector_path,
        "driver": ds.GetDriver().GetName(),
        "layer_count": ds.GetLayerCount(),
        "layers": [],
    }

    # Extract information for each layer
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i)
        layer_info = {
            "name": layer.GetName(),
            "feature_count": layer.GetFeatureCount(),
            "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
            "spatial_ref": (
                layer.GetSpatialRef().ExportToWkt() if layer.GetSpatialRef() else "None"
            ),
            "extent": layer.GetExtent(),
            "fields": [],
        }

        # Get field information
        defn = layer.GetLayerDefn()
        for j in range(defn.GetFieldCount()):
            field_defn = defn.GetFieldDefn(j)
            field_info = {
                "name": field_defn.GetName(),
                "type": field_defn.GetTypeName(),
                "width": field_defn.GetWidth(),
                "precision": field_defn.GetPrecision(),
            }
            layer_info["fields"].append(field_info)

        info["layers"].append(layer_info)

    # Close the dataset
    ds = None

    return info


def analyze_vector_attributes(
    vector_path: str, attribute_name: str
) -> Optional[Dict[str, Any]]:
    """Analyze a specific attribute in a vector dataset and create a histogram.

    Args:
        vector_path (str): Path to the vector file
        attribute_name (str): Name of the attribute to analyze

    Returns:
        dict: Dictionary containing analysis results for the attribute
    """
    try:
        gdf = gpd.read_file(vector_path)

        # Check if attribute exists
        if attribute_name not in gdf.columns:
            print(f"Attribute '{attribute_name}' not found in the dataset")
            return None

        # Get the attribute series
        attr = gdf[attribute_name]

        # Perform different analyses based on data type
        if pd.api.types.is_numeric_dtype(attr):
            # Numeric attribute
            analysis = {
                "attribute": attribute_name,
                "type": "numeric",
                "count": attr.count(),
                "null_count": attr.isna().sum(),
                "min": attr.min(),
                "max": attr.max(),
                "mean": attr.mean(),
                "median": attr.median(),
                "std": attr.std(),
                "unique_values": attr.nunique(),
            }

            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(attr.dropna(), bins=20, alpha=0.7, color="blue")
            plt.title(f"Histogram of {attribute_name}")
            plt.xlabel(attribute_name)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.show()

        else:
            # Categorical attribute
            analysis = {
                "attribute": attribute_name,
                "type": "categorical",
                "count": attr.count(),
                "null_count": attr.isna().sum(),
                "unique_values": attr.nunique(),
                "value_counts": attr.value_counts().to_dict(),
            }

            # Create bar plot for top categories
            top_n = min(10, attr.nunique())
            plt.figure(figsize=(10, 6))
            attr.value_counts().head(top_n).plot(kind="bar", color="skyblue")
            plt.title(f"Top {top_n} values for {attribute_name}")
            plt.xlabel(attribute_name)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return analysis

    except Exception as e:
        print(f"Error analyzing attribute: {str(e)}")
        return None


def visualize_vector_by_attribute(
    vector_path: str,
    attribute_name: str,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
) -> bool:
    """Create a thematic map visualization of vector data based on an attribute.

    Args:
        vector_path (str): Path to the vector file
        attribute_name (str): Name of the attribute to visualize
        cmap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        bool: True if visualization was successful, False otherwise
    """
    try:
        # Read the vector data
        gdf = gpd.read_file(vector_path)

        # Check if attribute exists
        if attribute_name not in gdf.columns:
            print(f"Attribute '{attribute_name}' not found in the dataset")
            return False

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Determine plot type based on data type
        if pd.api.types.is_numeric_dtype(gdf[attribute_name]):
            # Continuous data
            gdf.plot(column=attribute_name, cmap=cmap, legend=True, ax=ax)
        else:
            # Categorical data
            gdf.plot(column=attribute_name, categorical=True, legend=True, ax=ax)

        # Add title and labels
        ax.set_title(f"{os.path.basename(vector_path)} - {attribute_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add basemap or additional elements if available
        # Note: Additional options could be added here for more complex maps

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error visualizing data: {str(e)}")


def clip_raster_by_bbox(
    input_raster: str,
    output_raster: str,
    bbox: List[float],
    bands: Optional[List[int]] = None,
    bbox_type: str = "geo",
    bbox_crs: Optional[str] = None,
) -> str:
    """
    Clip a raster dataset using a bounding box and optionally select specific bands.

    Args:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path where the clipped raster will be saved.
        bbox (tuple): Bounding box coordinates either as:
                     - Geographic coordinates (minx, miny, maxx, maxy) if bbox_type="geo"
                     - Pixel indices (min_row, min_col, max_row, max_col) if bbox_type="pixel"
        bands (list, optional): List of band indices to keep (1-based indexing).
                               If None, all bands will be kept.
        bbox_type (str, optional): Type of bounding box coordinates. Either "geo" for
                                  geographic coordinates or "pixel" for row/column indices.
                                  Default is "geo".
        bbox_crs (str or dict, optional): CRS of the bbox if different from the raster CRS.
                                         Can be provided as EPSG code (e.g., "EPSG:4326") or
                                         as a proj4 string. Only applies when bbox_type="geo".
                                         If None, assumes bbox is in the same CRS as the raster.

    Returns:
        str: Path to the clipped output raster.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If the bbox is invalid, bands are out of range, or bbox_type is invalid.
        RuntimeError: If the clipping operation fails.

    Examples:
        Clip using geographic coordinates in the same CRS as the raster
        >>> clip_raster_by_bbox('input.tif', 'clipped_geo.tif', (100, 200, 300, 400))
        'clipped_geo.tif'

        Clip using WGS84 coordinates when the raster is in a different CRS
        >>> clip_raster_by_bbox('input.tif', 'clipped_wgs84.tif', (-122.5, 37.7, -122.4, 37.8),
        ...                     bbox_crs="EPSG:4326")
        'clipped_wgs84.tif'

        Clip using row/column indices
        >>> clip_raster_by_bbox('input.tif', 'clipped_pixel.tif', (50, 100, 150, 200),
        ...                     bbox_type="pixel")
        'clipped_pixel.tif'

        Clip with band selection
        >>> clip_raster_by_bbox('input.tif', 'clipped_bands.tif', (100, 200, 300, 400),
        ...                     bands=[1, 3])
        'clipped_bands.tif'
    """
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    # Validate bbox_type
    if bbox_type not in ["geo", "pixel"]:
        raise ValueError("bbox_type must be either 'geo' or 'pixel'")

    # Validate bbox
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")

    # Open the source raster
    with rasterio.open(input_raster) as src:
        # Get the source CRS
        src_crs = src.crs

        # Handle different bbox types
        if bbox_type == "geo":
            minx, miny, maxx, maxy = bbox

            # Validate geographic bbox
            if minx >= maxx or miny >= maxy:
                raise ValueError(
                    "Invalid geographic bbox. Expected (minx, miny, maxx, maxy) where minx < maxx and miny < maxy"
                )

            # If bbox_crs is provided and different from the source CRS, transform the bbox
            if bbox_crs is not None and bbox_crs != src_crs:
                try:
                    # Transform bbox coordinates from bbox_crs to src_crs
                    minx, miny, maxx, maxy = transform_bounds(
                        bbox_crs, src_crs, minx, miny, maxx, maxy
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to transform bbox from {bbox_crs} to {src_crs}: {str(e)}"
                    )

            # Calculate the pixel window from geographic coordinates
            window = src.window(minx, miny, maxx, maxy)

            # Use the same bounds for the output transform
            output_bounds = (minx, miny, maxx, maxy)

        else:  # bbox_type == "pixel"
            min_row, min_col, max_row, max_col = bbox

            # Validate pixel bbox
            if min_row >= max_row or min_col >= max_col:
                raise ValueError(
                    "Invalid pixel bbox. Expected (min_row, min_col, max_row, max_col) where min_row < max_row and min_col < max_col"
                )

            if (
                min_row < 0
                or min_col < 0
                or max_row > src.height
                or max_col > src.width
            ):
                raise ValueError(
                    f"Pixel indices out of bounds. Raster dimensions are {src.height} rows x {src.width} columns"
                )

            # Create a window from pixel coordinates
            window = Window(min_col, min_row, max_col - min_col, max_row - min_row)

            # Calculate the geographic bounds for this window
            window_transform = src.window_transform(window)
            output_bounds = rasterio.transform.array_bounds(
                window.height, window.width, window_transform
            )
            # Reorder to (minx, miny, maxx, maxy)
            output_bounds = (
                output_bounds[0],
                output_bounds[1],
                output_bounds[2],
                output_bounds[3],
            )

        # Get window dimensions
        window_width = int(window.width)
        window_height = int(window.height)

        # Check if the window is valid
        if window_width <= 0 or window_height <= 0:
            raise ValueError("Bounding box results in an empty window")

        # Handle band selection
        if bands is None:
            # Use all bands
            bands_to_read = list(range(1, src.count + 1))
        else:
            # Validate band indices
            if not all(1 <= b <= src.count for b in bands):
                raise ValueError(f"Band indices must be between 1 and {src.count}")
            bands_to_read = bands

        # Calculate new transform for the clipped raster
        new_transform = from_bounds(
            output_bounds[0],
            output_bounds[1],
            output_bounds[2],
            output_bounds[3],
            window_width,
            window_height,
        )

        # Create a metadata dictionary for the output
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": window_height,
                "width": window_width,
                "transform": new_transform,
                "count": len(bands_to_read),
            }
        )

        # Read the data for the selected bands
        data = []
        for band_idx in bands_to_read:
            band_data = src.read(band_idx, window=window)
            data.append(band_data)

        # Stack the bands into a single array
        if len(data) > 1:
            clipped_data = np.stack(data)
        else:
            clipped_data = data[0][np.newaxis, :, :]

        # Write the output raster
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(clipped_data)

    return output_raster


def raster_to_vector(
    raster_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0,
    min_area: float = 10,
    simplify_tolerance: Optional[float] = None,
    class_values: Optional[List[int]] = None,
    attribute_name: str = "class",
    unique_attribute_value: bool = False,
    output_format: str = "geojson",
    plot_result: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert a raster label mask to vector polygons.

    Args:
        raster_path (str): Path to the input raster file (e.g., GeoTIFF).
        output_path (str): Path to save the output vector file. If None, returns GeoDataFrame without saving.
        threshold (int/float): Pixel values greater than this threshold will be vectorized.
        min_area (float): Minimum polygon area in square map units to keep.
        simplify_tolerance (float): Tolerance for geometry simplification. None for no simplification.
        class_values (list): Specific pixel values to vectorize. If None, all values > threshold are vectorized.
        attribute_name (str): Name of the attribute field for the class values.
        unique_attribute_value (bool): Whether to generate unique values for each shape within a class.
        output_format (str): Format for output file - 'geojson', 'shapefile', 'gpkg'.
        plot_result (bool): Whether to plot the resulting polygons overlaid on the raster.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the vectorized polygons.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1)

        # Get metadata
        transform = src.transform
        crs = src.crs

        # Create mask based on threshold and class values
        if class_values is not None:
            # Create a mask for each specified class value
            masks = {val: (data == val) for val in class_values}
        else:
            # Create a mask for values above threshold
            masks = {1: (data > threshold)}
            class_values = [1]  # Default class

        # Initialize list to store features
        all_features = []

        # Process each class value
        for class_val in class_values:
            mask = masks[class_val]
            shape_count = 1
            # Vectorize the mask
            for geom, value in features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            ):
                # Convert to shapely geometry
                geom = shape(geom)

                # Skip small polygons
                if geom.area < min_area:
                    continue

                # Simplify geometry if requested
                if simplify_tolerance is not None:
                    geom = geom.simplify(simplify_tolerance)

                # Add to features list with class value
                if unique_attribute_value:
                    all_features.append(
                        {"geometry": geom, attribute_name: class_val * shape_count}
                    )
                else:
                    all_features.append({"geometry": geom, attribute_name: class_val})

                shape_count += 1

        # Create GeoDataFrame
        if all_features:
            gdf = gpd.GeoDataFrame(all_features, crs=crs)
        else:
            print("Warning: No features were extracted from the raster.")
            # Return empty GeoDataFrame with correct CRS
            gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)

        # Save to file if requested
        if output_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save to file based on format
            if output_format.lower() == "geojson":
                gdf.to_file(output_path, driver="GeoJSON")
            elif output_format.lower() == "shapefile":
                gdf.to_file(output_path)
            elif output_format.lower() == "gpkg":
                gdf.to_file(output_path, driver="GPKG")
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            print(f"Vectorized data saved to {output_path}")

        # Plot result if requested
        if plot_result:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Plot raster
            raster_img = src.read()
            if raster_img.shape[0] == 1:
                plt.imshow(raster_img[0], cmap="viridis", alpha=0.7)
            else:
                # Use first 3 bands for RGB display
                rgb = raster_img[:3].transpose(1, 2, 0)
                # Normalize for display
                rgb = np.clip(rgb / rgb.max(), 0, 1)
                plt.imshow(rgb)

            # Plot vector boundaries
            if not gdf.empty:
                gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

            plt.title("Raster with Vectorized Boundaries")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return gdf


def raster_to_vector_batch(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.tif",
    threshold: float = 0,
    min_area: float = 10,
    simplify_tolerance: Optional[float] = None,
    class_values: Optional[List[int]] = None,
    attribute_name: str = "class",
    output_format: str = "geojson",
    merge_output: bool = False,
    merge_filename: str = "merged_vectors",
) -> Optional[gpd.GeoDataFrame]:
    """
    Batch convert multiple raster files to vector polygons.

    Args:
        input_dir (str): Directory containing input raster files.
        output_dir (str): Directory to save output vector files.
        pattern (str): Pattern to match raster files (e.g., '*.tif').
        threshold (int/float): Pixel values greater than this threshold will be vectorized.
        min_area (float): Minimum polygon area in square map units to keep.
        simplify_tolerance (float): Tolerance for geometry simplification. None for no simplification.
        class_values (list): Specific pixel values to vectorize. If None, all values > threshold are vectorized.
        attribute_name (str): Name of the attribute field for the class values.
        output_format (str): Format for output files - 'geojson', 'shapefile', 'gpkg'.
        merge_output (bool): Whether to merge all output vectors into a single file.
        merge_filename (str): Filename for the merged output (without extension).

    Returns:
        geopandas.GeoDataFrame or None: If merge_output is True, returns the merged GeoDataFrame.
    """
    import glob

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of raster files
    raster_files = glob.glob(os.path.join(input_dir, pattern))

    if not raster_files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return None

    print(f"Found {len(raster_files)} raster files to process")

    # Process each raster file
    gdfs = []
    for raster_file in tqdm(raster_files, desc="Processing rasters"):
        # Get output filename
        base_name = os.path.splitext(os.path.basename(raster_file))[0]
        if output_format.lower() == "geojson":
            out_file = os.path.join(output_dir, f"{base_name}.geojson")
        elif output_format.lower() == "shapefile":
            out_file = os.path.join(output_dir, f"{base_name}.shp")
        elif output_format.lower() == "gpkg":
            out_file = os.path.join(output_dir, f"{base_name}.gpkg")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Convert raster to vector
        if merge_output:
            # Don't save individual files if merging
            gdf = raster_to_vector(
                raster_file,
                output_path=None,
                threshold=threshold,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
                class_values=class_values,
                attribute_name=attribute_name,
            )

            # Add filename as attribute
            if not gdf.empty:
                gdf["source_file"] = base_name
                gdfs.append(gdf)
        else:
            # Save individual files
            raster_to_vector(
                raster_file,
                output_path=out_file,
                threshold=threshold,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
                class_values=class_values,
                attribute_name=attribute_name,
                output_format=output_format,
            )

    # Merge output if requested
    if merge_output and gdfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        # Set CRS to the CRS of the first GeoDataFrame
        if merged_gdf.crs is None and gdfs:
            merged_gdf.crs = gdfs[0].crs

        # Save merged output
        if output_format.lower() == "geojson":
            merged_file = os.path.join(output_dir, f"{merge_filename}.geojson")
            merged_gdf.to_file(merged_file, driver="GeoJSON")
        elif output_format.lower() == "shapefile":
            merged_file = os.path.join(output_dir, f"{merge_filename}.shp")
            merged_gdf.to_file(merged_file)
        elif output_format.lower() == "gpkg":
            merged_file = os.path.join(output_dir, f"{merge_filename}.gpkg")
            merged_gdf.to_file(merged_file, driver="GPKG")

        print(f"Merged vector data saved to {merged_file}")
        return merged_gdf

    return None


def vector_to_raster(
    vector_path: Union[str, gpd.GeoDataFrame],
    output_path: Optional[str] = None,
    reference_raster: Optional[str] = None,
    attribute_field: Optional[str] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    transform: Optional[Any] = None,
    pixel_size: Optional[float] = None,
    bounds: Optional[List[float]] = None,
    crs: Optional[str] = None,
    all_touched: bool = False,
    fill_value: Union[int, float] = 0,
    dtype: Any = np.uint8,
    nodata: Optional[Union[int, float]] = None,
    plot_result: bool = False,
) -> np.ndarray:
    """
    Convert vector data to a raster.

    Args:
        vector_path (str or GeoDataFrame): Path to the input vector file or a GeoDataFrame.
        output_path (str): Path to save the output raster file. If None, returns the array without saving.
        reference_raster (str): Path to a reference raster for dimensions, transform and CRS.
        attribute_field (str): Field name in the vector data to use for pixel values.
            If None, all vector features will be burned with value 1.
        output_shape (tuple): Shape of the output raster as (height, width).
            Required if reference_raster is not provided.
        transform (affine.Affine): Affine transformation matrix.
            Required if reference_raster is not provided.
        pixel_size (float or tuple): Pixel size (resolution) as single value or (x_res, y_res).
            Used to calculate transform if transform is not provided.
        bounds (tuple): Bounds of the output raster as (left, bottom, right, top).
            Used to calculate transform if transform is not provided.
        crs (str or CRS): Coordinate reference system of the output raster.
            Required if reference_raster is not provided.
        all_touched (bool): If True, all pixels touched by geometries will be burned in.
            If False, only pixels whose center is within the geometry will be burned in.
        fill_value (int): Value to fill the raster with before burning in features.
        dtype (numpy.dtype): Data type of the output raster.
        nodata (int): No data value for the output raster.
        plot_result (bool): Whether to plot the resulting raster.

    Returns:
        numpy.ndarray: The rasterized data array if output_path is None, else None.
    """
    # Load vector data
    if isinstance(vector_path, gpd.GeoDataFrame):
        gdf = vector_path
    else:
        gdf = gpd.read_file(vector_path)

    # Check if vector data is empty
    if gdf.empty:
        warnings.warn("The input vector data is empty. Creating an empty raster.")

    # Get CRS from vector data if not provided
    if crs is None and reference_raster is None:
        crs = gdf.crs

    # Get transform and output shape from reference raster if provided
    if reference_raster is not None:
        with rasterio.open(reference_raster) as src:
            transform = src.transform
            output_shape = src.shape
            crs = src.crs
            if nodata is None:
                nodata = src.nodata
    else:
        # Check if we have all required parameters
        if transform is None:
            if pixel_size is None or bounds is None:
                raise ValueError(
                    "Either reference_raster, transform, or both pixel_size and bounds must be provided."
                )

            # Calculate transform from pixel size and bounds
            if isinstance(pixel_size, (int, float)):
                x_res = y_res = float(pixel_size)
            else:
                x_res, y_res = pixel_size
                y_res = abs(y_res) * -1  # Convert to negative for north-up raster

            left, bottom, right, top = bounds
            transform = rasterio.transform.from_bounds(
                left,
                bottom,
                right,
                top,
                int((right - left) / x_res),
                int((top - bottom) / abs(y_res)),
            )

        if output_shape is None:
            # Calculate output shape from bounds and pixel size
            if bounds is None or pixel_size is None:
                raise ValueError(
                    "output_shape must be provided if reference_raster is not provided and "
                    "cannot be calculated from bounds and pixel_size."
                )

            if isinstance(pixel_size, (int, float)):
                x_res = y_res = float(pixel_size)
            else:
                x_res, y_res = pixel_size

            left, bottom, right, top = bounds
            width = int((right - left) / x_res)
            height = int((top - bottom) / abs(y_res))
            output_shape = (height, width)

    # Ensure CRS is set
    if crs is None:
        raise ValueError(
            "CRS must be provided either directly, from reference_raster, or from input vector data."
        )

    # Reproject vector data if its CRS doesn't match the output CRS
    if gdf.crs != crs:
        print(f"Reprojecting vector data from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # Create empty raster filled with fill_value
    raster_data = np.full(output_shape, fill_value, dtype=dtype)

    # Burn vector features into raster
    if not gdf.empty:
        # Prepare shapes for burning
        if attribute_field is not None and attribute_field in gdf.columns:
            # Use attribute field for values
            shapes = [
                (geom, value) for geom, value in zip(gdf.geometry, gdf[attribute_field])
            ]
        else:
            # Burn with value 1
            shapes = [(geom, 1) for geom in gdf.geometry]

        # Burn shapes into raster
        burned = features.rasterize(
            shapes=shapes,
            out_shape=output_shape,
            transform=transform,
            fill=fill_value,
            all_touched=all_touched,
            dtype=dtype,
        )

        # Update raster data
        raster_data = burned

    # Save raster if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Define metadata
        metadata = {
            "driver": "GTiff",
            "height": output_shape[0],
            "width": output_shape[1],
            "count": 1,
            "dtype": raster_data.dtype,
            "crs": crs,
            "transform": transform,
        }

        # Add nodata value if provided
        if nodata is not None:
            metadata["nodata"] = nodata

        # Write raster
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(raster_data, 1)

        print(f"Rasterized data saved to {output_path}")

    # Plot result if requested
    if plot_result:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot raster
        im = ax.imshow(raster_data, cmap="viridis")
        plt.colorbar(im, ax=ax, label=attribute_field if attribute_field else "Value")

        # Plot vector boundaries for reference
        if output_path is not None:
            # Get the extent of the raster
            with rasterio.open(output_path) as src:
                bounds = src.bounds
                raster_bbox = box(*bounds)
        else:
            # Calculate extent from transform and shape
            height, width = output_shape
            left, top = transform * (0, 0)
            right, bottom = transform * (width, height)
            raster_bbox = box(left, bottom, right, top)

        # Clip vector to raster extent for clarity in plot
        if not gdf.empty:
            gdf_clipped = gpd.clip(gdf, raster_bbox)
            if not gdf_clipped.empty:
                gdf_clipped.boundary.plot(ax=ax, color="red", linewidth=1)

        plt.title("Rasterized Vector Data")
        plt.tight_layout()
        plt.show()

    return raster_data


def batch_vector_to_raster(
    vector_path,
    output_dir,
    attribute_field=None,
    reference_rasters=None,
    bounds_list=None,
    output_filename_pattern="{vector_name}_{index}",
    pixel_size=1.0,
    all_touched=False,
    fill_value=0,
    dtype=np.uint8,
    nodata=None,
) -> List[str]:
    """
    Batch convert vector data to multiple rasters based on different extents or reference rasters.

    Args:
        vector_path (str or GeoDataFrame): Path to the input vector file or a GeoDataFrame.
        output_dir (str): Directory to save output raster files.
        attribute_field (str): Field name in the vector data to use for pixel values.
        reference_rasters (list): List of paths to reference rasters for dimensions, transform and CRS.
        bounds_list (list): List of bounds tuples (left, bottom, right, top) to use if reference_rasters not provided.
        output_filename_pattern (str): Pattern for output filenames.
            Can include {vector_name} and {index} placeholders.
        pixel_size (float or tuple): Pixel size to use if reference_rasters not provided.
        all_touched (bool): If True, all pixels touched by geometries will be burned in.
        fill_value (int): Value to fill the raster with before burning in features.
        dtype (numpy.dtype): Data type of the output raster.
        nodata (int): No data value for the output raster.

    Returns:
        List[str]: List of paths to the created raster files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load vector data if it's a path
    if isinstance(vector_path, str):
        gdf = gpd.read_file(vector_path)
        vector_name = os.path.splitext(os.path.basename(vector_path))[0]
    else:
        gdf = vector_path
        vector_name = "vector"

    # Check input parameters
    if reference_rasters is None and bounds_list is None:
        raise ValueError("Either reference_rasters or bounds_list must be provided.")

    # Use reference_rasters if provided, otherwise use bounds_list
    if reference_rasters is not None:
        sources = reference_rasters
        is_raster_reference = True
    else:
        sources = bounds_list
        is_raster_reference = False

    # Create output filenames
    output_files = []

    # Process each source (reference raster or bounds)
    for i, source in enumerate(tqdm(sources, desc="Processing")):
        # Generate output filename
        output_filename = output_filename_pattern.format(
            vector_name=vector_name, index=i
        )
        if not output_filename.endswith(".tif"):
            output_filename += ".tif"
        output_path = os.path.join(output_dir, output_filename)

        if is_raster_reference:
            # Use reference raster
            vector_to_raster(
                vector_path=gdf,
                output_path=output_path,
                reference_raster=source,
                attribute_field=attribute_field,
                all_touched=all_touched,
                fill_value=fill_value,
                dtype=dtype,
                nodata=nodata,
            )
        else:
            # Use bounds
            vector_to_raster(
                vector_path=gdf,
                output_path=output_path,
                bounds=source,
                pixel_size=pixel_size,
                attribute_field=attribute_field,
                all_touched=all_touched,
                fill_value=fill_value,
                dtype=dtype,
                nodata=nodata,
            )

        output_files.append(output_path)

    return output_files


def get_default_augmentation_transforms(
    tile_size: int = 256,
    include_normalize: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    """
    Get default data augmentation transforms for geospatial imagery using albumentations.

    This function returns a composition of augmentation transforms commonly used
    for remote sensing and geospatial data. The transforms include geometric
    transformations (flips, rotations) and photometric adjustments (brightness,
    contrast, saturation).

    Args:
        tile_size (int): Target size for tiles. Defaults to 256.
        include_normalize (bool): Whether to include normalization transform.
            Defaults to False. Set to True if using for training with pretrained models.
        mean (tuple): Mean values for normalization (RGB). Defaults to ImageNet values.
        std (tuple): Standard deviation for normalization (RGB). Defaults to ImageNet values.

    Returns:
        albumentations.Compose: A composition of augmentation transforms.

    Example:
        >>> import albumentations as A
        >>> # Get default transforms
        >>> transform = get_default_augmentation_transforms()
        >>> # Apply to image and mask
        >>> augmented = transform(image=image, mask=mask)
        >>> aug_image = augmented['image']
        >>> aug_mask = augmented['mask']
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install it with: pip install albumentations"
        )

    transforms_list = [
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5,
        ),
        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]

    # Add normalization if requested
    if include_normalize:
        transforms_list.append(A.Normalize(mean=mean, std=std))

    return A.Compose(transforms_list)


def export_geotiff_tiles(
    in_raster,
    out_folder,
    in_class_data=None,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    quiet=False,
    all_touched=True,
    create_overview=False,
    skip_empty_tiles=False,
    metadata_format="PASCAL_VOC",
    apply_augmentation=False,
    augmentation_count=3,
    augmentation_transforms=None,
):
    """
    Export georeferenced GeoTIFF tiles and labels from raster and classification data.

    Args:
        in_raster (str): Path to input raster image
        out_folder (str): Path to output folder
        in_class_data (str, optional): Path to classification data - can be vector file or raster.
            If None, only image tiles will be exported without labels. Defaults to None.
        tile_size (int): Size of tiles in pixels (square)
        stride (int): Step size between tiles
        class_value_field (str): Field containing class values (for vector data)
        buffer_radius (float): Buffer to add around features (in units of the CRS)
        max_tiles (int): Maximum number of tiles to process (None for all)
        quiet (bool): If True, suppress non-essential output
        all_touched (bool): Whether to use all_touched=True in rasterization (for vector data)
        create_overview (bool): Whether to create an overview image of all tiles
        skip_empty_tiles (bool): If True, skip tiles with no features
        metadata_format (str): Output metadata format (PASCAL_VOC, COCO, YOLO). Default: PASCAL_VOC
        apply_augmentation (bool): If True, generate augmented versions of each tile.
            This will create multiple variants of each tile using data augmentation techniques.
            Defaults to False.
        augmentation_count (int): Number of augmented versions to generate per tile
            (only used if apply_augmentation=True). Defaults to 3.
        augmentation_transforms (albumentations.Compose, optional): Custom augmentation transforms.
            If None and apply_augmentation=True, uses default transforms from
            get_default_augmentation_transforms(). Should be an albumentations.Compose object.
            Defaults to None.

    Returns:
        None: Tiles and labels are saved to out_folder.

    Example:
        >>> # Export tiles without augmentation
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif')
        >>>
        >>> # Export tiles with default augmentation (3 augmented versions per tile)
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif',
        ...                      apply_augmentation=True)
        >>>
        >>> # Export with custom augmentation
        >>> import albumentations as A
        >>> custom_transform = A.Compose([
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif',
        ...                      apply_augmentation=True,
        ...                      augmentation_count=5,
        ...                      augmentation_transforms=custom_transform)
    """

    import logging

    logging.getLogger("rasterio").setLevel(logging.ERROR)

    # Initialize augmentation transforms if needed
    if apply_augmentation:
        if augmentation_transforms is None:
            augmentation_transforms = get_default_augmentation_transforms(
                tile_size=tile_size
            )
        if not quiet:
            print(
                f"Data augmentation enabled: generating {augmentation_count} augmented versions per tile"
            )

    # Create output directories
    os.makedirs(out_folder, exist_ok=True)
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Only create label and annotation directories if class data is provided
    if in_class_data is not None:
        label_dir = os.path.join(out_folder, "labels")
        os.makedirs(label_dir, exist_ok=True)

        # Create annotation directory based on metadata format
        if metadata_format in ["PASCAL_VOC", "COCO"]:
            ann_dir = os.path.join(out_folder, "annotations")
            os.makedirs(ann_dir, exist_ok=True)

        # Initialize COCO annotations dictionary
        if metadata_format == "COCO":
            coco_annotations = {"images": [], "annotations": [], "categories": []}
            ann_id = 0

    # Determine if class data is raster or vector (only if class data provided)
    is_class_data_raster = False
    if in_class_data is not None:
        if isinstance(in_class_data, str):
            file_ext = Path(in_class_data).suffix.lower()
            # Common raster extensions
            if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
                try:
                    with rasterio.open(in_class_data) as src:
                        is_class_data_raster = True
                        if not quiet:
                            print(f"Detected in_class_data as raster: {in_class_data}")
                            print(f"Raster CRS: {src.crs}")
                            print(f"Raster dimensions: {src.width} x {src.height}")
                except Exception:
                    is_class_data_raster = False
                    if not quiet:
                        print(
                            f"Unable to open {in_class_data} as raster, trying as vector"
                        )

    # Open the input raster
    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"\nRaster info for {in_raster}:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Resolution: {src.res}")
            print(f"  Bands: {src.count}")
            print(f"  Bounds: {src.bounds}")

        # Calculate number of tiles
        num_tiles_x = math.ceil((src.width - tile_size) / stride) + 1
        num_tiles_y = math.ceil((src.height - tile_size) / stride) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if max_tiles is None:
            max_tiles = total_tiles

        # Process classification data (only if class data provided)
        class_to_id = {}

        if in_class_data is not None and is_class_data_raster:
            # Load raster class data
            with rasterio.open(in_class_data) as class_src:
                # Check if raster CRS matches
                if class_src.crs != src.crs:
                    warnings.warn(
                        f"CRS mismatch: Class raster ({class_src.crs}) doesn't match input raster ({src.crs}). "
                        f"Results may be misaligned."
                    )

                # Get unique values from raster
                # Sample to avoid loading huge rasters
                sample_data = class_src.read(
                    1,
                    out_shape=(
                        1,
                        min(class_src.height, 1000),
                        min(class_src.width, 1000),
                    ),
                )

                unique_classes = np.unique(sample_data)
                unique_classes = unique_classes[
                    unique_classes > 0
                ]  # Remove 0 as it's typically background

                if not quiet:
                    print(
                        f"Found {len(unique_classes)} unique classes in raster: {unique_classes}"
                    )

                # Create class mapping
                class_to_id = {int(cls): i + 1 for i, cls in enumerate(unique_classes)}

                # Populate COCO categories
                if metadata_format == "COCO":
                    for cls_val in unique_classes:
                        coco_annotations["categories"].append(
                            {
                                "id": class_to_id[int(cls_val)],
                                "name": str(int(cls_val)),
                                "supercategory": "object",
                            }
                        )
        elif in_class_data is not None:
            # Load vector class data
            try:
                gdf = gpd.read_file(in_class_data)
                if not quiet:
                    print(f"Loaded {len(gdf)} features from {in_class_data}")
                    print(f"Vector CRS: {gdf.crs}")

                # Always reproject to match raster CRS
                if gdf.crs != src.crs:
                    if not quiet:
                        print(f"Reprojecting features from {gdf.crs} to {src.crs}")
                    gdf = gdf.to_crs(src.crs)

                # Apply buffer if specified
                if buffer_radius > 0:
                    gdf["geometry"] = gdf.buffer(buffer_radius)
                    if not quiet:
                        print(f"Applied buffer of {buffer_radius} units")

                # Check if class_value_field exists
                if class_value_field in gdf.columns:
                    unique_classes = gdf[class_value_field].unique()
                    if not quiet:
                        print(
                            f"Found {len(unique_classes)} unique classes: {unique_classes}"
                        )
                    # Create class mapping
                    class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}

                    # Populate COCO categories
                    if metadata_format == "COCO":
                        for cls_val in unique_classes:
                            coco_annotations["categories"].append(
                                {
                                    "id": class_to_id[cls_val],
                                    "name": str(cls_val),
                                    "supercategory": "object",
                                }
                            )
                else:
                    if not quiet:
                        print(
                            f"WARNING: '{class_value_field}' not found in vector data. Using default class ID 1."
                        )
                    class_to_id = {1: 1}  # Default mapping

                    # Populate COCO categories with default
                    if metadata_format == "COCO":
                        coco_annotations["categories"].append(
                            {
                                "id": 1,
                                "name": "object",
                                "supercategory": "object",
                            }
                        )
            except Exception as e:
                raise ValueError(f"Error processing vector data: {e}")

        # Create progress bar
        pbar = tqdm(
            total=min(total_tiles, max_tiles),
            desc="Generating tiles",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Track statistics for summary
        stats = {
            "total_tiles": 0,
            "tiles_with_features": 0,
            "feature_pixels": 0,
            "errors": 0,
            "tile_coordinates": [],  # For overview image
        }

        # Process tiles
        tile_index = 0
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                if tile_index >= max_tiles:
                    break

                # Calculate window coordinates
                window_x = x * stride
                window_y = y * stride

                # Adjust for edge cases
                if window_x + tile_size > src.width:
                    window_x = src.width - tile_size
                if window_y + tile_size > src.height:
                    window_y = src.height - tile_size

                # Define window
                window = Window(window_x, window_y, tile_size, tile_size)

                # Get window transform and bounds
                window_transform = src.window_transform(window)

                # Calculate window bounds
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size * window_transform[0]  # Add width
                miny = maxy + tile_size * window_transform[4]  # Add height

                window_bounds = box(minx, miny, maxx, maxy)

                # Store tile coordinates for overview
                if create_overview:
                    stats["tile_coordinates"].append(
                        {
                            "index": tile_index,
                            "x": window_x,
                            "y": window_y,
                            "bounds": [minx, miny, maxx, maxy],
                            "has_features": False,
                        }
                    )

                # Create label mask
                label_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                has_features = False

                # Process classification data to create labels (only if class data provided)
                if in_class_data is not None and is_class_data_raster:
                    # For raster class data
                    with rasterio.open(in_class_data) as class_src:
                        # Calculate window in class raster
                        src_bounds = src.bounds
                        class_bounds = class_src.bounds

                        # Check if windows overlap
                        if (
                            src_bounds.left > class_bounds.right
                            or src_bounds.right < class_bounds.left
                            or src_bounds.bottom > class_bounds.top
                            or src_bounds.top < class_bounds.bottom
                        ):
                            warnings.warn(
                                "Class raster and input raster do not overlap."
                            )
                        else:
                            # Get corresponding window in class raster
                            window_class = rasterio.windows.from_bounds(
                                minx, miny, maxx, maxy, class_src.transform
                            )

                            # Read label data
                            try:
                                label_data = class_src.read(
                                    1,
                                    window=window_class,
                                    boundless=True,
                                    out_shape=(tile_size, tile_size),
                                )

                                # Remap class values if needed
                                if class_to_id:
                                    remapped_data = np.zeros_like(label_data)
                                    for orig_val, new_val in class_to_id.items():
                                        remapped_data[label_data == orig_val] = new_val
                                    label_mask = remapped_data
                                else:
                                    label_mask = label_data

                                # Check if we have any features
                                if np.any(label_mask > 0):
                                    has_features = True
                                    stats["feature_pixels"] += np.count_nonzero(
                                        label_mask
                                    )
                            except Exception as e:
                                pbar.write(f"Error reading class raster window: {e}")
                                stats["errors"] += 1
                elif in_class_data is not None:
                    # For vector class data
                    # Find features that intersect with window
                    window_features = gdf[gdf.intersects(window_bounds)]

                    if len(window_features) > 0:
                        for idx, feature in window_features.iterrows():
                            # Get class value
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                class_id = class_to_id.get(class_val, 1)
                            else:
                                class_id = 1

                            # Get geometry in window coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                try:
                                    # Rasterize feature
                                    feature_mask = features.rasterize(
                                        [(geom, class_id)],
                                        out_shape=(tile_size, tile_size),
                                        transform=window_transform,
                                        fill=0,
                                        all_touched=all_touched,
                                    )

                                    # Add to label mask
                                    label_mask = np.maximum(label_mask, feature_mask)

                                    # Check if the feature was actually rasterized
                                    if np.any(feature_mask):
                                        has_features = True
                                        if create_overview and tile_index < len(
                                            stats["tile_coordinates"]
                                        ):
                                            stats["tile_coordinates"][tile_index][
                                                "has_features"
                                            ] = True
                                except Exception as e:
                                    pbar.write(f"Error rasterizing feature {idx}: {e}")
                                    stats["errors"] += 1

                # Skip tile if no features and skip_empty_tiles is True (only when class data provided)
                if in_class_data is not None and skip_empty_tiles and not has_features:
                    pbar.update(1)
                    tile_index += 1
                    continue

                # Read image data
                image_data = src.read(window=window)

                # Helper function to save a single tile (original or augmented)
                def save_tile(
                    img_data,
                    lbl_mask,
                    tile_id,
                    img_profile,
                    window_trans,
                    is_augmented=False,
                ):
                    """Save a single image and label tile."""
                    # Export image as GeoTIFF
                    image_path = os.path.join(image_dir, f"tile_{tile_id:06d}.tif")

                    # Update profile
                    img_profile_copy = img_profile.copy()
                    img_profile_copy.update(
                        {
                            "height": tile_size,
                            "width": tile_size,
                            "count": img_data.shape[0],
                            "transform": window_trans,
                        }
                    )

                    # Save image as GeoTIFF
                    try:
                        with rasterio.open(image_path, "w", **img_profile_copy) as dst:
                            dst.write(img_data)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        pbar.write(f"ERROR saving image GeoTIFF: {e}")
                        stats["errors"] += 1
                        return

                    # Export label as GeoTIFF (only if class data provided)
                    if in_class_data is not None:
                        # Create profile for label GeoTIFF
                        label_profile = {
                            "driver": "GTiff",
                            "height": tile_size,
                            "width": tile_size,
                            "count": 1,
                            "dtype": "uint8",
                            "crs": src.crs,
                            "transform": window_trans,
                        }

                        label_path = os.path.join(label_dir, f"tile_{tile_id:06d}.tif")
                        try:
                            with rasterio.open(label_path, "w", **label_profile) as dst:
                                dst.write(lbl_mask.astype(np.uint8), 1)

                            if not is_augmented and np.any(lbl_mask > 0):
                                stats["tiles_with_features"] += 1
                                stats["feature_pixels"] += np.count_nonzero(lbl_mask)
                        except Exception as e:
                            pbar.write(f"ERROR saving label GeoTIFF: {e}")
                            stats["errors"] += 1

                # Save original tile
                save_tile(
                    image_data,
                    label_mask,
                    tile_index,
                    src.profile,
                    window_transform,
                    is_augmented=False,
                )

                # Generate and save augmented tiles if enabled
                if apply_augmentation:
                    for aug_idx in range(augmentation_count):
                        # Prepare image for augmentation (convert from CHW to HWC)
                        img_for_aug = np.transpose(image_data, (1, 2, 0))

                        # Ensure uint8 data type for albumentations
                        # Albumentations expects uint8 for most transforms
                        if not np.issubdtype(img_for_aug.dtype, np.uint8):
                            # If image is float, scale to 0-255 and convert to uint8
                            if np.issubdtype(img_for_aug.dtype, np.floating):
                                img_for_aug = (
                                    (img_for_aug * 255).clip(0, 255).astype(np.uint8)
                                )
                            else:
                                img_for_aug = img_for_aug.astype(np.uint8)

                        # Apply augmentation
                        try:
                            if in_class_data is not None:
                                # Augment both image and mask
                                augmented = augmentation_transforms(
                                    image=img_for_aug, mask=label_mask
                                )
                                aug_image = augmented["image"]
                                aug_mask = augmented["mask"]
                            else:
                                # Augment only image
                                augmented = augmentation_transforms(image=img_for_aug)
                                aug_image = augmented["image"]
                                aug_mask = label_mask

                            # Convert back from HWC to CHW
                            aug_image = np.transpose(aug_image, (2, 0, 1))

                            # Ensure correct dtype for saving
                            aug_image = aug_image.astype(image_data.dtype)

                            # Generate unique tile ID for augmented version
                            # Use a collision-free numbering scheme: (tile_index * (augmentation_count + 1)) + aug_idx + 1
                            aug_tile_id = (
                                (tile_index * (augmentation_count + 1)) + aug_idx + 1
                            )

                            # Save augmented tile
                            save_tile(
                                aug_image,
                                aug_mask,
                                aug_tile_id,
                                src.profile,
                                window_transform,
                                is_augmented=True,
                            )

                        except Exception as e:
                            pbar.write(
                                f"ERROR applying augmentation {aug_idx} to tile {tile_index}: {e}"
                            )
                            stats["errors"] += 1

                # Create annotations for object detection if using vector class data
                if (
                    in_class_data is not None
                    and not is_class_data_raster
                    and "gdf" in locals()
                    and len(window_features) > 0
                ):
                    if metadata_format == "PASCAL_VOC":
                        # Create XML annotation
                        root = ET.Element("annotation")
                        ET.SubElement(root, "folder").text = "images"
                        ET.SubElement(root, "filename").text = (
                            f"tile_{tile_index:06d}.tif"
                        )

                        size = ET.SubElement(root, "size")
                        ET.SubElement(size, "width").text = str(tile_size)
                        ET.SubElement(size, "height").text = str(tile_size)
                        ET.SubElement(size, "depth").text = str(image_data.shape[0])

                        # Add georeference information
                        geo = ET.SubElement(root, "georeference")
                        ET.SubElement(geo, "crs").text = str(src.crs)
                        ET.SubElement(geo, "transform").text = str(
                            window_transform
                        ).replace("\n", "")
                        ET.SubElement(geo, "bounds").text = (
                            f"{minx}, {miny}, {maxx}, {maxy}"
                        )

                        # Add objects
                        for idx, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                            else:
                                class_val = "object"

                            # Get geometry bounds in pixel coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, int(col_min)))
                                ymin = max(0, min(tile_size, int(row_min)))
                                xmax = max(0, min(tile_size, int(col_max)))
                                ymax = max(0, min(tile_size, int(row_max)))

                                # Only add if the box has non-zero area
                                if xmax > xmin and ymax > ymin:
                                    obj = ET.SubElement(root, "object")
                                    ET.SubElement(obj, "name").text = str(class_val)
                                    ET.SubElement(obj, "difficult").text = "0"

                                    bbox = ET.SubElement(obj, "bndbox")
                                    ET.SubElement(bbox, "xmin").text = str(xmin)
                                    ET.SubElement(bbox, "ymin").text = str(ymin)
                                    ET.SubElement(bbox, "xmax").text = str(xmax)
                                    ET.SubElement(bbox, "ymax").text = str(ymax)

                        # Save XML
                        tree = ET.ElementTree(root)
                        xml_path = os.path.join(ann_dir, f"tile_{tile_index:06d}.xml")
                        tree.write(xml_path)

                    elif metadata_format == "COCO":
                        # Add image info
                        image_id = tile_index
                        coco_annotations["images"].append(
                            {
                                "id": image_id,
                                "file_name": f"tile_{tile_index:06d}.tif",
                                "width": tile_size,
                                "height": tile_size,
                                "crs": str(src.crs),
                                "transform": str(window_transform),
                            }
                        )

                        # Add annotations for each feature
                        for _, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                category_id = class_to_id.get(class_val, 1)
                            else:
                                category_id = 1

                            # Get geometry bounds
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, int(col_min)))
                                ymin = max(0, min(tile_size, int(row_min)))
                                xmax = max(0, min(tile_size, int(col_max)))
                                ymax = max(0, min(tile_size, int(row_max)))

                                # Skip if box is too small
                                if xmax - xmin < 1 or ymax - ymin < 1:
                                    continue

                                width = xmax - xmin
                                height = ymax - ymin

                                # Add annotation
                                ann_id += 1
                                coco_annotations["annotations"].append(
                                    {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "category_id": category_id,
                                        "bbox": [xmin, ymin, width, height],
                                        "area": width * height,
                                        "iscrowd": 0,
                                    }
                                )

                    elif metadata_format == "YOLO":
                        # Create YOLO format annotations
                        yolo_annotations = []

                        for _, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                # YOLO uses 0-indexed class IDs
                                class_id = class_to_id.get(class_val, 1) - 1
                            else:
                                class_id = 0

                            # Get geometry bounds
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, col_min))
                                ymin = max(0, min(tile_size, row_min))
                                xmax = max(0, min(tile_size, col_max))
                                ymax = max(0, min(tile_size, row_max))

                                # Skip if box is too small
                                if xmax - xmin < 1 or ymax - ymin < 1:
                                    continue

                                # Calculate normalized coordinates (YOLO format)
                                x_center = ((xmin + xmax) / 2) / tile_size
                                y_center = ((ymin + ymax) / 2) / tile_size
                                width = (xmax - xmin) / tile_size
                                height = (ymax - ymin) / tile_size

                                # Add YOLO annotation line
                                yolo_annotations.append(
                                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                )

                        # Save YOLO annotations to text file
                        if yolo_annotations:
                            yolo_path = os.path.join(
                                label_dir, f"tile_{tile_index:06d}.txt"
                            )
                            with open(yolo_path, "w") as f:
                                f.write("\n".join(yolo_annotations))

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Generated: {stats['total_tiles']}, With features: {stats['tiles_with_features']}"
                )

                tile_index += 1
                if tile_index >= max_tiles:
                    break

            if tile_index >= max_tiles:
                break

        # Close progress bar
        pbar.close()

        # Save COCO annotations if applicable (only if class data provided)
        if in_class_data is not None and metadata_format == "COCO":
            try:
                with open(os.path.join(ann_dir, "instances.json"), "w") as f:
                    json.dump(coco_annotations, f, indent=2)
                if not quiet:
                    print(
                        f"Saved COCO annotations: {len(coco_annotations['images'])} images, "
                        f"{len(coco_annotations['annotations'])} annotations, "
                        f"{len(coco_annotations['categories'])} categories"
                    )
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving COCO annotations: {e}")
                stats["errors"] += 1

        # Save YOLO classes file if applicable (only if class data provided)
        if in_class_data is not None and metadata_format == "YOLO":
            try:
                # Create classes.txt with class names
                classes_path = os.path.join(out_folder, "classes.txt")
                # Sort by class ID to ensure correct order
                sorted_classes = sorted(class_to_id.items(), key=lambda x: x[1])
                with open(classes_path, "w") as f:
                    for class_val, _ in sorted_classes:
                        f.write(f"{class_val}\n")
                if not quiet:
                    print(f"Saved YOLO classes file with {len(class_to_id)} classes")
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving YOLO classes file: {e}")
                stats["errors"] += 1

        # Create overview image if requested
        if create_overview and stats["tile_coordinates"]:
            try:
                create_overview_image(
                    src,
                    stats["tile_coordinates"],
                    os.path.join(out_folder, "overview.png"),
                    tile_size,
                    stride,
                )
            except Exception as e:
                print(f"Failed to create overview image: {e}")

        # Report results
        if not quiet:
            print("\n------- Export Summary -------")
            print(f"Total tiles exported: {stats['total_tiles']}")
            if in_class_data is not None:
                print(
                    f"Tiles with features: {stats['tiles_with_features']} ({stats['tiles_with_features']/max(1, stats['total_tiles'])*100:.1f}%)"
                )
                if stats["tiles_with_features"] > 0:
                    print(
                        f"Average feature pixels per tile: {stats['feature_pixels']/stats['tiles_with_features']:.1f}"
                    )
            if stats["errors"] > 0:
                print(f"Errors encountered: {stats['errors']}")
            print(f"Output saved to: {out_folder}")

            # Verify georeference in a sample image and label
            if stats["total_tiles"] > 0:
                print("\n------- Georeference Verification -------")
                sample_image = os.path.join(image_dir, f"tile_0.tif")

                if os.path.exists(sample_image):
                    try:
                        with rasterio.open(sample_image) as img:
                            print(f"Image CRS: {img.crs}")
                            print(f"Image transform: {img.transform}")
                            print(
                                f"Image has georeference: {img.crs is not None and img.transform is not None}"
                            )
                            print(
                                f"Image dimensions: {img.width}x{img.height}, {img.count} bands, {img.dtypes[0]} type"
                            )
                    except Exception as e:
                        print(f"Error verifying image georeference: {e}")

                # Only verify label if class data was provided
                if in_class_data is not None:
                    sample_label = os.path.join(label_dir, f"tile_0.tif")
                    if os.path.exists(sample_label):
                        try:
                            with rasterio.open(sample_label) as lbl:
                                print(f"Label CRS: {lbl.crs}")
                                print(f"Label transform: {lbl.transform}")
                                print(
                                    f"Label has georeference: {lbl.crs is not None and lbl.transform is not None}"
                                )
                                print(
                                    f"Label dimensions: {lbl.width}x{lbl.height}, {lbl.count} bands, {lbl.dtypes[0]} type"
                                )
                        except Exception as e:
                            print(f"Error verifying label georeference: {e}")

        # Return statistics dictionary for further processing if needed
        return stats


def export_geotiff_tiles_batch(
    images_folder,
    masks_folder=None,
    masks_file=None,
    output_folder=None,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    quiet=False,
    all_touched=True,
    skip_empty_tiles=False,
    image_extensions=None,
    mask_extensions=None,
    match_by_name=False,
    metadata_format="PASCAL_VOC",
) -> Dict[str, Any]:
    """
    Export georeferenced GeoTIFF tiles from images and optionally masks.

    This function supports four modes:
    1. Images only (no masks) - when neither masks_file nor masks_folder is provided
    2. Single vector file covering all images (masks_file parameter)
    3. Multiple vector files, one per image (masks_folder parameter)
    4. Multiple raster mask files (masks_folder parameter)

    For mode 1 (images only), only image tiles will be exported without labels.

    For mode 2 (single vector file), specify masks_file path. The function will
    use spatial intersection to determine which features apply to each image.

    For mode 3/4 (multiple mask files), specify masks_folder path. Images and masks
    are paired either by matching filenames (match_by_name=True) or by sorted order
    (match_by_name=False).

    All image tiles are saved to a single 'images' folder and all mask tiles (if provided)
    to a single 'masks' folder within the output directory.

    Args:
        images_folder (str): Path to folder containing raster images
        masks_folder (str, optional): Path to folder containing classification masks/vectors.
            Use this for multiple mask files (one per image or raster masks). If not provided
            and masks_file is also not provided, only image tiles will be exported.
        masks_file (str, optional): Path to a single vector file covering all images.
            Use this for a single GeoJSON/Shapefile that covers multiple images. If not provided
            and masks_folder is also not provided, only image tiles will be exported.
        output_folder (str, optional): Path to output folder. If None, creates 'tiles'
            subfolder in images_folder.
        tile_size (int): Size of tiles in pixels (square)
        stride (int): Step size between tiles
        class_value_field (str): Field containing class values (for vector data)
        buffer_radius (float): Buffer to add around features (in units of the CRS)
        max_tiles (int): Maximum number of tiles to process per image (None for all)
        quiet (bool): If True, suppress non-essential output
        all_touched (bool): Whether to use all_touched=True in rasterization (for vector data)
        create_overview (bool): Whether to create an overview image of all tiles
        skip_empty_tiles (bool): If True, skip tiles with no features
        image_extensions (list): List of image file extensions to process (default: common raster formats)
        mask_extensions (list): List of mask file extensions to process (default: common raster/vector formats)
        match_by_name (bool): If True, match image and mask files by base filename.
            If False, match by sorted order (alphabetically). Only applies when masks_folder is used.
        metadata_format (str): Annotation format - "PASCAL_VOC" (XML), "COCO" (JSON), or "YOLO" (TXT).
            Default is "PASCAL_VOC".

    Returns:
        Dict[str, Any]: Dictionary containing batch processing statistics

    Raises:
        ValueError: If no images found, or if masks_folder and masks_file are both specified,
            or if counts don't match when using masks_folder with match_by_name=False.

    Examples:
        # Images only (no masks)
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     output_folder='output/tiles'
        ... )

        # Single vector file covering all images
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_file='data/buildings.geojson',
        ...     output_folder='output/tiles'
        ... )

        # Multiple vector files, matched by filename
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_folder='data/masks',
        ...     output_folder='output/tiles',
        ...     match_by_name=True
        ... )

        # Multiple mask files, matched by sorted order
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_folder='data/masks',
        ...     output_folder='output/tiles',
        ...     match_by_name=False
        ... )
    """

    import logging

    logging.getLogger("rasterio").setLevel(logging.ERROR)

    # Validate input parameters
    if masks_folder is not None and masks_file is not None:
        raise ValueError(
            "Cannot specify both masks_folder and masks_file. Please use only one."
        )

    # Default output folder if not specified
    if output_folder is None:
        output_folder = os.path.join(images_folder, "tiles")

    # Default extensions if not provided
    if image_extensions is None:
        image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".img"]
    if mask_extensions is None:
        mask_extensions = [
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
            ".jp2",
            ".img",
            ".shp",
            ".geojson",
            ".gpkg",
            ".geoparquet",
            ".json",
        ]

    # Convert extensions to lowercase for comparison
    image_extensions = [ext.lower() for ext in image_extensions]
    mask_extensions = [ext.lower() for ext in mask_extensions]

    # Create output folder structure
    os.makedirs(output_folder, exist_ok=True)
    output_images_dir = os.path.join(output_folder, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Only create masks directory if masks are provided
    output_masks_dir = None
    if masks_folder is not None or masks_file is not None:
        output_masks_dir = os.path.join(output_folder, "masks")
        os.makedirs(output_masks_dir, exist_ok=True)

    # Create annotation directory based on metadata format (only if masks are provided)
    ann_dir = None
    if (masks_folder is not None or masks_file is not None) and metadata_format in [
        "PASCAL_VOC",
        "COCO",
    ]:
        ann_dir = os.path.join(output_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

    # Initialize COCO annotations dictionary (only if masks are provided)
    coco_annotations = None
    if (
        masks_folder is not None or masks_file is not None
    ) and metadata_format == "COCO":
        coco_annotations = {"images": [], "annotations": [], "categories": []}

    # Initialize YOLO class set (only if masks are provided)
    yolo_classes = (
        set()
        if (masks_folder is not None or masks_file is not None)
        and metadata_format == "YOLO"
        else None
    )

    # Get list of image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(images_folder, f"*{ext}")
        image_files.extend(glob.glob(pattern))

    # Sort files for consistent processing
    image_files.sort()

    if not image_files:
        raise ValueError(
            f"No image files found in {images_folder} with extensions {image_extensions}"
        )

    # Handle different mask input modes
    use_single_mask_file = masks_file is not None
    has_masks = masks_file is not None or masks_folder is not None
    mask_files = []
    image_mask_pairs = []

    if not has_masks:
        # Mode 0: No masks - create pairs with None for mask
        for image_file in image_files:
            image_mask_pairs.append((image_file, None, None))

    elif use_single_mask_file:
        # Mode 1: Single vector file covering all images
        if not os.path.exists(masks_file):
            raise ValueError(f"Mask file not found: {masks_file}")

        # Load the single mask file once - will be spatially filtered per image
        single_mask_gdf = gpd.read_file(masks_file)

        if not quiet:
            print(f"Using single mask file: {masks_file}")
            print(
                f"Mask contains {len(single_mask_gdf)} features in CRS: {single_mask_gdf.crs}"
            )

        # Create pairs with the same mask file for all images
        for image_file in image_files:
            image_mask_pairs.append((image_file, masks_file, single_mask_gdf))

    else:
        # Mode 2/3: Multiple mask files (vector or raster)
        # Get list of mask files
        for ext in mask_extensions:
            pattern = os.path.join(masks_folder, f"*{ext}")
            mask_files.extend(glob.glob(pattern))

        # Sort files for consistent processing
        mask_files.sort()

        if not mask_files:
            raise ValueError(
                f"No mask files found in {masks_folder} with extensions {mask_extensions}"
            )

        # Match images to masks
        if match_by_name:
            # Match by base filename
            image_dict = {
                os.path.splitext(os.path.basename(f))[0]: f for f in image_files
            }
            mask_dict = {
                os.path.splitext(os.path.basename(f))[0]: f for f in mask_files
            }

            # Find matching pairs
            for img_base, img_path in image_dict.items():
                if img_base in mask_dict:
                    image_mask_pairs.append((img_path, mask_dict[img_base], None))
                else:
                    if not quiet:
                        print(f"Warning: No mask found for image {img_base}")

            if not image_mask_pairs:
                # Provide detailed error message with found files
                image_bases = list(image_dict.keys())
                mask_bases = list(mask_dict.keys())
                error_msg = (
                    "No matching image-mask pairs found when matching by filename. "
                    "Check that image and mask files have matching base names.\n"
                    f"Found {len(image_bases)} image(s): "
                    f"{', '.join(image_bases[:5]) if image_bases else 'None found'}"
                    f"{'...' if len(image_bases) > 5 else ''}\n"
                    f"Found {len(mask_bases)} mask(s): "
                    f"{', '.join(mask_bases[:5]) if mask_bases else 'None found'}"
                    f"{'...' if len(mask_bases) > 5 else ''}\n"
                    "Tip: Set match_by_name=False to match by sorted order, or ensure filenames match."
                )
                raise ValueError(error_msg)

        else:
            # Match by sorted order
            if len(image_files) != len(mask_files):
                raise ValueError(
                    f"Number of image files ({len(image_files)}) does not match "
                    f"number of mask files ({len(mask_files)}) when matching by sorted order. "
                    f"Use match_by_name=True for filename-based matching."
                )

            # Create pairs by sorted order
            for image_file, mask_file in zip(image_files, mask_files):
                image_mask_pairs.append((image_file, mask_file, None))

    # Initialize batch statistics
    batch_stats = {
        "total_image_pairs": 0,
        "processed_pairs": 0,
        "total_tiles": 0,
        "tiles_with_features": 0,
        "errors": 0,
        "processed_files": [],
        "failed_files": [],
    }

    if not quiet:
        if not has_masks:
            print(
                f"Found {len(image_files)} image files to process (images only, no masks)"
            )
        elif use_single_mask_file:
            print(f"Found {len(image_files)} image files to process")
            print(f"Using single mask file: {masks_file}")
        else:
            print(f"Found {len(image_mask_pairs)} matching image-mask pairs to process")
            print(f"Processing batch from {images_folder} and {masks_folder}")
        print(f"Output folder: {output_folder}")
        print("-" * 60)

    # Global tile counter for unique naming
    global_tile_counter = 0

    # Process each image-mask pair
    for idx, (image_file, mask_file, mask_gdf) in enumerate(
        tqdm(
            image_mask_pairs,
            desc="Processing image pairs",
            disable=quiet,
        )
    ):
        batch_stats["total_image_pairs"] += 1

        # Get base filename without extension for naming (use image filename)
        base_name = os.path.splitext(os.path.basename(image_file))[0]

        try:
            if not quiet:
                print(f"\nProcessing: {base_name}")
                print(f"  Image: {os.path.basename(image_file)}")
                if mask_file is not None:
                    if use_single_mask_file:
                        print(
                            f"  Mask: {os.path.basename(mask_file)} (spatially filtered)"
                        )
                    else:
                        print(f"  Mask: {os.path.basename(mask_file)}")
                else:
                    print(f"  Mask: None (images only)")

            # Process the image-mask pair
            tiles_generated = _process_image_mask_pair(
                image_file=image_file,
                mask_file=mask_file,
                base_name=base_name,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                global_tile_counter=global_tile_counter,
                tile_size=tile_size,
                stride=stride,
                class_value_field=class_value_field,
                buffer_radius=buffer_radius,
                max_tiles=max_tiles,
                all_touched=all_touched,
                skip_empty_tiles=skip_empty_tiles,
                quiet=quiet,
                mask_gdf=mask_gdf,  # Pass pre-loaded GeoDataFrame if using single mask
                use_single_mask_file=use_single_mask_file,
                metadata_format=metadata_format,
                ann_dir=(
                    ann_dir
                    if "ann_dir" in locals()
                    and metadata_format in ["PASCAL_VOC", "COCO"]
                    else None
                ),
            )

            # Update counters
            global_tile_counter += tiles_generated["total_tiles"]

            # Update batch statistics
            batch_stats["processed_pairs"] += 1
            batch_stats["total_tiles"] += tiles_generated["total_tiles"]
            batch_stats["tiles_with_features"] += tiles_generated["tiles_with_features"]
            batch_stats["errors"] += tiles_generated["errors"]

            batch_stats["processed_files"].append(
                {
                    "image": image_file,
                    "mask": mask_file,
                    "base_name": base_name,
                    "tiles_generated": tiles_generated["total_tiles"],
                    "tiles_with_features": tiles_generated["tiles_with_features"],
                }
            )

            # Aggregate COCO annotations
            if metadata_format == "COCO" and "coco_data" in tiles_generated:
                coco_data = tiles_generated["coco_data"]
                # Add images and annotations
                coco_annotations["images"].extend(coco_data.get("images", []))
                coco_annotations["annotations"].extend(coco_data.get("annotations", []))
                # Merge categories (avoid duplicates)
                for cat in coco_data.get("categories", []):
                    if not any(
                        c["id"] == cat["id"] for c in coco_annotations["categories"]
                    ):
                        coco_annotations["categories"].append(cat)

            # Aggregate YOLO classes
            if metadata_format == "YOLO" and "yolo_classes" in tiles_generated:
                yolo_classes.update(tiles_generated["yolo_classes"])

        except Exception as e:
            if not quiet:
                print(f"ERROR processing {base_name}: {e}")
            batch_stats["failed_files"].append(
                {"image": image_file, "mask": mask_file, "error": str(e)}
            )
            batch_stats["errors"] += 1

    # Save aggregated COCO annotations
    if metadata_format == "COCO" and coco_annotations:
        import json

        coco_path = os.path.join(ann_dir, "instances.json")
        with open(coco_path, "w") as f:
            json.dump(coco_annotations, f, indent=2)
        if not quiet:
            print(f"\nSaved COCO annotations: {coco_path}")
            print(
                f"  Images: {len(coco_annotations['images'])}, "
                f"Annotations: {len(coco_annotations['annotations'])}, "
                f"Categories: {len(coco_annotations['categories'])}"
            )

    # Save aggregated YOLO classes
    if metadata_format == "YOLO" and yolo_classes:
        classes_path = os.path.join(output_folder, "labels", "classes.txt")
        os.makedirs(os.path.dirname(classes_path), exist_ok=True)
        sorted_classes = sorted(yolo_classes)
        with open(classes_path, "w") as f:
            for cls in sorted_classes:
                f.write(f"{cls}\n")
        if not quiet:
            print(f"\nSaved YOLO classes: {classes_path}")
            print(f"  Total classes: {len(sorted_classes)}")

    # Print batch summary
    if not quiet:
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total image pairs found: {batch_stats['total_image_pairs']}")
        print(f"Successfully processed: {batch_stats['processed_pairs']}")
        print(f"Failed to process: {len(batch_stats['failed_files'])}")
        print(f"Total tiles generated: {batch_stats['total_tiles']}")
        print(f"Tiles with features: {batch_stats['tiles_with_features']}")

        if batch_stats["total_tiles"] > 0:
            feature_percentage = (
                batch_stats["tiles_with_features"] / batch_stats["total_tiles"]
            ) * 100
            print(f"Feature percentage: {feature_percentage:.1f}%")

        if batch_stats["errors"] > 0:
            print(f"Total errors: {batch_stats['errors']}")

        print(f"Output saved to: {output_folder}")
        print(f"  Images: {output_images_dir}")
        if output_masks_dir is not None:
            print(f"  Masks: {output_masks_dir}")
            if metadata_format in ["PASCAL_VOC", "COCO"] and ann_dir is not None:
                print(f"  Annotations: {ann_dir}")
            elif metadata_format == "YOLO":
                print(f"  Labels: {os.path.join(output_folder, 'labels')}")

        # List failed files if any
        if batch_stats["failed_files"]:
            print(f"\nFailed files:")
            for failed in batch_stats["failed_files"]:
                print(f"  - {os.path.basename(failed['image'])}: {failed['error']}")

    return batch_stats


def _process_image_mask_pair(
    image_file,
    mask_file,
    base_name,
    output_images_dir,
    output_masks_dir,
    global_tile_counter,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    all_touched=True,
    skip_empty_tiles=False,
    quiet=False,
    mask_gdf=None,
    use_single_mask_file=False,
    metadata_format="PASCAL_VOC",
    ann_dir=None,
):
    """
    Process a single image-mask pair and save tiles directly to output directories.

    Args:
        mask_gdf (GeoDataFrame, optional): Pre-loaded GeoDataFrame when using single mask file
        use_single_mask_file (bool): If True, spatially filter mask_gdf to image bounds

    Returns:
        dict: Statistics for this image-mask pair
    """
    import warnings

    # Determine if mask data is raster or vector (only if mask_file is provided)
    is_class_data_raster = False
    if mask_file is not None and isinstance(mask_file, str):
        file_ext = Path(mask_file).suffix.lower()
        # Common raster extensions
        if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
            try:
                with rasterio.open(mask_file) as src:
                    is_class_data_raster = True
            except Exception:
                is_class_data_raster = False

    # Track statistics
    stats = {
        "total_tiles": 0,
        "tiles_with_features": 0,
        "errors": 0,
    }

    # Initialize COCO/YOLO tracking for this image
    if metadata_format == "COCO":
        stats["coco_data"] = {"images": [], "annotations": [], "categories": []}
        coco_ann_id = 0
    if metadata_format == "YOLO":
        stats["yolo_classes"] = set()

    # Open the input raster
    with rasterio.open(image_file) as src:
        # Calculate number of tiles
        num_tiles_x = math.ceil((src.width - tile_size) / stride) + 1
        num_tiles_y = math.ceil((src.height - tile_size) / stride) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if max_tiles is None:
            max_tiles = total_tiles

        # Process classification data (only if mask_file is provided)
        class_to_id = {}

        if mask_file is not None and is_class_data_raster:
            # Load raster class data
            with rasterio.open(mask_file) as class_src:
                # Check if raster CRS matches
                if class_src.crs != src.crs:
                    warnings.warn(
                        f"CRS mismatch: Class raster ({class_src.crs}) doesn't match input raster ({src.crs}). "
                        f"Results may be misaligned."
                    )

                # Get unique values from raster
                sample_data = class_src.read(
                    1,
                    out_shape=(
                        1,
                        min(class_src.height, 1000),
                        min(class_src.width, 1000),
                    ),
                )

                unique_classes = np.unique(sample_data)
                unique_classes = unique_classes[
                    unique_classes > 0
                ]  # Remove 0 as it's typically background

                # Create class mapping
                class_to_id = {int(cls): i + 1 for i, cls in enumerate(unique_classes)}
        elif mask_file is not None:
            # Load vector class data
            try:
                if use_single_mask_file and mask_gdf is not None:
                    # Using pre-loaded single mask file - spatially filter to image bounds
                    # Get image bounds
                    image_bounds = box(*src.bounds)
                    image_gdf = gpd.GeoDataFrame(
                        {"geometry": [image_bounds]}, crs=src.crs
                    )

                    # Reproject mask if needed
                    if mask_gdf.crs != src.crs:
                        mask_gdf_reprojected = mask_gdf.to_crs(src.crs)
                    else:
                        mask_gdf_reprojected = mask_gdf

                    # Spatially filter features that intersect with image bounds
                    gdf = mask_gdf_reprojected[
                        mask_gdf_reprojected.intersects(image_bounds)
                    ].copy()

                    if not quiet and len(gdf) > 0:
                        print(
                            f"  Filtered to {len(gdf)} features intersecting image bounds"
                        )
                else:
                    # Load individual mask file
                    gdf = gpd.read_file(mask_file)

                    # Always reproject to match raster CRS
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)

                # Apply buffer if specified
                if buffer_radius > 0:
                    gdf["geometry"] = gdf.buffer(buffer_radius)

                # Check if class_value_field exists
                if class_value_field in gdf.columns:
                    unique_classes = gdf[class_value_field].unique()
                    # Create class mapping
                    class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}
                else:
                    class_to_id = {1: 1}  # Default mapping
            except Exception as e:
                raise ValueError(f"Error processing vector data: {e}")

        # Process tiles
        tile_index = 0
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate window coordinates
                window_x = x * stride
                window_y = y * stride

                # Adjust for edge cases
                if window_x + tile_size > src.width:
                    window_x = src.width - tile_size
                if window_y + tile_size > src.height:
                    window_y = src.height - tile_size

                # Define window
                window = Window(window_x, window_y, tile_size, tile_size)

                # Get window transform and bounds
                window_transform = src.window_transform(window)

                # Calculate window bounds
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size * window_transform[0]  # Add width
                miny = maxy + tile_size * window_transform[4]  # Add height

                window_bounds = box(minx, miny, maxx, maxy)

                # Create label mask (only if mask_file is provided)
                label_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                has_features = False

                # Process classification data to create labels (only if mask_file is provided)
                if mask_file is not None and is_class_data_raster:
                    # For raster class data
                    with rasterio.open(mask_file) as class_src:
                        # Get corresponding window in class raster
                        window_class = rasterio.windows.from_bounds(
                            minx, miny, maxx, maxy, class_src.transform
                        )

                        # Read label data
                        try:
                            label_data = class_src.read(
                                1,
                                window=window_class,
                                boundless=True,
                                out_shape=(tile_size, tile_size),
                            )

                            # Remap class values if needed
                            if class_to_id:
                                remapped_data = np.zeros_like(label_data)
                                for orig_val, new_val in class_to_id.items():
                                    remapped_data[label_data == orig_val] = new_val
                                label_mask = remapped_data
                            else:
                                label_mask = label_data

                            # Check if we have any features
                            if np.any(label_mask > 0):
                                has_features = True
                        except Exception as e:
                            if not quiet:
                                print(f"Error reading class raster window: {e}")
                            stats["errors"] += 1
                elif mask_file is not None:
                    # For vector class data
                    # Find features that intersect with window
                    window_features = gdf[gdf.intersects(window_bounds)]

                    if len(window_features) > 0:
                        for idx, feature in window_features.iterrows():
                            # Get class value
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                class_id = class_to_id.get(class_val, 1)
                            else:
                                class_id = 1

                            # Get geometry in window coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                try:
                                    # Rasterize feature
                                    feature_mask = features.rasterize(
                                        [(geom, class_id)],
                                        out_shape=(tile_size, tile_size),
                                        transform=window_transform,
                                        fill=0,
                                        all_touched=all_touched,
                                    )

                                    # Add to label mask
                                    label_mask = np.maximum(label_mask, feature_mask)

                                    # Check if the feature was actually rasterized
                                    if np.any(feature_mask):
                                        has_features = True
                                except Exception as e:
                                    if not quiet:
                                        print(f"Error rasterizing feature {idx}: {e}")
                                    stats["errors"] += 1

                # Skip tile if no features and skip_empty_tiles is True (only applies when masks are provided)
                if mask_file is not None and skip_empty_tiles and not has_features:
                    continue

                # Check if we've reached max_tiles before saving
                if tile_index >= max_tiles:
                    break

                # Generate unique tile name
                tile_name = f"{base_name}_{global_tile_counter + tile_index:06d}"

                # Read image data
                image_data = src.read(window=window)

                # Export image as GeoTIFF
                image_path = os.path.join(output_images_dir, f"{tile_name}.tif")

                # Create profile for image GeoTIFF
                image_profile = src.profile.copy()
                image_profile.update(
                    {
                        "height": tile_size,
                        "width": tile_size,
                        "count": image_data.shape[0],
                        "transform": window_transform,
                    }
                )

                # Save image as GeoTIFF
                try:
                    with rasterio.open(image_path, "w", **image_profile) as dst:
                        dst.write(image_data)
                    stats["total_tiles"] += 1
                except Exception as e:
                    if not quiet:
                        print(f"ERROR saving image GeoTIFF: {e}")
                    stats["errors"] += 1

                # Export label as GeoTIFF (only if mask_file and output_masks_dir are provided)
                if mask_file is not None and output_masks_dir is not None:
                    # Create profile for label GeoTIFF
                    label_profile = {
                        "driver": "GTiff",
                        "height": tile_size,
                        "width": tile_size,
                        "count": 1,
                        "dtype": "uint8",
                        "crs": src.crs,
                        "transform": window_transform,
                    }

                    label_path = os.path.join(output_masks_dir, f"{tile_name}.tif")
                    try:
                        with rasterio.open(label_path, "w", **label_profile) as dst:
                            dst.write(label_mask.astype(np.uint8), 1)

                        if has_features:
                            stats["tiles_with_features"] += 1
                    except Exception as e:
                        if not quiet:
                            print(f"ERROR saving label GeoTIFF: {e}")
                        stats["errors"] += 1

                # Generate annotation metadata based on format (only if mask_file is provided)
                if (
                    mask_file is not None
                    and metadata_format == "PASCAL_VOC"
                    and ann_dir
                ):
                    # Create PASCAL VOC XML annotation
                    from lxml import etree as ET

                    annotation = ET.Element("annotation")
                    ET.SubElement(annotation, "folder").text = os.path.basename(
                        output_images_dir
                    )
                    ET.SubElement(annotation, "filename").text = f"{tile_name}.tif"
                    ET.SubElement(annotation, "path").text = image_path

                    source = ET.SubElement(annotation, "source")
                    ET.SubElement(source, "database").text = "GeoAI"

                    size = ET.SubElement(annotation, "size")
                    ET.SubElement(size, "width").text = str(tile_size)
                    ET.SubElement(size, "height").text = str(tile_size)
                    ET.SubElement(size, "depth").text = str(image_data.shape[0])

                    ET.SubElement(annotation, "segmented").text = "1"

                    # Find connected components for instance segmentation
                    from scipy import ndimage

                    for class_id in np.unique(label_mask):
                        if class_id == 0:
                            continue

                        class_mask = (label_mask == class_id).astype(np.uint8)
                        labeled_array, num_features = ndimage.label(class_mask)

                        for instance_id in range(1, num_features + 1):
                            instance_mask = labeled_array == instance_id
                            coords = np.argwhere(instance_mask)

                            if len(coords) == 0:
                                continue

                            ymin, xmin = coords.min(axis=0)
                            ymax, xmax = coords.max(axis=0)

                            obj = ET.SubElement(annotation, "object")
                            class_name = next(
                                (k for k, v in class_to_id.items() if v == class_id),
                                str(class_id),
                            )
                            ET.SubElement(obj, "name").text = str(class_name)
                            ET.SubElement(obj, "pose").text = "Unspecified"
                            ET.SubElement(obj, "truncated").text = "0"
                            ET.SubElement(obj, "difficult").text = "0"

                            bndbox = ET.SubElement(obj, "bndbox")
                            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                            ET.SubElement(bndbox, "ymax").text = str(int(ymax))

                    # Save XML file
                    xml_path = os.path.join(ann_dir, f"{tile_name}.xml")
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path, pretty_print=True, encoding="utf-8")

                elif mask_file is not None and metadata_format == "COCO":
                    # Add COCO image entry
                    image_id = int(global_tile_counter + tile_index)
                    stats["coco_data"]["images"].append(
                        {
                            "id": image_id,
                            "file_name": f"{tile_name}.tif",
                            "width": int(tile_size),
                            "height": int(tile_size),
                        }
                    )

                    # Add COCO categories (only once per unique class)
                    for class_val, class_id in class_to_id.items():
                        if not any(
                            c["id"] == class_id
                            for c in stats["coco_data"]["categories"]
                        ):
                            stats["coco_data"]["categories"].append(
                                {
                                    "id": int(class_id),
                                    "name": str(class_val),
                                    "supercategory": "object",
                                }
                            )

                    # Add COCO annotations (instance segmentation)
                    from scipy import ndimage
                    from skimage import measure

                    for class_id in np.unique(label_mask):
                        if class_id == 0:
                            continue

                        class_mask = (label_mask == class_id).astype(np.uint8)
                        labeled_array, num_features = ndimage.label(class_mask)

                        for instance_id in range(1, num_features + 1):
                            instance_mask = (labeled_array == instance_id).astype(
                                np.uint8
                            )
                            coords = np.argwhere(instance_mask)

                            if len(coords) == 0:
                                continue

                            ymin, xmin = coords.min(axis=0)
                            ymax, xmax = coords.max(axis=0)

                            bbox = [
                                int(xmin),
                                int(ymin),
                                int(xmax - xmin),
                                int(ymax - ymin),
                            ]
                            area = int(np.sum(instance_mask))

                            # Find contours for segmentation
                            contours = measure.find_contours(instance_mask, 0.5)
                            segmentation = []
                            for contour in contours:
                                contour = np.flip(contour, axis=1)
                                segmentation_points = contour.ravel().tolist()
                                if len(segmentation_points) >= 6:
                                    segmentation.append(segmentation_points)

                            if segmentation:
                                stats["coco_data"]["annotations"].append(
                                    {
                                        "id": int(coco_ann_id),
                                        "image_id": int(image_id),
                                        "category_id": int(class_id),
                                        "bbox": bbox,
                                        "area": area,
                                        "segmentation": segmentation,
                                        "iscrowd": 0,
                                    }
                                )
                                coco_ann_id += 1

                elif mask_file is not None and metadata_format == "YOLO":
                    # Create YOLO labels directory if needed
                    labels_dir = os.path.join(
                        os.path.dirname(output_images_dir), "labels"
                    )
                    os.makedirs(labels_dir, exist_ok=True)

                    # Generate YOLO annotation file
                    yolo_path = os.path.join(labels_dir, f"{tile_name}.txt")
                    from scipy import ndimage

                    with open(yolo_path, "w") as yolo_file:
                        for class_id in np.unique(label_mask):
                            if class_id == 0:
                                continue

                            # Track class for classes.txt
                            class_name = next(
                                (k for k, v in class_to_id.items() if v == class_id),
                                str(class_id),
                            )
                            stats["yolo_classes"].add(class_name)

                            class_mask = (label_mask == class_id).astype(np.uint8)
                            labeled_array, num_features = ndimage.label(class_mask)

                            for instance_id in range(1, num_features + 1):
                                instance_mask = labeled_array == instance_id
                                coords = np.argwhere(instance_mask)

                                if len(coords) == 0:
                                    continue

                                ymin, xmin = coords.min(axis=0)
                                ymax, xmax = coords.max(axis=0)

                                # Convert to YOLO format (normalized center coordinates)
                                x_center = ((xmin + xmax) / 2) / tile_size
                                y_center = ((ymin + ymax) / 2) / tile_size
                                width = (xmax - xmin) / tile_size
                                height = (ymax - ymin) / tile_size

                                # YOLO uses 0-based class indices
                                yolo_class_id = class_id - 1
                                yolo_file.write(
                                    f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                )

                tile_index += 1
                if tile_index >= max_tiles:
                    break

            if tile_index >= max_tiles:
                break

    return stats


def display_training_tiles(
    output_dir,
    num_tiles=6,
    figsize=(18, 6),
    cmap="gray",
    save_path=None,
):
    """
    Display image and mask tile pairs from training data output.

    Args:
        output_dir (str): Path to output directory containing 'images' and 'masks' subdirectories
        num_tiles (int): Number of tile pairs to display (default: 6)
        figsize (tuple): Figure size as (width, height) in inches (default: (18, 6))
        cmap (str): Colormap for mask display (default: 'gray')
        save_path (str, optional): If provided, save figure to this path instead of displaying

    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects

    Example:
        >>> fig, axes = display_training_tiles('output/tiles', num_tiles=6)
        >>> # Or save to file
        >>> display_training_tiles('output/tiles', num_tiles=4, save_path='tiles_preview.png')
    """
    import matplotlib.pyplot as plt

    # Get list of image tiles
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")

    image_tiles = sorted(os.listdir(images_dir))[:num_tiles]

    if not image_tiles:
        raise ValueError(f"No image tiles found in {images_dir}")

    # Limit to available tiles
    num_tiles = min(num_tiles, len(image_tiles))

    # Create figure with subplots
    fig, axes = plt.subplots(2, num_tiles, figsize=figsize)

    # Handle case where num_tiles is 1
    if num_tiles == 1:
        axes = axes.reshape(2, 1)

    for idx, tile_name in enumerate(image_tiles):
        # Load and display image tile
        image_path = os.path.join(output_dir, "images", tile_name)
        with rasterio.open(image_path) as src:
            show(src, ax=axes[0, idx], title=f"Image {idx+1}")

        # Load and display mask tile
        mask_path = os.path.join(output_dir, "masks", tile_name)
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                show(src, ax=axes[1, idx], title=f"Mask {idx+1}", cmap=cmap)
        else:
            axes[1, idx].text(
                0.5,
                0.5,
                "Mask not found",
                ha="center",
                va="center",
                transform=axes[1, idx].transAxes,
            )
            axes[1, idx].set_title(f"Mask {idx+1}")

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig, axes


def display_image_with_vector(
    image_path,
    vector_path,
    figsize=(16, 8),
    vector_color="red",
    vector_linewidth=1,
    vector_facecolor="none",
    save_path=None,
):
    """
    Display a raster image alongside the same image with vector overlay.

    Args:
        image_path (str): Path to raster image file
        vector_path (str): Path to vector file (GeoJSON, Shapefile, etc.)
        figsize (tuple): Figure size as (width, height) in inches (default: (16, 8))
        vector_color (str): Edge color for vector features (default: 'red')
        vector_linewidth (float): Line width for vector features (default: 1)
        vector_facecolor (str): Fill color for vector features (default: 'none')
        save_path (str, optional): If provided, save figure to this path instead of displaying

    Returns:
        tuple: (fig, axes, info_dict) where info_dict contains image and vector metadata

    Example:
        >>> fig, axes, info = display_image_with_vector(
        ...     'image.tif',
        ...     'buildings.geojson',
        ...     vector_color='blue'
        ... )
        >>> print(f"Number of features: {info['num_features']}")
    """
    import matplotlib.pyplot as plt

    # Validate inputs
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    if not os.path.exists(vector_path):
        raise ValueError(f"Vector file not found: {vector_path}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Load and display image
    with rasterio.open(image_path) as src:
        # Plot image only
        show(src, ax=ax1, title="Image")

        # Load vector data
        vector_data = gpd.read_file(vector_path)

        # Reproject to image CRS if needed
        if vector_data.crs != src.crs:
            vector_data = vector_data.to_crs(src.crs)

        # Plot image with vector overlay
        show(
            src,
            ax=ax2,
            title=f"Image with {len(vector_data)} Vector Features",
        )
        vector_data.plot(
            ax=ax2,
            facecolor=vector_facecolor,
            edgecolor=vector_color,
            linewidth=vector_linewidth,
        )

        # Collect metadata
        info = {
            "image_shape": src.shape,
            "image_crs": src.crs,
            "image_bounds": src.bounds,
            "num_features": len(vector_data),
            "vector_crs": vector_data.crs,
            "vector_bounds": vector_data.total_bounds,
        }

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2), info


def create_overview_image(
    src, tile_coordinates, output_path, tile_size, stride, geojson_path=None
) -> str:
    """Create an overview image showing all tiles and their status, with optional GeoJSON export.

    Args:
        src (rasterio.io.DatasetReader): The source raster dataset.
        tile_coordinates (list): A list of dictionaries containing tile information.
        output_path (str): The path where the overview image will be saved.
        tile_size (int): The size of each tile in pixels.
        stride (int): The stride between tiles in pixels. Controls overlap between adjacent tiles.
        geojson_path (str, optional): If provided, exports the tile rectangles as GeoJSON to this path.

    Returns:
        str: Path to the saved overview image.
    """
    # Read a reduced version of the source image
    overview_scale = max(
        1, int(max(src.width, src.height) / 2000)
    )  # Scale to max ~2000px
    overview_width = src.width // overview_scale
    overview_height = src.height // overview_scale

    # Read downsampled image
    overview_data = src.read(
        out_shape=(src.count, overview_height, overview_width),
        resampling=rasterio.enums.Resampling.average,
    )

    # Create RGB image for display
    if overview_data.shape[0] >= 3:
        rgb = np.moveaxis(overview_data[:3], 0, -1)
    else:
        # For single band, create grayscale RGB
        rgb = np.stack([overview_data[0], overview_data[0], overview_data[0]], axis=-1)

    # Normalize for display
    for i in range(rgb.shape[-1]):
        band = rgb[..., i]
        non_zero = band[band > 0]
        if len(non_zero) > 0:
            p2, p98 = np.percentile(non_zero, (2, 98))
            rgb[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)

    # Create figure
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)

    # If GeoJSON export is requested, prepare GeoJSON structures
    if geojson_path:
        features = []

    # Draw tile boundaries
    for tile in tile_coordinates:
        # Convert bounds to pixel coordinates in overview
        bounds = tile["bounds"]
        # Calculate scaled pixel coordinates
        x_min = int((tile["x"]) / overview_scale)
        y_min = int((tile["y"]) / overview_scale)
        width = int(tile_size / overview_scale)
        height = int(tile_size / overview_scale)

        # Draw rectangle
        color = "lime" if tile["has_features"] else "red"
        rect = plt.Rectangle(
            (x_min, y_min), width, height, fill=False, edgecolor=color, linewidth=0.5
        )
        plt.gca().add_patch(rect)

        # Add tile number if not too crowded
        if width > 20 and height > 20:
            plt.text(
                x_min + width / 2,
                y_min + height / 2,
                str(tile["index"]),
                color="white",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Add to GeoJSON features if exporting
        if geojson_path:
            # Create a polygon from the bounds (already in geo-coordinates)
            minx, miny, maxx, maxy = bounds
            polygon = box(minx, miny, maxx, maxy)

            # Calculate overlap with neighboring tiles
            overlap = 0
            if stride < tile_size:
                overlap = tile_size - stride

            # Create a GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "index": tile["index"],
                    "has_features": tile["has_features"],
                    "bounds_pixel": [
                        tile["x"],
                        tile["y"],
                        tile["x"] + tile_size,
                        tile["y"] + tile_size,
                    ],
                    "tile_size_px": tile_size,
                    "stride_px": stride,
                    "overlap_px": overlap,
                },
            }

            # Add any additional properties from the tile
            for key, value in tile.items():
                if key not in ["x", "y", "index", "has_features", "bounds"]:
                    feature["properties"][key] = value

            features.append(feature)

    plt.title("Tile Overview (Green = Contains Features, Red = Empty)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Overview image saved to {output_path}")

    # Export GeoJSON if requested
    if geojson_path:
        geojson_collection = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "crs": (
                    src.crs.to_string()
                    if hasattr(src.crs, "to_string")
                    else str(src.crs)
                ),
                "total_tiles": len(features),
                "source_raster_dimensions": [src.width, src.height],
            },
        }

        # Save to file
        with open(geojson_path, "w") as f:
            json.dump(geojson_collection, f)

        print(f"GeoJSON saved to {geojson_path}")

    return output_path


def export_tiles_to_geojson(
    tile_coordinates, src, output_path, tile_size=None, stride=None
) -> str:
    """
    Export tile rectangles directly to GeoJSON without creating an overview image.

    Args:
        tile_coordinates (list): A list of dictionaries containing tile information.
        src (rasterio.io.DatasetReader): The source raster dataset.
        output_path (str): The path where the GeoJSON will be saved.
        tile_size (int, optional): The size of each tile in pixels. Only needed if not in tile_coordinates.
        stride (int, optional): The stride between tiles in pixels. Used to calculate overlaps between tiles.

    Returns:
        str: Path to the saved GeoJSON file.
    """
    features = []

    for tile in tile_coordinates:
        # Get the size from the tile or use the provided parameter
        tile_width = tile.get("width", tile.get("size", tile_size))
        tile_height = tile.get("height", tile.get("size", tile_size))

        if tile_width is None or tile_height is None:
            raise ValueError(
                "Tile size not found in tile data and no tile_size parameter provided"
            )

        # Get bounds from the tile
        if "bounds" in tile:
            # If bounds are already in geo coordinates
            minx, miny, maxx, maxy = tile["bounds"]
        else:
            # Try to calculate bounds from transform if available
            if hasattr(src, "transform"):
                # Convert pixel coordinates to geo coordinates
                window_transform = src.transform
                x, y = tile["x"], tile["y"]
                minx = window_transform[2] + x * window_transform[0]
                maxy = window_transform[5] + y * window_transform[4]
                maxx = minx + tile_width * window_transform[0]
                miny = maxy + tile_height * window_transform[4]
            else:
                raise ValueError(
                    "Cannot determine bounds. Neither 'bounds' in tile nor transform in src."
                )

        # Calculate overlap with neighboring tiles if stride is provided
        overlap = 0
        if stride is not None and stride < tile_width:
            overlap = tile_width - stride

        # Create a polygon from the bounds
        polygon = box(minx, miny, maxx, maxy)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "index": tile["index"],
                "has_features": tile.get("has_features", False),
                "tile_width_px": tile_width,
                "tile_height_px": tile_height,
            },
        }

        # Add overlap information if stride is provided
        if stride is not None:
            feature["properties"]["stride_px"] = stride
            feature["properties"]["overlap_px"] = overlap

        # Add additional properties from the tile
        for key, value in tile.items():
            if key not in ["bounds", "geometry"]:
                feature["properties"][key] = value

        features.append(feature)

    # Create the GeoJSON collection
    geojson_collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "crs": (
                src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
            ),
            "total_tiles": len(features),
            "source_raster_dimensions": (
                [src.width, src.height] if hasattr(src, "width") else None
            ),
        },
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(geojson_collection, f)

    print(f"GeoJSON saved to {output_path}")
    return output_path


def export_training_data(
    in_raster,
    out_folder,
    in_class_data,
    image_chip_format="GEOTIFF",
    tile_size_x=256,
    tile_size_y=256,
    stride_x=None,
    stride_y=None,
    output_nofeature_tiles=True,
    metadata_format="PASCAL_VOC",
    start_index=0,
    class_value_field="class",
    buffer_radius=0,
    in_mask_polygons=None,
    rotation_angle=0,
    reference_system=None,
    blacken_around_feature=False,
    crop_mode="FIXED_SIZE",  # Implemented but not fully used yet
    in_raster2=None,
    in_instance_data=None,
    instance_class_value_field=None,  # Implemented but not fully used yet
    min_polygon_overlap_ratio=0.0,
    all_touched=True,
    save_geotiff=True,
    quiet=False,
):
    """
    Export training data for deep learning using TorchGeo with progress bar.

    Args:
        in_raster (str): Path to input raster image.
        out_folder (str): Output folder path where chips and labels will be saved.
        in_class_data (str): Path to vector file containing class polygons.
        image_chip_format (str): Output image format (PNG, JPEG, TIFF, GEOTIFF).
        tile_size_x (int): Width of image chips in pixels.
        tile_size_y (int): Height of image chips in pixels.
        stride_x (int): Horizontal stride between chips. If None, uses tile_size_x.
        stride_y (int): Vertical stride between chips. If None, uses tile_size_y.
        output_nofeature_tiles (bool): Whether to export chips without features.
        metadata_format (str): Output metadata format (PASCAL_VOC, KITTI, COCO).
        start_index (int): Starting index for chip filenames.
        class_value_field (str): Field name in in_class_data containing class values.
        buffer_radius (float): Buffer radius around features (in CRS units).
        in_mask_polygons (str): Path to vector file containing mask polygons.
        rotation_angle (float): Rotation angle in degrees.
        reference_system (str): Reference system code.
        blacken_around_feature (bool): Whether to mask areas outside of features.
        crop_mode (str): Crop mode (FIXED_SIZE, CENTERED_ON_FEATURE).
        in_raster2 (str): Path to secondary raster image.
        in_instance_data (str): Path to vector file containing instance polygons.
        instance_class_value_field (str): Field name in in_instance_data for instance classes.
        min_polygon_overlap_ratio (float): Minimum overlap ratio for polygons.
        all_touched (bool): Whether to use all_touched=True in rasterization.
        save_geotiff (bool): Whether to save as GeoTIFF with georeferencing.
        quiet (bool): If True, suppress most output messages.
    """
    # Create output directories
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    label_dir = os.path.join(out_folder, "labels")
    os.makedirs(label_dir, exist_ok=True)

    # Define annotation directories based on metadata format
    if metadata_format == "PASCAL_VOC":
        ann_dir = os.path.join(out_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
    elif metadata_format == "COCO":
        ann_dir = os.path.join(out_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        # Initialize COCO annotations dictionary
        coco_annotations = {"images": [], "annotations": [], "categories": []}

    # Initialize statistics dictionary
    stats = {
        "total_tiles": 0,
        "tiles_with_features": 0,
        "feature_pixels": 0,
        "errors": 0,
    }

    # Open raster
    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"\nRaster info for {in_raster}:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bounds: {src.bounds}")

        # Set defaults for stride if not provided
        if stride_x is None:
            stride_x = tile_size_x
        if stride_y is None:
            stride_y = tile_size_y

        # Calculate number of tiles in x and y directions
        num_tiles_x = math.ceil((src.width - tile_size_x) / stride_x) + 1
        num_tiles_y = math.ceil((src.height - tile_size_y) / stride_y) + 1
        total_tiles = num_tiles_x * num_tiles_y

        # Read class data
        gdf = gpd.read_file(in_class_data)
        if not quiet:
            print(f"Loaded {len(gdf)} features from {in_class_data}")
            print(f"Available columns: {gdf.columns.tolist()}")
            print(f"GeoJSON CRS: {gdf.crs}")

        # Check if class_value_field exists
        if class_value_field not in gdf.columns:
            if not quiet:
                print(
                    f"WARNING: '{class_value_field}' field not found in the input data. Using default class value 1."
                )
            # Add a default class column
            gdf[class_value_field] = 1
            unique_classes = [1]
        else:
            # Print unique classes for debugging
            unique_classes = gdf[class_value_field].unique()
            if not quiet:
                print(f"Found {len(unique_classes)} unique classes: {unique_classes}")

        # CRITICAL: Always reproject to match raster CRS to ensure proper alignment
        if gdf.crs != src.crs:
            if not quiet:
                print(f"Reprojecting features from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        elif reference_system and gdf.crs != reference_system:
            if not quiet:
                print(
                    f"Reprojecting features to specified reference system {reference_system}"
                )
            gdf = gdf.to_crs(reference_system)

        # Check overlap between raster and vector data
        raster_bounds = box(*src.bounds)
        vector_bounds = box(*gdf.total_bounds)
        if not raster_bounds.intersects(vector_bounds):
            if not quiet:
                print(
                    "WARNING: The vector data doesn't intersect with the raster extent!"
                )
                print(f"Raster bounds: {src.bounds}")
                print(f"Vector bounds: {gdf.total_bounds}")
        else:
            overlap = (
                raster_bounds.intersection(vector_bounds).area / vector_bounds.area
            )
            if not quiet:
                print(f"Overlap between raster and vector: {overlap:.2%}")

        # Apply buffer if specified
        if buffer_radius > 0:
            gdf["geometry"] = gdf.buffer(buffer_radius)

        # Initialize class mapping (ensure all classes are mapped to non-zero values)
        class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}

        # Store category info for COCO format
        if metadata_format == "COCO":
            for cls_val in unique_classes:
                coco_annotations["categories"].append(
                    {
                        "id": class_to_id[cls_val],
                        "name": str(cls_val),
                        "supercategory": "object",
                    }
                )

        # Load mask polygons if provided
        mask_gdf = None
        if in_mask_polygons:
            mask_gdf = gpd.read_file(in_mask_polygons)
            if reference_system:
                mask_gdf = mask_gdf.to_crs(reference_system)
            elif mask_gdf.crs != src.crs:
                mask_gdf = mask_gdf.to_crs(src.crs)

        # Process instance data if provided
        instance_gdf = None
        if in_instance_data:
            instance_gdf = gpd.read_file(in_instance_data)
            if reference_system:
                instance_gdf = instance_gdf.to_crs(reference_system)
            elif instance_gdf.crs != src.crs:
                instance_gdf = instance_gdf.to_crs(src.crs)

        # Load secondary raster if provided
        src2 = None
        if in_raster2:
            src2 = rasterio.open(in_raster2)

        # Set up augmentation if rotation is specified
        augmentation = None
        if rotation_angle != 0:
            # Fixed: Added data_keys parameter to AugmentationSequential
            augmentation = torchgeo.transforms.AugmentationSequential(
                torch.nn.ModuleList([RandomRotation(rotation_angle)]),
                data_keys=["image"],  # Add data_keys parameter
            )

        # Initialize annotation ID for COCO format
        ann_id = 0

        # Create progress bar
        pbar = tqdm(
            total=total_tiles,
            desc=f"Generating tiles (with features: 0)",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Generate tiles
        chip_index = start_index
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate window coordinates
                window_x = x * stride_x
                window_y = y * stride_y

                # Adjust for edge cases
                if window_x + tile_size_x > src.width:
                    window_x = src.width - tile_size_x
                if window_y + tile_size_y > src.height:
                    window_y = src.height - tile_size_y

                # Adjust window based on crop_mode
                if crop_mode == "CENTERED_ON_FEATURE" and len(gdf) > 0:
                    # Find the nearest feature to the center of this window
                    window_center_x = window_x + tile_size_x // 2
                    window_center_y = window_y + tile_size_y // 2

                    # Convert center to world coordinates
                    center_x, center_y = src.xy(window_center_y, window_center_x)
                    center_point = gpd.points_from_xy([center_x], [center_y])[0]

                    # Find nearest feature
                    distances = gdf.geometry.distance(center_point)
                    nearest_idx = distances.idxmin()
                    nearest_feature = gdf.iloc[nearest_idx]

                    # Get centroid of nearest feature
                    feature_centroid = nearest_feature.geometry.centroid

                    # Convert feature centroid to pixel coordinates
                    feature_row, feature_col = src.index(
                        feature_centroid.x, feature_centroid.y
                    )

                    # Adjust window to center on feature
                    window_x = max(
                        0, min(src.width - tile_size_x, feature_col - tile_size_x // 2)
                    )
                    window_y = max(
                        0, min(src.height - tile_size_y, feature_row - tile_size_y // 2)
                    )

                # Define window
                window = Window(window_x, window_y, tile_size_x, tile_size_y)

                # Get window transform and bounds in source CRS
                window_transform = src.window_transform(window)

                # Calculate window bounds more explicitly and accurately
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size_x * window_transform[0]  # Add width
                miny = (
                    maxy + tile_size_y * window_transform[4]
                )  # Add height (note: transform[4] is typically negative)

                window_bounds = box(minx, miny, maxx, maxy)

                # Apply rotation if specified
                if rotation_angle != 0:
                    window_bounds = rotate(
                        window_bounds, rotation_angle, origin="center"
                    )

                # Find features that intersect with window
                window_features = gdf[gdf.intersects(window_bounds)]

                # Process instance data if provided
                window_instances = None
                if instance_gdf is not None and instance_class_value_field is not None:
                    window_instances = instance_gdf[
                        instance_gdf.intersects(window_bounds)
                    ]
                    if len(window_instances) > 0:
                        if not quiet:
                            pbar.write(
                                f"Found {len(window_instances)} instances in tile {chip_index}"
                            )

                # Skip if no features and output_nofeature_tiles is False
                if not output_nofeature_tiles and len(window_features) == 0:
                    pbar.update(1)  # Still update progress bar
                    continue

                # Check polygon overlap ratio if specified
                if min_polygon_overlap_ratio > 0 and len(window_features) > 0:
                    valid_features = []
                    for _, feature in window_features.iterrows():
                        overlap_ratio = (
                            feature.geometry.intersection(window_bounds).area
                            / feature.geometry.area
                        )
                        if overlap_ratio >= min_polygon_overlap_ratio:
                            valid_features.append(feature)

                    if len(valid_features) > 0:
                        window_features = gpd.GeoDataFrame(valid_features)
                    elif not output_nofeature_tiles:
                        pbar.update(1)  # Still update progress bar
                        continue

                # Apply mask if provided
                if mask_gdf is not None:
                    mask_features = mask_gdf[mask_gdf.intersects(window_bounds)]
                    if len(mask_features) == 0:
                        pbar.update(1)  # Still update progress bar
                        continue

                # Read image data - keep original for GeoTIFF export
                orig_image_data = src.read(window=window)

                # Create a copy for processing
                image_data = orig_image_data.copy().astype(np.float32)

                # Normalize image data for processing
                for band in range(image_data.shape[0]):
                    band_min, band_max = np.percentile(image_data[band], (1, 99))
                    if band_max > band_min:
                        image_data[band] = np.clip(
                            (image_data[band] - band_min) / (band_max - band_min), 0, 1
                        )

                # Read secondary image data if provided
                if src2:
                    image_data2 = src2.read(window=window)
                    # Stack the two images
                    image_data = np.vstack((image_data, image_data2))

                # Apply blacken_around_feature if needed
                if blacken_around_feature and len(window_features) > 0:
                    mask = np.zeros((tile_size_y, tile_size_x), dtype=bool)
                    for _, feature in window_features.iterrows():
                        # Project feature to pixel coordinates
                        feature_pixels = features.rasterize(
                            [(feature.geometry, 1)],
                            out_shape=(tile_size_y, tile_size_x),
                            transform=window_transform,
                        )
                        mask = np.logical_or(mask, feature_pixels.astype(bool))

                    # Apply mask to image
                    for band in range(image_data.shape[0]):
                        temp = image_data[band, :, :]
                        temp[~mask] = 0
                        image_data[band, :, :] = temp

                # Apply rotation if specified
                if augmentation:
                    # Convert to torch tensor for augmentation
                    image_tensor = torch.from_numpy(image_data).unsqueeze(
                        0
                    )  # Add batch dimension
                    # Apply augmentation with proper data format
                    augmented = augmentation({"image": image_tensor})
                    image_data = (
                        augmented["image"].squeeze(0).numpy()
                    )  # Remove batch dimension

                # Create a processed version for regular image formats
                processed_image = (image_data * 255).astype(np.uint8)

                # Create label mask
                label_mask = np.zeros((tile_size_y, tile_size_x), dtype=np.uint8)
                has_features = False

                if len(window_features) > 0:
                    for idx, feature in window_features.iterrows():
                        # Get class value
                        class_val = (
                            feature[class_value_field]
                            if class_value_field in feature
                            else 1
                        )
                        if isinstance(class_val, str):
                            # If class is a string, use its position in the unique classes list
                            class_id = class_to_id.get(class_val, 1)
                        else:
                            # If class is already a number, use it directly
                            class_id = int(class_val) if class_val > 0 else 1

                        # Get the geometry in pixel coordinates
                        geom = feature.geometry.intersection(window_bounds)
                        if not geom.is_empty:
                            try:
                                # Rasterize the feature
                                feature_mask = features.rasterize(
                                    [(geom, class_id)],
                                    out_shape=(tile_size_y, tile_size_x),
                                    transform=window_transform,
                                    fill=0,
                                    all_touched=all_touched,
                                )

                                # Update mask with higher class values taking precedence
                                label_mask = np.maximum(label_mask, feature_mask)

                                # Check if any pixels were added
                                if np.any(feature_mask):
                                    has_features = True
                            except Exception as e:
                                if not quiet:
                                    pbar.write(f"Error rasterizing feature {idx}: {e}")
                                stats["errors"] += 1

                # Save as GeoTIFF if requested
                if save_geotiff or image_chip_format.upper() in [
                    "TIFF",
                    "TIF",
                    "GEOTIFF",
                ]:
                    # Standardize extension to .tif for GeoTIFF files
                    image_filename = f"tile_{chip_index:06d}.tif"
                    image_path = os.path.join(image_dir, image_filename)

                    # Create profile for the GeoTIFF
                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": tile_size_y,
                            "width": tile_size_x,
                            "count": orig_image_data.shape[0],
                            "transform": window_transform,
                        }
                    )

                    # Save the GeoTIFF with original data
                    try:
                        with rasterio.open(image_path, "w", **profile) as dst:
                            dst.write(orig_image_data)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving image GeoTIFF for tile {chip_index}: {e}"
                            )
                        stats["errors"] += 1
                else:
                    # For non-GeoTIFF formats, use PIL to save the image
                    image_filename = (
                        f"tile_{chip_index:06d}.{image_chip_format.lower()}"
                    )
                    image_path = os.path.join(image_dir, image_filename)

                    # Create PIL image for saving
                    if processed_image.shape[0] == 1:
                        img = Image.fromarray(processed_image[0])
                    elif processed_image.shape[0] == 3:
                        # For RGB, need to transpose and make sure it's the right data type
                        rgb_data = np.transpose(processed_image, (1, 2, 0))
                        img = Image.fromarray(rgb_data)
                    else:
                        # For multiband images, save only RGB or first three bands
                        rgb_data = np.transpose(processed_image[:3], (1, 2, 0))
                        img = Image.fromarray(rgb_data)

                    # Save image
                    try:
                        img.save(image_path)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        if not quiet:
                            pbar.write(f"ERROR saving image for tile {chip_index}: {e}")
                        stats["errors"] += 1

                # Save label as GeoTIFF
                label_filename = f"tile_{chip_index:06d}.tif"
                label_path = os.path.join(label_dir, label_filename)

                # Create profile for label GeoTIFF
                label_profile = {
                    "driver": "GTiff",
                    "height": tile_size_y,
                    "width": tile_size_x,
                    "count": 1,
                    "dtype": "uint8",
                    "crs": src.crs,
                    "transform": window_transform,
                }

                # Save label GeoTIFF
                try:
                    with rasterio.open(label_path, "w", **label_profile) as dst:
                        dst.write(label_mask, 1)

                    if has_features:
                        pixel_count = np.count_nonzero(label_mask)
                        stats["tiles_with_features"] += 1
                        stats["feature_pixels"] += pixel_count
                except Exception as e:
                    if not quiet:
                        pbar.write(f"ERROR saving label for tile {chip_index}: {e}")
                    stats["errors"] += 1

                # Also save a PNG version for easy visualization if requested
                if metadata_format == "PASCAL_VOC":
                    try:
                        # Ensure correct data type for PIL
                        png_label = label_mask.astype(np.uint8)
                        label_img = Image.fromarray(png_label)
                        label_png_path = os.path.join(
                            label_dir, f"tile_{chip_index:06d}.png"
                        )
                        label_img.save(label_png_path)
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving PNG label for tile {chip_index}: {e}"
                            )
                            pbar.write(
                                f"  Label mask shape: {label_mask.shape}, dtype: {label_mask.dtype}"
                            )
                            # Try again with explicit conversion
                            try:
                                # Alternative approach for problematic arrays
                                png_data = np.zeros(
                                    (tile_size_y, tile_size_x), dtype=np.uint8
                                )
                                np.copyto(png_data, label_mask, casting="unsafe")
                                label_img = Image.fromarray(png_data)
                                label_img.save(label_png_path)
                                pbar.write(
                                    f"  Succeeded using alternative conversion method"
                                )
                            except Exception as e2:
                                pbar.write(f"  Second attempt also failed: {e2}")
                                stats["errors"] += 1

                # Generate annotations
                if metadata_format == "PASCAL_VOC" and len(window_features) > 0:
                    # Create XML annotation
                    root = ET.Element("annotation")
                    ET.SubElement(root, "folder").text = "images"
                    ET.SubElement(root, "filename").text = image_filename

                    size = ET.SubElement(root, "size")
                    ET.SubElement(size, "width").text = str(tile_size_x)
                    ET.SubElement(size, "height").text = str(tile_size_y)
                    ET.SubElement(size, "depth").text = str(min(image_data.shape[0], 3))

                    # Add georeference information
                    geo = ET.SubElement(root, "georeference")
                    ET.SubElement(geo, "crs").text = str(src.crs)
                    ET.SubElement(geo, "transform").text = str(
                        window_transform
                    ).replace("\n", "")
                    ET.SubElement(geo, "bounds").text = (
                        f"{minx}, {miny}, {maxx}, {maxy}"
                    )

                    for _, feature in window_features.iterrows():
                        # Convert feature geometry to pixel coordinates
                        feature_bounds = feature.geometry.intersection(window_bounds)
                        if feature_bounds.is_empty:
                            continue

                        # Get pixel coordinates of bounds
                        minx_f, miny_f, maxx_f, maxy_f = feature_bounds.bounds

                        # Convert to pixel coordinates
                        col_min, row_min = ~window_transform * (minx_f, maxy_f)
                        col_max, row_max = ~window_transform * (maxx_f, miny_f)

                        # Ensure coordinates are within bounds
                        xmin = max(0, min(tile_size_x, int(col_min)))
                        ymin = max(0, min(tile_size_y, int(row_min)))
                        xmax = max(0, min(tile_size_x, int(col_max)))
                        ymax = max(0, min(tile_size_y, int(row_max)))

                        # Skip if box is too small
                        if xmax - xmin < 1 or ymax - ymin < 1:
                            continue

                        obj = ET.SubElement(root, "object")
                        ET.SubElement(obj, "name").text = str(
                            feature[class_value_field]
                        )
                        ET.SubElement(obj, "difficult").text = "0"

                        bbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bbox, "xmin").text = str(xmin)
                        ET.SubElement(bbox, "ymin").text = str(ymin)
                        ET.SubElement(bbox, "xmax").text = str(xmax)
                        ET.SubElement(bbox, "ymax").text = str(ymax)

                    # Save XML
                    try:
                        tree = ET.ElementTree(root)
                        xml_path = os.path.join(ann_dir, f"tile_{chip_index:06d}.xml")
                        tree.write(xml_path)
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving XML annotation for tile {chip_index}: {e}"
                            )
                        stats["errors"] += 1

                elif metadata_format == "COCO" and len(window_features) > 0:
                    # Add image info
                    image_id = chip_index
                    coco_annotations["images"].append(
                        {
                            "id": image_id,
                            "file_name": image_filename,
                            "width": tile_size_x,
                            "height": tile_size_y,
                            "crs": str(src.crs),
                            "transform": str(window_transform),
                        }
                    )

                    # Add annotations for each feature
                    for _, feature in window_features.iterrows():
                        feature_bounds = feature.geometry.intersection(window_bounds)
                        if feature_bounds.is_empty:
                            continue

                        # Get pixel coordinates of bounds
                        minx_f, miny_f, maxx_f, maxy_f = feature_bounds.bounds

                        # Convert to pixel coordinates
                        col_min, row_min = ~window_transform * (minx_f, maxy_f)
                        col_max, row_max = ~window_transform * (maxx_f, miny_f)

                        # Ensure coordinates are within bounds
                        xmin = max(0, min(tile_size_x, int(col_min)))
                        ymin = max(0, min(tile_size_y, int(row_min)))
                        xmax = max(0, min(tile_size_x, int(col_max)))
                        ymax = max(0, min(tile_size_y, int(row_max)))

                        # Skip if box is too small
                        if xmax - xmin < 1 or ymax - ymin < 1:
                            continue

                        width = xmax - xmin
                        height = ymax - ymin

                        # Add annotation
                        ann_id += 1
                        category_id = class_to_id[feature[class_value_field]]

                        coco_annotations["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [xmin, ymin, width, height],
                                "area": width * height,
                                "iscrowd": 0,
                            }
                        )

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Generated: {stats['total_tiles']}, With features: {stats['tiles_with_features']}"
                )

                chip_index += 1

        # Close progress bar
        pbar.close()

        # Save COCO annotations if applicable
        if metadata_format == "COCO":
            try:
                with open(os.path.join(ann_dir, "instances.json"), "w") as f:
                    json.dump(coco_annotations, f)
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving COCO annotations: {e}")
                stats["errors"] += 1

        # Close secondary raster if opened
        if src2:
            src2.close()

    # Print summary
    if not quiet:
        print("\n------- Export Summary -------")
        print(f"Total tiles exported: {stats['total_tiles']}")
        print(
            f"Tiles with features: {stats['tiles_with_features']} ({stats['tiles_with_features']/max(1, stats['total_tiles'])*100:.1f}%)"
        )
        if stats["tiles_with_features"] > 0:
            print(
                f"Average feature pixels per tile: {stats['feature_pixels']/stats['tiles_with_features']:.1f}"
            )
        if stats["errors"] > 0:
            print(f"Errors encountered: {stats['errors']}")
        print(f"Output saved to: {out_folder}")

        # Verify georeference in a sample image and label
        if stats["total_tiles"] > 0:
            print("\n------- Georeference Verification -------")
            sample_image = os.path.join(image_dir, f"tile_{start_index}.tif")
            sample_label = os.path.join(label_dir, f"tile_{start_index}.tif")

            if os.path.exists(sample_image):
                try:
                    with rasterio.open(sample_image) as img:
                        print(f"Image CRS: {img.crs}")
                        print(f"Image transform: {img.transform}")
                        print(
                            f"Image has georeference: {img.crs is not None and img.transform is not None}"
                        )
                        print(
                            f"Image dimensions: {img.width}x{img.height}, {img.count} bands, {img.dtypes[0]} type"
                        )
                except Exception as e:
                    print(f"Error verifying image georeference: {e}")

            if os.path.exists(sample_label):
                try:
                    with rasterio.open(sample_label) as lbl:
                        print(f"Label CRS: {lbl.crs}")
                        print(f"Label transform: {lbl.transform}")
                        print(
                            f"Label has georeference: {lbl.crs is not None and lbl.transform is not None}"
                        )
                        print(
                            f"Label dimensions: {lbl.width}x{lbl.height}, {lbl.count} bands, {lbl.dtypes[0]} type"
                        )
                except Exception as e:
                    print(f"Error verifying label georeference: {e}")

    # Return statistics
    return stats, out_folder


def masks_to_vector(
    mask_path: str,
    output_path: Optional[str] = None,
    simplify_tolerance: float = 1.0,
    mask_threshold: float = 0.5,
    min_object_area: int = 100,
    max_object_area: Optional[int] = None,
    nms_iou_threshold: float = 0.5,
) -> Any:
    """
    Convert a building mask GeoTIFF to vector polygons and save as a vector dataset.

    Args:
        mask_path: Path to the building masks GeoTIFF
        output_path: Path to save the output GeoJSON (default: mask_path with .geojson extension)
        simplify_tolerance: Tolerance for polygon simplification (default: self.simplify_tolerance)
        mask_threshold: Threshold for mask binarization (default: self.mask_threshold)
        min_object_area: Minimum area in pixels to keep a building (default: self.min_object_area)
        max_object_area: Maximum area in pixels to keep a building (default: self.max_object_area)
        nms_iou_threshold: IoU threshold for non-maximum suppression (default: self.nms_iou_threshold)

    Returns:
        Any: GeoDataFrame with building footprints
    """
    # Set default output path if not provided
    # if output_path is None:
    #     output_path = os.path.splitext(mask_path)[0] + ".geojson"

    print(f"Converting mask to GeoJSON with parameters:")
    print(f"- Mask threshold: {mask_threshold}")
    print(f"- Min building area: {min_object_area}")
    print(f"- Simplify tolerance: {simplify_tolerance}")
    print(f"- NMS IoU threshold: {nms_iou_threshold}")

    # Open the mask raster
    with rasterio.open(mask_path) as src:
        # Read the mask data
        mask_data = src.read(1)
        transform = src.transform
        crs = src.crs

        # Print mask statistics
        print(f"Mask dimensions: {mask_data.shape}")
        print(f"Mask value range: {mask_data.min()} to {mask_data.max()}")

        # Prepare for connected component analysis
        # Binarize the mask based on threshold
        binary_mask = (mask_data > (mask_threshold * 255)).astype(np.uint8)

        # Apply morphological operations for better results (optional)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        print(f"Found {num_labels-1} potential buildings")  # Subtract 1 for background

        # Create list to store polygons and confidence values
        all_polygons = []
        all_confidences = []

        # Process each component (skip the first one which is background)
        for i in tqdm(range(1, num_labels)):
            # Extract this building
            area = stats[i, cv2.CC_STAT_AREA]

            # Skip if too small
            if area < min_object_area:
                continue

            # Skip if too large
            if max_object_area is not None and area > max_object_area:
                continue

            # Create a mask for this building
            building_mask = (labels == i).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process each contour
            for contour in contours:
                # Skip if too few points
                if contour.shape[0] < 3:
                    continue

                # Simplify contour if it has many points
                if contour.shape[0] > 50 and simplify_tolerance > 0:
                    epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to list of (x, y) coordinates
                polygon_points = contour.reshape(-1, 2)

                # Convert pixel coordinates to geographic coordinates
                geo_points = []
                for x, y in polygon_points:
                    gx, gy = transform * (x, y)
                    geo_points.append((gx, gy))

                # Create Shapely polygon
                if len(geo_points) >= 3:
                    try:
                        shapely_poly = Polygon(geo_points)
                        if shapely_poly.is_valid and shapely_poly.area > 0:
                            all_polygons.append(shapely_poly)

                            # Calculate "confidence" as normalized size
                            # This is a proxy since we don't have model confidence scores
                            normalized_size = min(1.0, area / 1000)  # Cap at 1.0
                            all_confidences.append(normalized_size)
                    except Exception as e:
                        print(f"Error creating polygon: {e}")

        print(f"Created {len(all_polygons)} valid polygons")

        # Create GeoDataFrame
        if not all_polygons:
            print("No valid polygons found")
            return None

        gdf = gpd.GeoDataFrame(
            {
                "geometry": all_polygons,
                "confidence": all_confidences,
                "class": 1,  # Building class
            },
            crs=crs,
        )

        def filter_overlapping_polygons(gdf, **kwargs):
            """
            Filter overlapping polygons using non-maximum suppression.

            Args:
                gdf: GeoDataFrame with polygons
                **kwargs: Optional parameters:
                    nms_iou_threshold: IoU threshold for filtering

            Returns:
                Filtered GeoDataFrame
            """
            if len(gdf) <= 1:
                return gdf

            # Get parameters from kwargs or use instance defaults
            iou_threshold = kwargs.get("nms_iou_threshold", nms_iou_threshold)

            # Sort by confidence
            gdf = gdf.sort_values("confidence", ascending=False)

            # Fix any invalid geometries
            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: geom.buffer(0) if not geom.is_valid else geom
            )

            keep_indices = []
            polygons = gdf.geometry.values

            for i in range(len(polygons)):
                if i in keep_indices:
                    continue

                keep = True
                for j in keep_indices:
                    # Skip invalid geometries
                    if not polygons[i].is_valid or not polygons[j].is_valid:
                        continue

                    # Calculate IoU
                    try:
                        intersection = polygons[i].intersection(polygons[j]).area
                        union = polygons[i].area + polygons[j].area - intersection
                        iou = intersection / union if union > 0 else 0

                        if iou > iou_threshold:
                            keep = False
                            break
                    except Exception:
                        # Skip on topology exceptions
                        continue

                if keep:
                    keep_indices.append(i)

            return gdf.iloc[keep_indices]

        # Apply non-maximum suppression to remove overlapping polygons
        gdf = filter_overlapping_polygons(gdf, nms_iou_threshold=nms_iou_threshold)

        print(f"Final building count after filtering: {len(gdf)}")

        # Save to file
        if output_path is not None:
            gdf.to_file(output_path)
            print(f"Saved {len(gdf)} building footprints to {output_path}")

        return gdf


def read_vector(
    source: str, layer: Optional[str] = None, **kwargs: Any
) -> gpd.GeoDataFrame:
    """Reads vector data from various formats including GeoParquet.

    This function dynamically determines the file type based on extension
    and reads it into a GeoDataFrame. It supports both local files and HTTP/HTTPS URLs.

    Args:
        source: String path to the vector file or URL.
        layer: String or integer specifying which layer to read from multi-layer
            files (only applicable for formats like GPKG, GeoJSON, etc.).
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the underlying reader.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the vector data.

    Raises:
        ValueError: If the file format is not supported or source cannot be accessed.

    Examples:
        Read a local shapefile
        >>> gdf = read_vector("path/to/data.shp")
        >>>
        Read a GeoParquet file from URL
        >>> gdf = read_vector("https://example.com/data.parquet")
        >>>
        Read a specific layer from a GeoPackage
        >>> gdf = read_vector("path/to/data.gpkg", layer="layer_name")
    """

    import urllib.parse

    import fiona

    # Determine if source is a URL or local file
    parsed_url = urllib.parse.urlparse(source)
    is_url = parsed_url.scheme in ["http", "https"]

    # If it's a local file, check if it exists
    if not is_url and not os.path.exists(source):
        raise ValueError(f"File does not exist: {source}")

    # Get file extension
    _, ext = os.path.splitext(source)
    ext = ext.lower()

    # Handle GeoParquet files
    if ext in [".parquet", ".pq", ".geoparquet"]:
        return gpd.read_parquet(source, **kwargs)

    # Handle common vector formats
    if ext in [".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml", ".gpx"]:
        # For formats that might have multiple layers
        if ext in [".gpkg", ".gml"] and layer is not None:
            return gpd.read_file(source, layer=layer, **kwargs)
        return gpd.read_file(source, **kwargs)

    # Try to use fiona to identify valid layers for formats that might have them
    # Only attempt this for local files as fiona.listlayers might not work with URLs
    if layer is None and ext in [".gpkg", ".gml"] and not is_url:
        try:
            layers = fiona.listlayers(source)
            if layers:
                return gpd.read_file(source, layer=layers[0], **kwargs)
        except Exception:
            # If listing layers fails, we'll fall through to the generic read attempt
            pass

    # For other formats or when layer listing fails, attempt to read using GeoPandas
    try:
        return gpd.read_file(source, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not read from source '{source}': {str(e)}")


def read_raster(
    source: str,
    band: Optional[Union[int, List[int]]] = None,
    masked: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Reads raster data from various formats using rioxarray.

    This function reads raster data from local files or URLs into a rioxarray
    data structure with preserved geospatial metadata.

    Args:
        source: String path to the raster file or URL.
        band: Integer or list of integers specifying which band(s) to read.
            Defaults to None (all bands).
        masked: Boolean indicating whether to mask nodata values.
            Defaults to True.
        **kwargs: Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns:
        xarray.DataArray: A DataArray containing the raster data with geospatial
            metadata preserved.

    Raises:
        ValueError: If the file format is not supported or source cannot be accessed.

    Examples:
        Read a local GeoTIFF
        >>> raster = read_raster("path/to/data.tif")
        >>>
        Read only band 1 from a remote GeoTIFF
        >>> raster = read_raster("https://example.com/data.tif", band=1)
        >>>
        Read a raster without masking nodata values
        >>> raster = read_raster("path/to/data.tif", masked=False)
    """
    import urllib.parse

    from rasterio.errors import RasterioIOError

    # Determine if source is a URL or local file
    parsed_url = urllib.parse.urlparse(source)
    is_url = parsed_url.scheme in ["http", "https"]

    # If it's a local file, check if it exists
    if not is_url and not os.path.exists(source):
        raise ValueError(f"Raster file does not exist: {source}")

    try:
        # Open the raster with rioxarray
        raster = rxr.open_rasterio(source, masked=masked, **kwargs)

        # Handle band selection if specified
        if band is not None:
            if isinstance(band, (list, tuple)):
                # Convert from 1-based indexing to 0-based indexing
                band_indices = [b - 1 for b in band]
                raster = raster.isel(band=band_indices)
            else:
                # Single band selection (convert from 1-based to 0-based indexing)
                raster = raster.isel(band=band - 1)

        return raster

    except RasterioIOError as e:
        raise ValueError(f"Could not read raster from source '{source}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading raster data: {str(e)}")


def temp_file_path(ext: str) -> str:
    """Returns a temporary file path.

    Args:
        ext (str): The file extension.

    Returns:
        str: The temporary file path.
    """

    import tempfile
    import uuid

    if not ext.startswith("."):
        ext = "." + ext
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{ext}")

    return file_path


def region_groups(
    image: Union[str, "xr.DataArray", np.ndarray],
    connectivity: int = 1,
    min_size: int = 10,
    max_size: Optional[int] = None,
    threshold: Optional[int] = None,
    properties: Optional[List[str]] = None,
    intensity_image: Optional[Union[str, "xr.DataArray", np.ndarray]] = None,
    out_csv: Optional[str] = None,
    out_vector: Optional[str] = None,
    out_image: Optional[str] = None,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, "pd.DataFrame"], Tuple["xr.DataArray", "pd.DataFrame"]]:
    """
    Segment regions in an image and filter them based on size.

    Args:
        image (Union[str, xr.DataArray, np.ndarray]): Input image, can be a file
            path, xarray DataArray, or numpy array.
        connectivity (int, optional): Connectivity for labeling. Defaults to 1
            for 4-connectivity. Use 2 for 8-connectivity.
        min_size (int, optional): Minimum size of regions to keep. Defaults to 10.
        max_size (Optional[int], optional): Maximum size of regions to keep.
            Defaults to None.
        threshold (Optional[int], optional): Threshold for filling holes.
            Defaults to None, which is equal to min_size.
        properties (Optional[List[str]], optional): List of properties to measure.
            See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            Defaults to None.
        intensity_image (Optional[Union[str, xr.DataArray, np.ndarray]], optional):
            Intensity image to measure properties. Defaults to None.
        out_csv (Optional[str], optional): Path to save the properties as a CSV file.
            Defaults to None.
        out_vector (Optional[str], optional): Path to save the vector file.
            Defaults to None.
        out_image (Optional[str], optional): Path to save the output image.
            Defaults to None.

    Returns:
        Union[Tuple[np.ndarray, pd.DataFrame], Tuple[xr.DataArray, pd.DataFrame]]: Labeled image and properties DataFrame.
    """
    import scipy.ndimage as ndi
    from skimage import measure

    if isinstance(image, str):
        ds = rxr.open_rasterio(image)
        da = ds.sel(band=1)
        array = da.values.squeeze()
    elif isinstance(image, xr.DataArray):
        da = image
        array = image.values.squeeze()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise ValueError(
            "The input image must be a file path, xarray DataArray, or numpy array."
        )

    if threshold is None:
        threshold = min_size

    # Define a custom function to calculate median intensity
    def intensity_median(region, intensity_image):
        # Extract the intensity values for the region
        return np.median(intensity_image[region])

    # Add your custom function to the list of extra properties
    if intensity_image is not None:
        extra_props = (intensity_median,)
    else:
        extra_props = None

    if properties is None:
        properties = [
            "label",
            "area",
            "area_bbox",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "diameter_areagth",
            "extent",
            "orientation",
            "perimeter",
            "solidity",
        ]

        if intensity_image is not None:

            properties += [
                "intensity_max",
                "intensity_mean",
                "intensity_min",
                "intensity_std",
            ]

    if intensity_image is not None:
        if isinstance(intensity_image, str):
            ds = rxr.open_rasterio(intensity_image)
            intensity_da = ds.sel(band=1)
            intensity_image = intensity_da.values.squeeze()
        elif isinstance(intensity_image, xr.DataArray):
            intensity_image = intensity_image.values.squeeze()
        elif isinstance(intensity_image, np.ndarray):
            pass
        else:
            raise ValueError(
                "The intensity_image must be a file path, xarray DataArray, or numpy array."
            )

    label_image = measure.label(array, connectivity=connectivity)
    props = measure.regionprops_table(
        label_image, properties=properties, intensity_image=intensity_image, **kwargs
    )

    df = pd.DataFrame(props)

    # Get the labels of regions with area smaller than the threshold
    small_regions = df[df["area"] < min_size]["label"].values
    # Set the corresponding labels in the label_image to zero
    for region_label in small_regions:
        label_image[label_image == region_label] = 0

    if max_size is not None:
        large_regions = df[df["area"] > max_size]["label"].values
        for region_label in large_regions:
            label_image[label_image == region_label] = 0

    # Find the background (holes) which are zeros
    holes = label_image == 0

    # Label the holes (connected components in the background)
    labeled_holes, _ = ndi.label(holes)

    # Measure properties of the labeled holes, including area and bounding box
    hole_props = measure.regionprops(labeled_holes)

    # Loop through each hole and fill it if it is smaller than the threshold
    for prop in hole_props:
        if prop.area < threshold:
            # Get the coordinates of the small hole
            coords = prop.coords

            # Find the surrounding region's ID (non-zero value near the hole)
            surrounding_region_values = []
            for coord in coords:
                x, y = coord
                # Get a 3x3 neighborhood around the hole pixel
                neighbors = label_image[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2]
                # Exclude the hole pixels (zeros) and get region values
                region_values = neighbors[neighbors != 0]
                if region_values.size > 0:
                    surrounding_region_values.append(
                        region_values[0]
                    )  # Take the first non-zero value

            if surrounding_region_values:
                # Fill the hole with the mode (most frequent) of the surrounding region values
                fill_value = max(
                    set(surrounding_region_values), key=surrounding_region_values.count
                )
                label_image[coords[:, 0], coords[:, 1]] = fill_value

    label_image, num_labels = measure.label(
        label_image, connectivity=connectivity, return_num=True
    )
    props = measure.regionprops_table(
        label_image,
        properties=properties,
        intensity_image=intensity_image,
        extra_properties=extra_props,
        **kwargs,
    )

    df = pd.DataFrame(props)
    df["elongation"] = df["axis_major_length"] / df["axis_minor_length"]

    dtype = "uint8"
    if num_labels > 255 and num_labels <= 65535:
        dtype = "uint16"
    elif num_labels > 65535:
        dtype = "uint32"

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

    if isinstance(image, np.ndarray):
        return label_image, df
    else:
        da.values = label_image
        if out_image is not None:
            da.rio.to_raster(out_image, dtype=dtype)

        if out_vector is not None:
            tmp_raster = None
            tmp_vector = None
            try:
                if out_image is None:
                    tmp_raster = temp_file_path(".tif")
                    da.rio.to_raster(tmp_raster, dtype=dtype)
                    tmp_vector = temp_file_path(".gpkg")
                    raster_to_vector(
                        tmp_raster,
                        tmp_vector,
                        attribute_name="value",
                        unique_attribute_value=True,
                    )
                else:
                    tmp_vector = temp_file_path(".gpkg")
                    raster_to_vector(
                        out_image,
                        tmp_vector,
                        attribute_name="value",
                        unique_attribute_value=True,
                    )
                gdf = gpd.read_file(tmp_vector)
                gdf["label"] = gdf["value"].astype(int)
                gdf.drop(columns=["value"], inplace=True)
                gdf2 = pd.merge(gdf, df, on="label", how="left")
                gdf2.to_file(out_vector)
                gdf2.sort_values("label", inplace=True)
                df = gdf2
            finally:
                try:
                    if tmp_raster is not None and os.path.exists(tmp_raster):
                        os.remove(tmp_raster)
                    if tmp_vector is not None and os.path.exists(tmp_vector):
                        os.remove(tmp_vector)
                except Exception as e:
                    print(f"Warning: Failed to delete temporary files: {str(e)}")

        return da, df


def add_geometric_properties(
    data: gpd.GeoDataFrame,
    properties: Optional[List[str]] = None,
    area_unit: str = "m2",
    length_unit: str = "m",
) -> gpd.GeoDataFrame:
    """Calculates geometric properties and adds them to the GeoDataFrame.

    This function calculates various geometric properties of features in a
    GeoDataFrame and adds them as new columns without modifying existing attributes.

    Args:
        data: GeoDataFrame containing vector features.
        properties: List of geometric properties to calculate. Options include:
            'area', 'length', 'perimeter', 'centroid_x', 'centroid_y', 'bounds',
            'convex_hull_area', 'orientation', 'complexity', 'area_bbox',
            'area_convex', 'area_filled', 'major_length', 'minor_length',
            'eccentricity', 'diameter_areagth', 'extent', 'solidity',
            'elongation'.
            Defaults to ['area', 'length'] if None.
        area_unit: String specifying the unit for area calculation ('m2', 'km2',
            'ha'). Defaults to 'm2'.
        length_unit: String specifying the unit for length calculation ('m', 'km').
            Defaults to 'm'.

    Returns:
        geopandas.GeoDataFrame: A copy of the input GeoDataFrame with added
        geometric property columns.
    """
    from shapely.ops import unary_union

    if isinstance(data, str):
        data = read_vector(data)

    # Make a copy to avoid modifying the original
    result = data.copy()

    # Default properties to calculate
    if properties is None:
        properties = [
            "area",
            "length",
            "perimeter",
            "convex_hull_area",
            "orientation",
            "complexity",
            "area_bbox",
            "area_convex",
            "area_filled",
            "major_length",
            "minor_length",
            "eccentricity",
            "diameter_area",
            "extent",
            "solidity",
            "elongation",
        ]

    # Make sure we're working with a GeoDataFrame with a valid CRS

    if not isinstance(result, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")

    if result.crs is None:
        raise ValueError(
            "GeoDataFrame must have a defined coordinate reference system (CRS)"
        )

    # Ensure we're working with a projected CRS for accurate measurements
    if result.crs.is_geographic:
        # Reproject to a suitable projected CRS for accurate measurements
        result = result.to_crs(result.estimate_utm_crs())

    # Basic area calculation with unit conversion
    if "area" in properties:
        # Calculate area (only for polygons)
        result["area"] = result.geometry.apply(
            lambda geom: geom.area if isinstance(geom, (Polygon, MultiPolygon)) else 0
        )

        # Convert to requested units
        if area_unit == "km2":
            result["area"] = result["area"] / 1_000_000  # m² to km²
            result.rename(columns={"area": "area_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area"] = result["area"] / 10_000  # m² to hectares
            result.rename(columns={"area": "area_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area": "area_m2"}, inplace=True)

    # Length calculation with unit conversion
    if "length" in properties:
        # Calculate length (works for lines and polygon boundaries)
        result["length"] = result.geometry.length

        # Convert to requested units
        if length_unit == "km":
            result["length"] = result["length"] / 1_000  # m to km
            result.rename(columns={"length": "length_km"}, inplace=True)
        else:  # Default is m
            result.rename(columns={"length": "length_m"}, inplace=True)

    # Perimeter calculation (for polygons)
    if "perimeter" in properties:
        result["perimeter"] = result.geometry.apply(
            lambda geom: (
                geom.boundary.length if isinstance(geom, (Polygon, MultiPolygon)) else 0
            )
        )

        # Convert to requested units
        if length_unit == "km":
            result["perimeter"] = result["perimeter"] / 1_000  # m to km
            result.rename(columns={"perimeter": "perimeter_km"}, inplace=True)
        else:  # Default is m
            result.rename(columns={"perimeter": "perimeter_m"}, inplace=True)

    # Centroid coordinates
    if "centroid_x" in properties or "centroid_y" in properties:
        centroids = result.geometry.centroid

        if "centroid_x" in properties:
            result["centroid_x"] = centroids.x

        if "centroid_y" in properties:
            result["centroid_y"] = centroids.y

    # Bounding box properties
    if "bounds" in properties:
        bounds = result.geometry.bounds
        result["minx"] = bounds.minx
        result["miny"] = bounds.miny
        result["maxx"] = bounds.maxx
        result["maxy"] = bounds.maxy

    # Area of bounding box
    if "area_bbox" in properties:
        bounds = result.geometry.bounds
        result["area_bbox"] = (bounds.maxx - bounds.minx) * (bounds.maxy - bounds.miny)

        # Convert to requested units
        if area_unit == "km2":
            result["area_bbox"] = result["area_bbox"] / 1_000_000
            result.rename(columns={"area_bbox": "area_bbox_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_bbox"] = result["area_bbox"] / 10_000
            result.rename(columns={"area_bbox": "area_bbox_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_bbox": "area_bbox_m2"}, inplace=True)

    # Area of convex hull
    if "area_convex" in properties or "convex_hull_area" in properties:
        result["area_convex"] = result.geometry.convex_hull.area

        # Convert to requested units
        if area_unit == "km2":
            result["area_convex"] = result["area_convex"] / 1_000_000
            result.rename(columns={"area_convex": "area_convex_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_convex"] = result["area_convex"] / 10_000
            result.rename(columns={"area_convex": "area_convex_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_convex": "area_convex_m2"}, inplace=True)

        # For backward compatibility
        if "convex_hull_area" in properties and "area_convex" not in properties:
            result["convex_hull_area"] = result["area_convex"]
            if area_unit == "km2":
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_km2"}, inplace=True
                )
            elif area_unit == "ha":
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_ha"}, inplace=True
                )
            else:
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_m2"}, inplace=True
                )

    # Area of filled geometry (no holes)
    if "area_filled" in properties:

        def get_filled_area(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return 0

            if isinstance(geom, MultiPolygon):
                # For MultiPolygon, fill all constituent polygons
                filled_polys = [Polygon(p.exterior) for p in geom.geoms]
                return unary_union(filled_polys).area
            else:
                # For single Polygon, create a new one with just the exterior ring
                return Polygon(geom.exterior).area

        result["area_filled"] = result.geometry.apply(get_filled_area)

        # Convert to requested units
        if area_unit == "km2":
            result["area_filled"] = result["area_filled"] / 1_000_000
            result.rename(columns={"area_filled": "area_filled_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_filled"] = result["area_filled"] / 10_000
            result.rename(columns={"area_filled": "area_filled_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_filled": "area_filled_m2"}, inplace=True)

    # Axes lengths, eccentricity, orientation, and elongation
    if any(
        p in properties
        for p in [
            "major_length",
            "minor_length",
            "eccentricity",
            "orientation",
            "elongation",
        ]
    ):

        def get_axes_properties(geom):
            # Skip non-polygons
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None, None, None, None, None

            # Handle multipolygons by using the largest polygon
            if isinstance(geom, MultiPolygon):
                # Get the polygon with the largest area
                geom = sorted(list(geom.geoms), key=lambda p: p.area, reverse=True)[0]

            try:
                # Get the minimum rotated rectangle
                rect = geom.minimum_rotated_rectangle

                # Extract coordinates
                coords = list(rect.exterior.coords)[
                    :-1
                ]  # Remove the duplicated last point

                if len(coords) < 4:
                    return None, None, None, None, None

                # Calculate lengths of all four sides
                sides = []
                for i in range(len(coords)):
                    p1 = coords[i]
                    p2 = coords[(i + 1) % len(coords)]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    angle = np.degrees(np.arctan2(dy, dx)) % 180
                    sides.append((length, angle, p1, p2))

                # Group sides by length (allowing for small differences due to floating point precision)
                # This ensures we correctly identify the rectangle's dimensions
                sides_grouped = {}
                tolerance = 1e-6  # Tolerance for length comparison

                for s in sides:
                    length, angle = s[0], s[1]
                    matched = False

                    for key in sides_grouped:
                        if abs(length - key) < tolerance:
                            sides_grouped[key].append(s)
                            matched = True
                            break

                    if not matched:
                        sides_grouped[length] = [s]

                # Get unique lengths (should be 2 for a rectangle, parallel sides have equal length)
                unique_lengths = sorted(sides_grouped.keys(), reverse=True)

                if len(unique_lengths) != 2:
                    # If we don't get exactly 2 unique lengths, something is wrong with the rectangle
                    # Fall back to simpler method using bounds
                    bounds = rect.bounds
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    major_length = max(width, height)
                    minor_length = min(width, height)
                    orientation = 0 if width > height else 90
                else:
                    major_length = unique_lengths[0]
                    minor_length = unique_lengths[1]
                    # Get orientation from the major axis
                    orientation = sides_grouped[major_length][0][1]

                # Calculate eccentricity
                if major_length > 0:
                    # Eccentricity for an ellipse: e = sqrt(1 - (b²/a²))
                    # where a is the semi-major axis and b is the semi-minor axis
                    eccentricity = np.sqrt(
                        1 - ((minor_length / 2) ** 2 / (major_length / 2) ** 2)
                    )
                else:
                    eccentricity = 0

                # Calculate elongation (ratio of minor to major axis)
                elongation = major_length / minor_length if major_length > 0 else 1

                return major_length, minor_length, eccentricity, orientation, elongation

            except Exception as e:
                # For debugging
                # print(f"Error calculating axes: {e}")
                return None, None, None, None, None

        # Apply the function and split the results
        axes_data = result.geometry.apply(get_axes_properties)

        if "major_length" in properties:
            result["major_length"] = axes_data.apply(lambda x: x[0] if x else None)
            # Convert to requested units
            if length_unit == "km":
                result["major_length"] = result["major_length"] / 1_000
                result.rename(columns={"major_length": "major_length_km"}, inplace=True)
            else:
                result.rename(columns={"major_length": "major_length_m"}, inplace=True)

        if "minor_length" in properties:
            result["minor_length"] = axes_data.apply(lambda x: x[1] if x else None)
            # Convert to requested units
            if length_unit == "km":
                result["minor_length"] = result["minor_length"] / 1_000
                result.rename(columns={"minor_length": "minor_length_km"}, inplace=True)
            else:
                result.rename(columns={"minor_length": "minor_length_m"}, inplace=True)

        if "eccentricity" in properties:
            result["eccentricity"] = axes_data.apply(lambda x: x[2] if x else None)

        if "orientation" in properties:
            result["orientation"] = axes_data.apply(lambda x: x[3] if x else None)

        if "elongation" in properties:
            result["elongation"] = axes_data.apply(lambda x: x[4] if x else None)

    # Equivalent diameter based on area
    if "diameter_areagth" in properties:

        def get_equivalent_diameter(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None
            # Diameter of a circle with the same area: d = 2 * sqrt(A / π)
            return 2 * np.sqrt(geom.area / np.pi)

        result["diameter_areagth"] = result.geometry.apply(get_equivalent_diameter)

        # Convert to requested units
        if length_unit == "km":
            result["diameter_areagth"] = result["diameter_areagth"] / 1_000
            result.rename(
                columns={"diameter_areagth": "equivalent_diameter_area_km"},
                inplace=True,
            )
        else:
            result.rename(
                columns={"diameter_areagth": "equivalent_diameter_area_m"},
                inplace=True,
            )

    # Extent (ratio of shape area to bounding box area)
    if "extent" in properties:

        def get_extent(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None

            bounds = geom.bounds
            bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

            if bbox_area > 0:
                return geom.area / bbox_area
            return None

        result["extent"] = result.geometry.apply(get_extent)

    # Solidity (ratio of shape area to convex hull area)
    if "solidity" in properties:

        def get_solidity(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None

            convex_hull_area = geom.convex_hull.area

            if convex_hull_area > 0:
                return geom.area / convex_hull_area
            return None

        result["solidity"] = result.geometry.apply(get_solidity)

    # Complexity (ratio of perimeter to area)
    if "complexity" in properties:

        def calc_complexity(geom):
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.area > 0:
                # Shape index: P / (2 * sqrt(π * A))
                # Normalized to 1 for a circle, higher for more complex shapes
                return geom.boundary.length / (2 * np.sqrt(np.pi * geom.area))
            return None

        result["complexity"] = result.geometry.apply(calc_complexity)

    return result


def orthogonalize(
    input_path,
    output_path=None,
    epsilon=0.2,
    min_area=10,
    min_segments=4,
    area_tolerance=0.7,
    detect_triangles=True,
) -> Any:
    """
    Orthogonalizes object masks in a GeoTIFF file.

    This function reads a GeoTIFF containing object masks (binary or labeled regions),
    converts the raster masks to vector polygons, applies orthogonalization to each polygon,
    and optionally writes the result to a GeoJSON file.
    The source code is adapted from the Solar Panel Detection algorithm by Esri.
    See https://www.arcgis.com/home/item.html?id=c2508d72f2614104bfcfd5ccf1429284.
    Credits to Esri for the original code.

    Args:
        input_path (str): Path to the input GeoTIFF file.
        output_path (str, optional): Path to save the output GeoJSON file. If None, no file is saved.
        epsilon (float, optional): Simplification tolerance for the Douglas-Peucker algorithm.
            Higher values result in more simplification. Default is 0.2.
        min_area (float, optional): Minimum area of polygons to process (smaller ones are kept as-is).
        min_segments (int, optional): Minimum number of segments to keep after simplification.
            Default is 4 (for rectangular shapes).
        area_tolerance (float, optional): Allowed ratio of area change. Values less than 1.0 restrict
            area change. Default is 0.7 (allows reduction to 70% of original area).
        detect_triangles (bool, optional): If True, performs additional check to avoid creating triangular shapes.

    Returns:
        Any: A GeoDataFrame containing the orthogonalized features.
    """

    from functools import partial

    def orthogonalize_ring(ring, epsilon=0.2, min_segments=4):
        """
        Orthogonalizes a ring (list of coordinates).

        Args:
            ring (list): List of [x, y] coordinates forming a ring
            epsilon (float, optional): Simplification tolerance
            min_segments (int, optional): Minimum number of segments to keep

        Returns:
            list: Orthogonalized list of coordinates
        """
        if len(ring) <= 3:
            return ring

        # Convert to numpy array
        ring_arr = np.array(ring)

        # Get orientation
        angle = math.degrees(get_orientation(ring_arr))

        # Simplify using Ramer-Douglas-Peucker algorithm
        ring_arr = simplify(ring_arr, eps=epsilon)

        # If simplified too much, adjust epsilon to maintain minimum segments
        if len(ring_arr) < min_segments:
            # Try with smaller epsilon until we get at least min_segments points
            for adjust_factor in [0.75, 0.5, 0.25, 0.1]:
                test_arr = simplify(np.array(ring), eps=epsilon * adjust_factor)
                if len(test_arr) >= min_segments:
                    ring_arr = test_arr
                    break

        # Convert to dataframe for processing
        df = to_dataframe(ring_arr)

        # Add orientation information
        add_orientation(df, angle)

        # Align segments to orthogonal directions
        df = align(df)

        # Merge collinear line segments
        df = merge_lines(df)

        if len(df) == 0:
            return ring

        # If we have a triangle-like result (3 segments or less), return the original shape
        if len(df) <= 3:
            return ring

        # Join the orthogonalized segments back into a ring
        joined_ring = join_ring(df)

        # If the join operation didn't produce a valid ring, return the original
        if len(joined_ring) == 0 or len(joined_ring[0]) < 3:
            return ring

        # Enhanced validation: check for triangular result and geometric validity
        result_coords = joined_ring[0]

        # If result has 3 or fewer points (triangle), use original
        if len(result_coords) <= 3:  # 2 points + closing point (degenerate)
            return ring

        # Additional validation: check for degenerate geometry
        # Calculate area ratio to detect if the shape got severely distorted
        def calculate_polygon_area(coords):
            if len(coords) < 3:
                return 0
            area = 0
            n = len(coords)
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            return abs(area) / 2

        original_area = calculate_polygon_area(ring)
        result_area = calculate_polygon_area(result_coords)

        # If the area changed dramatically (more than 30% shrinkage or 300% growth), use original
        if original_area > 0 and result_area > 0:
            area_ratio = result_area / original_area
            if area_ratio < 0.3 or area_ratio > 3.0:
                return ring

        # Check for triangular spikes and problematic artifacts
        very_acute_angle_count = 0
        triangular_spike_detected = False

        for i in range(len(result_coords) - 1):  # -1 to exclude closing point
            p1 = result_coords[i - 1]
            p2 = result_coords[i]
            p3 = result_coords[(i + 1) % (len(result_coords) - 1)]

            # Calculate angle at p2
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # Count very acute angles (< 20 degrees) - these are likely spikes
                if angle < np.pi / 9:  # 20 degrees
                    very_acute_angle_count += 1
                    # If it's very acute with short sides, it's definitely a spike
                    if v1_norm < 5 or v2_norm < 5:
                        triangular_spike_detected = True

        # Check for excessively long edges that might be artifacts
        edge_lengths = []
        for i in range(len(result_coords) - 1):
            edge_len = np.sqrt(
                (result_coords[i + 1][0] - result_coords[i][0]) ** 2
                + (result_coords[i + 1][1] - result_coords[i][1]) ** 2
            )
            edge_lengths.append(edge_len)

        excessive_edge_detected = False
        if len(edge_lengths) > 0:
            avg_edge_length = np.mean(edge_lengths)
            max_edge_length = np.max(edge_lengths)
            # Only reject if edge is extremely disproportionate (8x average)
            if max_edge_length > avg_edge_length * 8:
                excessive_edge_detected = True

        # Check for triangular artifacts by detecting spikes that extend beyond bounds
        # Calculate original bounds
        orig_xs = [p[0] for p in ring]
        orig_ys = [p[1] for p in ring]
        orig_min_x, orig_max_x = min(orig_xs), max(orig_xs)
        orig_min_y, orig_max_y = min(orig_ys), max(orig_ys)
        orig_width = orig_max_x - orig_min_x
        orig_height = orig_max_y - orig_min_y

        # Calculate result bounds
        result_xs = [p[0] for p in result_coords]
        result_ys = [p[1] for p in result_coords]
        result_min_x, result_max_x = min(result_xs), max(result_xs)
        result_min_y, result_max_y = min(result_ys), max(result_ys)

        # Stricter bounds checking to catch triangular artifacts
        bounds_extension_detected = False
        # More conservative: only allow 10% extension
        tolerance_x = max(orig_width * 0.1, 1.0)  # 10% tolerance, at least 1 unit
        tolerance_y = max(orig_height * 0.1, 1.0)  # 10% tolerance, at least 1 unit

        if (
            result_min_x < orig_min_x - tolerance_x
            or result_max_x > orig_max_x + tolerance_x
            or result_min_y < orig_min_y - tolerance_y
            or result_max_y > orig_max_y + tolerance_y
        ):
            bounds_extension_detected = True

        # Reject if we detect triangular spikes, excessive edges, or bounds violations
        if (
            triangular_spike_detected
            or very_acute_angle_count > 2  # Multiple very acute angles
            or excessive_edge_detected
            or bounds_extension_detected
        ):  # Any significant bounds extension
            return ring

        # Convert back to a list and ensure it's closed
        result = joined_ring[0].tolist()
        if len(result) > 0 and (result[0] != result[-1]):
            result.append(result[0])

        return result

    def vectorize_mask(mask, transform):
        """
        Converts a binary mask to vector polygons.

        Args:
            mask (numpy.ndarray): Binary mask where non-zero values represent objects
            transform (rasterio.transform.Affine): Affine transformation matrix

        Returns:
            list: List of GeoJSON features
        """
        shapes = features.shapes(mask, transform=transform)
        features_list = []

        for shape, value in shapes:
            if value > 0:  # Only process non-zero values (actual objects)
                features_list.append(
                    {
                        "type": "Feature",
                        "properties": {"value": int(value)},
                        "geometry": shape,
                    }
                )

        return features_list

    def rasterize_features(features, shape, transform, dtype=np.uint8):
        """
        Converts vector features back to a raster mask.

        Args:
            features (list): List of GeoJSON features
            shape (tuple): Shape of the output raster (height, width)
            transform (rasterio.transform.Affine): Affine transformation matrix
            dtype (numpy.dtype, optional): Data type of the output raster

        Returns:
            numpy.ndarray: Rasterized mask
        """
        mask = features.rasterize(
            [
                (feature["geometry"], feature["properties"]["value"])
                for feature in features
            ],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=dtype,
        )

        return mask

    # The following helper functions are from the original code
    def get_orientation(contour):
        """
        Calculate the orientation angle of a contour.

        Args:
            contour (numpy.ndarray): Array of shape (n, 2) containing point coordinates

        Returns:
            float: Orientation angle in radians
        """
        box = cv2.minAreaRect(contour.astype(int))
        (cx, cy), (w, h), angle = box
        return math.radians(angle)

    def simplify(contour, eps=0.2):
        """
        Simplify a contour using the Ramer-Douglas-Peucker algorithm.

        Args:
            contour (numpy.ndarray): Array of shape (n, 2) containing point coordinates
            eps (float, optional): Epsilon value for simplification

        Returns:
            numpy.ndarray: Simplified contour
        """
        return rdp(contour, epsilon=eps)

    def to_dataframe(ring):
        """
        Convert a ring to a pandas DataFrame with line segment information.

        Args:
            ring (numpy.ndarray): Array of shape (n, 2) containing point coordinates

        Returns:
            pandas.DataFrame: DataFrame with line segment information
        """
        df = pd.DataFrame(ring, columns=["x1", "y1"])
        df["x2"] = df["x1"].shift(-1)
        df["y2"] = df["y1"].shift(-1)
        df.dropna(inplace=True)
        df["angle_atan"] = np.arctan2((df["y2"] - df["y1"]), (df["x2"] - df["x1"]))
        df["angle_atan_deg"] = df["angle_atan"] * 57.2958
        df["len"] = np.sqrt((df["y2"] - df["y1"]) ** 2 + (df["x2"] - df["x1"]) ** 2)
        df["cx"] = (df["x2"] + df["x1"]) / 2.0
        df["cy"] = (df["y2"] + df["y1"]) / 2.0
        return df

    def add_orientation(df, angle):
        """
        Add orientation information to the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with line segment information
            angle (float): Orientation angle in degrees

        Returns:
            None: Modifies the DataFrame in-place
        """
        rtangle = angle + 90
        is_parallel = (
            (df["angle_atan_deg"] > (angle - 45))
            & (df["angle_atan_deg"] < (angle + 45))
        ) | (
            (df["angle_atan_deg"] + 180 > (angle - 45))
            & (df["angle_atan_deg"] + 180 < (angle + 45))
        )
        df["angle"] = math.radians(angle)
        df["angle"] = df["angle"].where(is_parallel, math.radians(rtangle))

    def align(df):
        """
        Align line segments to their nearest orthogonal direction.

        Args:
            df (pandas.DataFrame): DataFrame with line segment information

        Returns:
            pandas.DataFrame: DataFrame with aligned line segments
        """
        # Handle edge case with empty dataframe
        if len(df) == 0:
            return df.copy()

        df_clone = df.copy()

        # Ensure angle column exists and has valid values
        if "angle" not in df_clone.columns or df_clone["angle"].isna().any():
            # If angle data is missing, add default angles based on atan2
            df_clone["angle"] = df_clone["angle_atan"]

        # Ensure length and center point data is valid
        if "len" not in df_clone.columns or df_clone["len"].isna().any():
            # Recalculate lengths if missing
            df_clone["len"] = np.sqrt(
                (df_clone["x2"] - df_clone["x1"]) ** 2
                + (df_clone["y2"] - df_clone["y1"]) ** 2
            )

        if "cx" not in df_clone.columns or df_clone["cx"].isna().any():
            df_clone["cx"] = (df_clone["x1"] + df_clone["x2"]) / 2.0

        if "cy" not in df_clone.columns or df_clone["cy"].isna().any():
            df_clone["cy"] = (df_clone["y1"] + df_clone["y2"]) / 2.0

        # Apply orthogonal alignment
        df_clone["x1"] = df_clone["cx"] - ((df_clone["len"] / 2) * np.cos(df["angle"]))
        df_clone["x2"] = df_clone["cx"] + ((df_clone["len"] / 2) * np.cos(df["angle"]))
        df_clone["y1"] = df_clone["cy"] - ((df_clone["len"] / 2) * np.sin(df["angle"]))
        df_clone["y2"] = df_clone["cy"] + ((df_clone["len"] / 2) * np.sin(df["angle"]))

        return df_clone

    def merge_lines(df_aligned):
        """
        Merge collinear line segments.

        Args:
            df_aligned (pandas.DataFrame): DataFrame with aligned line segments

        Returns:
            pandas.DataFrame: DataFrame with merged line segments
        """
        ortho_lines = []
        groups = df_aligned.groupby(
            (df_aligned["angle"].shift() != df_aligned["angle"]).cumsum()
        )
        for x, y in groups:
            group_cx = (y["cx"] * y["len"]).sum() / y["len"].sum()
            group_cy = (y["cy"] * y["len"]).sum() / y["len"].sum()
            cumlen = y["len"].sum()

            ortho_lines.append((group_cx, group_cy, cumlen, y["angle"].iloc[0]))

        ortho_list = []
        for cx, cy, length, rot_angle in ortho_lines:
            X1 = cx - (length / 2) * math.cos(rot_angle)
            X2 = cx + (length / 2) * math.cos(rot_angle)
            Y1 = cy - (length / 2) * math.sin(rot_angle)
            Y2 = cy + (length / 2) * math.sin(rot_angle)

            ortho_list.append(
                {
                    "x1": X1,
                    "y1": Y1,
                    "x2": X2,
                    "y2": Y2,
                    "len": length,
                    "cx": cx,
                    "cy": cy,
                    "angle": rot_angle,
                }
            )

        # Improved fix: Prevent merging that would create triangular or problematic shapes
        if (
            len(ortho_list) > 3 and ortho_list[0]["angle"] == ortho_list[-1]["angle"]
        ):  # join first and last segment if they're in same direction
            # Check if merging would result in 3 or 4 segments (potentially triangular)
            resulting_segments = len(ortho_list) - 1
            if resulting_segments <= 4:
                # For very small polygons, be extra cautious about merging
                # Calculate the spatial relationship between first and last segments
                first_center = np.array([ortho_list[0]["cx"], ortho_list[0]["cy"]])
                last_center = np.array([ortho_list[-1]["cx"], ortho_list[-1]["cy"]])
                center_distance = np.linalg.norm(first_center - last_center)

                # Get average segment length for comparison
                avg_length = sum(seg["len"] for seg in ortho_list) / len(ortho_list)

                # Only merge if segments are close enough and it won't create degenerate shapes
                if center_distance > avg_length * 1.5:
                    # Skip merging - segments are too far apart
                    pass
                else:
                    # Proceed with merging only for well-connected segments
                    totlen = ortho_list[0]["len"] + ortho_list[-1]["len"]
                    merge_cx = (
                        (ortho_list[0]["cx"] * ortho_list[0]["len"])
                        + (ortho_list[-1]["cx"] * ortho_list[-1]["len"])
                    ) / totlen

                    merge_cy = (
                        (ortho_list[0]["cy"] * ortho_list[0]["len"])
                        + (ortho_list[-1]["cy"] * ortho_list[-1]["len"])
                    ) / totlen

                    rot_angle = ortho_list[0]["angle"]
                    X1 = merge_cx - (totlen / 2) * math.cos(rot_angle)
                    X2 = merge_cx + (totlen / 2) * math.cos(rot_angle)
                    Y1 = merge_cy - (totlen / 2) * math.sin(rot_angle)
                    Y2 = merge_cy + (totlen / 2) * math.sin(rot_angle)

                    ortho_list[-1] = {
                        "x1": X1,
                        "y1": Y1,
                        "x2": X2,
                        "y2": Y2,
                        "len": totlen,
                        "cx": merge_cx,
                        "cy": merge_cy,
                        "angle": rot_angle,
                    }
                    ortho_list = ortho_list[1:]
            else:
                # For larger polygons, proceed with standard merging
                totlen = ortho_list[0]["len"] + ortho_list[-1]["len"]
                merge_cx = (
                    (ortho_list[0]["cx"] * ortho_list[0]["len"])
                    + (ortho_list[-1]["cx"] * ortho_list[-1]["len"])
                ) / totlen

                merge_cy = (
                    (ortho_list[0]["cy"] * ortho_list[0]["len"])
                    + (ortho_list[-1]["cy"] * ortho_list[-1]["len"])
                ) / totlen

                rot_angle = ortho_list[0]["angle"]
                X1 = merge_cx - (totlen / 2) * math.cos(rot_angle)
                X2 = merge_cx + (totlen / 2) * math.cos(rot_angle)
                Y1 = merge_cy - (totlen / 2) * math.sin(rot_angle)
                Y2 = merge_cy + (totlen / 2) * math.sin(rot_angle)

                ortho_list[-1] = {
                    "x1": X1,
                    "y1": Y1,
                    "x2": X2,
                    "y2": Y2,
                    "len": totlen,
                    "cx": merge_cx,
                    "cy": merge_cy,
                    "angle": rot_angle,
                }
                ortho_list = ortho_list[1:]
        ortho_df = pd.DataFrame(ortho_list)
        return ortho_df

    def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Find the intersection point of two line segments.

        Args:
            x1, y1, x2, y2: Coordinates of the first line segment
            x3, y3, x4, y4: Coordinates of the second line segment

        Returns:
            list: [x, y] coordinates of the intersection point

        Raises:
            ZeroDivisionError: If the lines are parallel or collinear
        """
        # Calculate the denominator of the intersection formula
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Check if lines are parallel or collinear (denominator close to zero)
        if abs(denominator) < 1e-10:
            raise ZeroDivisionError("Lines are parallel or collinear")

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator

        # Check if the intersection point is within a reasonable distance
        # from both line segments to avoid extreme extrapolation
        def point_on_segment(x, y, x1, y1, x2, y2, tolerance=2.0):
            # Check if point (x,y) is near the line segment from (x1,y1) to (x2,y2)
            # First check if it's near the infinite line
            line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_len < 1e-10:
                return np.sqrt((x - x1) ** 2 + (y - y1) ** 2) <= tolerance

            t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_len**2)

            # Check distance to the infinite line
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            dist_to_line = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

            # Check if the projection is near the segment, not just the infinite line
            if t < -tolerance or t > 1 + tolerance:
                # If far from the segment, compute distance to the nearest endpoint
                dist_to_start = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                dist_to_end = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                return min(dist_to_start, dist_to_end) <= tolerance * 2

            return dist_to_line <= tolerance

        # Check if intersection is reasonably close to both line segments
        if not (
            point_on_segment(px, py, x1, y1, x2, y2)
            and point_on_segment(px, py, x3, y3, x4, y4)
        ):
            # If intersection is far from segments, it's probably extrapolating too much
            raise ValueError("Intersection point too far from line segments")

        return [px, py]

    def join_ring(merged_df):
        """
        Join line segments to form a closed ring.

        Args:
            merged_df (pandas.DataFrame): DataFrame with merged line segments

        Returns:
            numpy.ndarray: Array of shape (1, n, 2) containing the ring coordinates
        """
        # Handle edge cases
        if len(merged_df) < 3:
            # Not enough segments to form a valid polygon
            return np.array([[]])

        ring = []

        # Find intersections between adjacent line segments
        for i in range(len(merged_df) - 1):
            x1, y1, x2, y2, *_ = merged_df.iloc[i]
            x3, y3, x4, y4, *_ = merged_df.iloc[i + 1]

            try:
                intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)

                # Check if the intersection point is too far from either line segment
                # This helps prevent extending edges beyond reasonable bounds
                dist_to_seg1 = min(
                    np.sqrt((intersection[0] - x1) ** 2 + (intersection[1] - y1) ** 2),
                    np.sqrt((intersection[0] - x2) ** 2 + (intersection[1] - y2) ** 2),
                )
                dist_to_seg2 = min(
                    np.sqrt((intersection[0] - x3) ** 2 + (intersection[1] - y3) ** 2),
                    np.sqrt((intersection[0] - x4) ** 2 + (intersection[1] - y4) ** 2),
                )

                # Use the maximum of line segment lengths as a reference
                max_len = max(
                    np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                    np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2),
                )

                # Improved intersection validation
                # Calculate angle between segments to detect sharp corners
                v1 = np.array([x2 - x1, y2 - y1])
                v2 = np.array([x4 - x3, y4 - y3])
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)

                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)

                    # Check for very sharp angles that could create triangular artifacts
                    is_sharp_angle = (
                        angle < np.pi / 6 or angle > 5 * np.pi / 6
                    )  # <30° or >150°
                else:
                    is_sharp_angle = False

                # Determine whether to use intersection or segment endpoint
                if (
                    dist_to_seg1 > max_len * 0.5
                    or dist_to_seg2 > max_len * 0.5
                    or is_sharp_angle
                ):
                    # Use a more conservative approach for problematic intersections
                    # Use the closer endpoint between segments
                    dist_x2_to_seg2 = min(
                        np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                        np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
                    )
                    dist_x3_to_seg1 = min(
                        np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2),
                        np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2),
                    )

                    if dist_x2_to_seg2 <= dist_x3_to_seg1:
                        ring.append([x2, y2])
                    else:
                        ring.append([x3, y3])
                else:
                    ring.append(intersection)
            except Exception:
                # If intersection calculation fails, use the endpoint of the first segment
                ring.append([x2, y2])

        # Connect last segment with first segment
        x1, y1, x2, y2, *_ = merged_df.iloc[-1]
        x3, y3, x4, y4, *_ = merged_df.iloc[0]

        try:
            intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)

            # Check if the intersection point is too far from either line segment
            dist_to_seg1 = min(
                np.sqrt((intersection[0] - x1) ** 2 + (intersection[1] - y1) ** 2),
                np.sqrt((intersection[0] - x2) ** 2 + (intersection[1] - y2) ** 2),
            )
            dist_to_seg2 = min(
                np.sqrt((intersection[0] - x3) ** 2 + (intersection[1] - y3) ** 2),
                np.sqrt((intersection[0] - x4) ** 2 + (intersection[1] - y4) ** 2),
            )

            max_len = max(
                np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2),
            )

            # Apply same sharp angle detection for closing segment
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x4 - x3, y4 - y3])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                is_sharp_angle = angle < np.pi / 6 or angle > 5 * np.pi / 6
            else:
                is_sharp_angle = False

            if (
                dist_to_seg1 > max_len * 0.5
                or dist_to_seg2 > max_len * 0.5
                or is_sharp_angle
            ):
                # Use conservative approach for closing segment
                dist_x2_to_seg2 = min(
                    np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                    np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
                )
                dist_x3_to_seg1 = min(
                    np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2),
                    np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2),
                )

                if dist_x2_to_seg2 <= dist_x3_to_seg1:
                    ring.append([x2, y2])
                else:
                    ring.append([x3, y3])
            else:
                ring.append(intersection)
        except Exception:
            # If intersection calculation fails, use the endpoint of the last segment
            ring.append([x2, y2])

        # Ensure the ring is closed
        if len(ring) > 0 and (ring[0][0] != ring[-1][0] or ring[0][1] != ring[-1][1]):
            ring.append(ring[0])

        return np.array([ring])

    def rdp(M, epsilon=0, dist=None, algo="iter", return_mask=False):
        """
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float, optional): Epsilon value for simplification
            dist (callable, optional): Distance function
            algo (str, optional): Algorithm to use ('iter' or 'rec')
            return_mask (bool, optional): Whether to return a mask instead of the simplified array

        Returns:
            numpy.ndarray or list: Simplified points or mask
        """
        if dist is None:
            dist = pldist

        if algo == "iter":
            algo = partial(rdp_iter, return_mask=return_mask)
        elif algo == "rec":
            if return_mask:
                raise NotImplementedError(
                    'return_mask=True not supported with algo="rec"'
                )
            algo = rdp_rec

        if "numpy" in str(type(M)):
            return algo(M, epsilon, dist)

        return algo(np.array(M), epsilon, dist).tolist()

    def pldist(point, start, end):
        """
        Calculates the distance from 'point' to the line given by 'start' and 'end'.

        Args:
            point (numpy.ndarray): Point coordinates
            start (numpy.ndarray): Start point of the line
            end (numpy.ndarray): End point of the line

        Returns:
            float: Distance from point to line
        """
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)

        # Fix for NumPy 2.0 deprecation warning - handle 2D vectors properly
        # Instead of using cross product directly, calculate the area of the
        # parallelogram formed by the vectors and divide by the length of the line
        line_vec = end - start
        point_vec = point - start

        # Area of parallelogram = |a|*|b|*sin(θ)
        # For 2D vectors: |a×b| = |a|*|b|*sin(θ) = determinant([ax, ay], [bx, by])
        area = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0])

        # Distance = Area / |line_vec|
        return area / np.linalg.norm(line_vec)

    def rdp_rec(M, epsilon, dist=pldist):
        """
        Recursive implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function

        Returns:
            numpy.ndarray: Simplified points
        """
        dmax = 0.0
        index = -1

        for i in range(1, M.shape[0]):
            d = dist(M[i], M[0], M[-1])

            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            r1 = rdp_rec(M[: index + 1], epsilon, dist)
            r2 = rdp_rec(M[index:], epsilon, dist)

            return np.vstack((r1[:-1], r2))
        else:
            return np.vstack((M[0], M[-1]))

    def _rdp_iter(M, start_index, last_index, epsilon, dist=pldist):
        """
        Internal iterative implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            start_index (int): Start index
            last_index (int): Last index
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function

        Returns:
            numpy.ndarray: Boolean mask of points to keep
        """
        stk = []
        stk.append([start_index, last_index])
        global_start_index = start_index
        indices = np.ones(last_index - start_index + 1, dtype=bool)

        while stk:
            start_index, last_index = stk.pop()

            dmax = 0.0
            index = start_index

            for i in range(index + 1, last_index):
                if indices[i - global_start_index]:
                    d = dist(M[i], M[start_index], M[last_index])
                    if d > dmax:
                        index = i
                        dmax = d

            if dmax > epsilon:
                stk.append([start_index, index])
                stk.append([index, last_index])
            else:
                for i in range(start_index + 1, last_index):
                    indices[i - global_start_index] = False

        return indices

    def rdp_iter(M, epsilon, dist=pldist, return_mask=False):
        """
        Iterative implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function
            return_mask (bool, optional): Whether to return a mask instead of the simplified array

        Returns:
            numpy.ndarray: Simplified points or boolean mask
        """
        mask = _rdp_iter(M, 0, len(M) - 1, epsilon, dist)

        if return_mask:
            return mask

        return M[mask]

    # Read the raster data
    with rasterio.open(input_path) as src:
        # Read the first band (assuming it contains the mask)
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        # Extract shapes from the raster mask
        shapes = list(features.shapes(mask, transform=transform))

        # Initialize progress bar
        print(f"Processing {len(shapes)} features...")

        # Convert shapes to GeoJSON features
        features_list = []
        for shape, value in tqdm(shapes, desc="Converting features", unit="shape"):
            if value > 0:  # Only process non-zero values (actual objects)
                # Convert GeoJSON geometry to Shapely polygon
                polygon = Polygon(shape["coordinates"][0])

                # Skip tiny polygons
                if polygon.area < min_area:
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": shape,
                        }
                    )
                    continue

                # Check if shape is triangular and if we want to avoid triangular shapes
                if detect_triangles:
                    # Create a simplified version to check number of vertices
                    simple_polygon = polygon.simplify(epsilon)
                    if (
                        len(simple_polygon.exterior.coords) <= 4
                    ):  # 3 points + closing point
                        # Likely a triangular shape - skip orthogonalization
                        features_list.append(
                            {
                                "type": "Feature",
                                "properties": {"value": int(value)},
                                "geometry": shape,
                            }
                        )
                        continue

                # Process larger, non-triangular polygons
                try:
                    # Convert shapely polygon to a ring format for orthogonalization
                    exterior_ring = list(polygon.exterior.coords)
                    interior_rings = [
                        list(interior.coords) for interior in polygon.interiors
                    ]

                    # Calculate bounding box aspect ratio to help with parameter tuning
                    minx, miny, maxx, maxy = polygon.bounds
                    width = maxx - minx
                    height = maxy - miny
                    aspect_ratio = max(width, height) / max(1.0, min(width, height))

                    # Determine if this shape is likely to be a building/rectangular object
                    # Long thin objects might require different treatment
                    is_rectangular = aspect_ratio < 3.0

                    # Rectangular objects usually need more careful orthogonalization
                    epsilon_adjusted = epsilon
                    min_segments_adjusted = min_segments

                    if is_rectangular:
                        # For rectangular objects, use more conservative epsilon
                        epsilon_adjusted = epsilon * 0.75
                        # Ensure we get at least 4 points for a proper rectangle
                        min_segments_adjusted = max(4, min_segments)

                    # Orthogonalize the exterior and interior rings
                    orthogonalized_exterior = orthogonalize_ring(
                        exterior_ring,
                        epsilon=epsilon_adjusted,
                        min_segments=min_segments_adjusted,
                    )

                    orthogonalized_interiors = [
                        orthogonalize_ring(
                            ring,
                            epsilon=epsilon_adjusted,
                            min_segments=min_segments_adjusted,
                        )
                        for ring in interior_rings
                    ]

                    # Validate the result - calculate area change
                    original_area = polygon.area
                    orthogonalized_poly = Polygon(orthogonalized_exterior)

                    if orthogonalized_poly.is_valid:
                        area_ratio = (
                            orthogonalized_poly.area / original_area
                            if original_area > 0
                            else 0
                        )

                        # If area changed too much, revert to original
                        if area_ratio < area_tolerance or area_ratio > (
                            1.0 / area_tolerance
                        ):
                            # Use original polygon instead
                            geometry = shape
                        else:
                            # Create a new geometry with orthogonalized rings
                            geometry = {
                                "type": "Polygon",
                                "coordinates": [orthogonalized_exterior],
                            }

                            # Add interior rings if they exist
                            if orthogonalized_interiors:
                                geometry["coordinates"].extend(
                                    [ring for ring in orthogonalized_interiors]
                                )
                    else:
                        # If resulting polygon is invalid, use original
                        geometry = shape

                    # Add the feature to the list
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": geometry,
                        }
                    )
                except Exception as e:
                    # Keep the original shape if orthogonalization fails
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": shape,
                        }
                    )

        # Create the final GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": features_list,
        }

        # Convert to GeoDataFrame and set the CRS
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=crs)

        # Save to file if output_path is provided
        if output_path:
            print(f"Saving to {output_path}...")
            gdf.to_file(output_path)
            print("Done!")

        return gdf


def inspect_pth_file(pth_path: str) -> Dict[str, Any]:
    """
    Inspect a PyTorch .pth model file to determine its architecture.

    Args:
        pth_path: Path to the .pth file to inspect

    Returns:
        Information about the model architecture
    """
    # Check if file exists
    if not os.path.exists(pth_path):
        print(f"Error: File {pth_path} not found")
        return

    # Load the checkpoint
    try:
        checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))
        print(f"\n{'='*50}")
        print(f"Inspecting model file: {pth_path}")
        print(f"{'='*50}\n")

        # Check if it's a state_dict or a complete model
        if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                print("Found 'state_dict' key in the checkpoint.")
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                print("Found 'model_state_dict' key in the checkpoint.")
                state_dict = checkpoint["model_state_dict"]
            else:
                print("Assuming file contains a direct state_dict.")
                state_dict = checkpoint

            # Print the keys in the checkpoint
            print("\nCheckpoint contains the following keys:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"- {key} (dictionary with {len(checkpoint[key])} items)")
                elif isinstance(checkpoint[key], (torch.Tensor, list, tuple)):
                    print(
                        f"- {key} (shape/size: {len(checkpoint[key]) if isinstance(checkpoint[key], (list, tuple)) else checkpoint[key].shape})"
                    )
                else:
                    print(f"- {key} ({type(checkpoint[key]).__name__})")

            # Try to infer the model architecture from the state_dict keys
            print("\nAnalyzing model architecture from state_dict...")

            # Extract layer keys for analysis
            layer_keys = list(state_dict.keys())

            # Print the first few layer keys to understand naming pattern
            print("\nFirst 10 layer names in state_dict:")
            for i, key in enumerate(layer_keys[:10]):
                shape = state_dict[key].shape
                print(f"- {key} (shape: {shape})")

            # Look for architecture indicators in the keys
            architecture_indicators = {
                "conv": 0,
                "bn": 0,
                "layer": 0,
                "fc": 0,
                "backbone": 0,
                "encoder": 0,
                "decoder": 0,
                "unet": 0,
                "resnet": 0,
                "classifier": 0,
                "deeplab": 0,
                "fcn": 0,
            }

            for key in layer_keys:
                for indicator in architecture_indicators:
                    if indicator in key.lower():
                        architecture_indicators[indicator] += 1

            print("\nArchitecture indicators found in layer names:")
            for indicator, count in architecture_indicators.items():
                if count > 0:
                    print(f"- '{indicator}' appears {count} times")

            # Count total parameters
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"\nTotal parameters: {total_params:,}")

            # Try to load the model with different architectures
            print("\nAttempting to match with common architectures...")

            # Try to identify if it's a segmentation model
            if any("out" in k or "classifier" in k for k in layer_keys):
                print("Model appears to be a segmentation model.")

                # Check if it might be a UNet
                if (
                    architecture_indicators["encoder"] > 0
                    and architecture_indicators["decoder"] > 0
                ):
                    print(
                        "Architecture seems to be a UNet-based model with encoder-decoder structure."
                    )
                # Check for FCN or DeepLab indicators
                elif architecture_indicators["fcn"] > 0:
                    print(
                        "Architecture seems to be FCN-based (Fully Convolutional Network)."
                    )
                elif architecture_indicators["deeplab"] > 0:
                    print("Architecture seems to be DeepLab-based.")
                elif architecture_indicators["backbone"] > 0:
                    print(
                        "Model has a backbone architecture, likely a modern segmentation model."
                    )

            # Try to infer output classes from the final layer
            output_layer_keys = [
                k for k in layer_keys if "classifier" in k or k.endswith(".out.weight")
            ]
            if output_layer_keys:
                output_shape = state_dict[output_layer_keys[0]].shape
                if len(output_shape) >= 2:
                    num_classes = output_shape[0]
                    print(f"\nModel likely has {num_classes} output classes.")

            print("\nSUMMARY:")
            print("The model appears to be", end=" ")
            if architecture_indicators["unet"] > 0:
                print("a UNet architecture.", end=" ")
            elif architecture_indicators["fcn"] > 0:
                print("an FCN architecture.", end=" ")
            elif architecture_indicators["deeplab"] > 0:
                print("a DeepLab architecture.", end=" ")
            elif architecture_indicators["resnet"] > 0:
                print("ResNet-based.", end=" ")
            else:
                print("a custom architecture.", end=" ")

            # Try to load with common models
            try_common_architectures(state_dict)

        else:
            print(
                "The file contains an entire model object rather than just a state dictionary."
            )
            # If it's a complete model, we can directly examine its architecture
            print(checkpoint)

    except Exception as e:
        print(f"Error loading the model file: {str(e)}")


def try_common_architectures(state_dict: Dict[str, Any]) -> Optional[str]:
    """
    Try to load the state_dict into common architectures to see which one fits.

    Args:
        state_dict: The model's state dictionary
    """
    import torchinfo

    # Test models and their initializations
    models_to_try = {
        "FCN-ResNet50": lambda: fcn_resnet50(num_classes=9),
        "DeepLabV3-ResNet50": lambda: deeplabv3_resnet50(num_classes=9),
    }

    print("\nTrying to load state_dict into common architectures:")

    for name, model_fn in models_to_try.items():
        try:
            model = model_fn()
            # Sometimes state_dict keys have 'model.' prefix
            if all(k.startswith("model.") for k in state_dict.keys()):
                cleaned_state_dict = {k[6:]: v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            print(
                f"- {name}: Successfully loaded (may have missing or unexpected keys)"
            )

            # Generate model summary
            print(f"\nSummary of {name} architecture:")
            summary = torchinfo.summary(model, input_size=(1, 3, 224, 224), verbose=0)
            print(summary)

        except Exception as e:
            print(f"- {name}: Failed to load - {str(e)}")


def mosaic_geotiffs(
    input_dir: str, output_file: str, mask_file: Optional[str] = None
) -> None:
    """Create a mosaic from all GeoTIFF files as a Cloud Optimized GeoTIFF (COG).

    This function identifies all GeoTIFF files in the specified directory,
    creates a seamless mosaic with proper handling of nodata values, and saves
    as a Cloud Optimized GeoTIFF format. If a mask file is provided, the output
    will be clipped to the extent of the mask.

    Args:
        input_dir (str): Path to the directory containing GeoTIFF files.
        output_file (str): Path to the output Cloud Optimized GeoTIFF file.
        mask_file (str, optional): Path to a mask file to clip the output.
            If provided, the output will be clipped to the extent of this mask.
            Defaults to None.

    Returns:
        bool: True if the mosaic was created successfully, False otherwise.

    Examples:
        >>> mosaic_geotiffs('naip', 'merged_naip.tif')
        True
        >>> mosaic_geotiffs('naip', 'merged_naip.tif', 'boundary.tif')
        True
    """
    import glob

    from osgeo import gdal

    gdal.UseExceptions()
    # Get all tif files in the directory
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

    if not tif_files:
        print("No GeoTIFF files found in the specified directory.")
        return False

    # Analyze the first input file to determine compression and nodata settings
    ds = gdal.Open(tif_files[0])
    if ds is None:
        print(f"Unable to open {tif_files[0]}")
        return False

    # Get driver metadata from the first file
    driver = ds.GetDriver()
    creation_options = []

    # Check compression type
    metadata = ds.GetMetadata("IMAGE_STRUCTURE")
    if "COMPRESSION" in metadata:
        compression = metadata["COMPRESSION"]
        creation_options.append(f"COMPRESS={compression}")
    else:
        # Default compression if none detected
        creation_options.append("COMPRESS=LZW")

    # Add COG-specific creation options
    creation_options.extend(["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])

    # Check for nodata value in the first band of the first file
    band = ds.GetRasterBand(1)
    has_nodata = band.GetNoDataValue() is not None
    nodata_value = band.GetNoDataValue() if has_nodata else None

    # Close the dataset
    ds = None

    # Create a temporary VRT (Virtual Dataset)
    vrt_path = os.path.join(input_dir, "temp_mosaic.vrt")

    # Build VRT from input files with proper nodata handling
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg="nearest",
        srcNodata=nodata_value if has_nodata else None,
        VRTNodata=nodata_value if has_nodata else None,
    )
    vrt_dataset = gdal.BuildVRT(vrt_path, tif_files, options=vrt_options)

    # Close the VRT dataset to flush it to disk
    vrt_dataset = None

    # Create temp mosaic
    temp_mosaic = output_file + ".temp.tif"

    # Convert VRT to GeoTIFF with the same compression as input
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=creation_options,
        noData=nodata_value if has_nodata else None,
    )
    gdal.Translate(temp_mosaic, vrt_path, options=translate_options)

    # Apply mask if provided
    if mask_file and os.path.exists(mask_file):
        print(f"Clipping mosaic to mask: {mask_file}")

        # Create a temporary clipped file
        clipped_mosaic = output_file + ".clipped.tif"

        # Open mask file
        mask_ds = gdal.Open(mask_file)
        if mask_ds is None:
            print(f"Unable to open mask file: {mask_file}")
            # Continue without clipping
        else:
            # Get mask extent
            mask_geotransform = mask_ds.GetGeoTransform()
            mask_projection = mask_ds.GetProjection()
            mask_ulx = mask_geotransform[0]
            mask_uly = mask_geotransform[3]
            mask_lrx = mask_ulx + (mask_geotransform[1] * mask_ds.RasterXSize)
            mask_lry = mask_uly + (mask_geotransform[5] * mask_ds.RasterYSize)

            # Close mask dataset
            mask_ds = None

            # Use warp options to clip
            warp_options = gdal.WarpOptions(
                format="GTiff",
                outputBounds=[mask_ulx, mask_lry, mask_lrx, mask_uly],
                dstSRS=mask_projection,
                creationOptions=creation_options,
                srcNodata=nodata_value if has_nodata else None,
                dstNodata=nodata_value if has_nodata else None,
            )

            # Apply clipping
            gdal.Warp(clipped_mosaic, temp_mosaic, options=warp_options)

            # Remove the unclipped temp mosaic and use the clipped one
            os.remove(temp_mosaic)
            temp_mosaic = clipped_mosaic

    # Create internal overviews for the temp mosaic
    ds = gdal.Open(temp_mosaic, gdal.GA_Update)
    overview_list = [2, 4, 8, 16, 32]
    ds.BuildOverviews("NEAREST", overview_list)
    ds = None  # Close the dataset to ensure overviews are written

    # Convert the temp mosaic to a proper COG
    cog_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COPY_SRC_OVERVIEWS=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ],
        noData=nodata_value if has_nodata else None,
    )
    gdal.Translate(output_file, temp_mosaic, options=cog_options)

    # Clean up temporary files
    if os.path.exists(vrt_path):
        os.remove(vrt_path)
    if os.path.exists(temp_mosaic):
        os.remove(temp_mosaic)

    print(f"Cloud Optimized GeoTIFF mosaic created successfully: {output_file}")
    return True


def download_model_from_hf(model_path: str, repo_id: Optional[str] = None) -> str:
    """
    Download the object detection model from Hugging Face.

    Args:
        model_path: Path to the model file.
        repo_id: Hugging Face repository ID.

    Returns:
        Path to the downloaded model file
    """
    from huggingface_hub import hf_hub_download

    try:

        # Define the repository ID and model filename
        if repo_id is None:
            print(
                "Repo is not specified, using default Hugging Face repo_id: giswqs/geoai"
            )
            repo_id = "giswqs/geoai"

        # Download the model
        model_path = hf_hub_download(repo_id=repo_id, filename=model_path)
        print(f"Model downloaded to: {model_path}")

        return model_path

    except Exception as e:
        print(f"Error downloading model from Hugging Face: {e}")
        print("Please specify a local model path or ensure internet connectivity.")
        raise


def regularize(
    data: Union[gpd.GeoDataFrame, str],
    parallel_threshold: float = 1.0,
    target_crs: Optional[Union[str, "pyproj.CRS"]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 1,
    include_metadata: bool = False,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Regularizes polygon geometries in a GeoDataFrame by aligning edges.

    Aligns edges to be parallel or perpendicular (optionally also 45 degrees)
    to their main direction. Handles reprojection, initial simplification,
    regularization, geometry cleanup, and parallel processing.

    This function is a wrapper around the `regularize_geodataframe` function
    from the `buildingregulariser` package. Credits to the original author
    Nick Wright. Check out the repo at https://github.com/DPIRD-DMA/Building-Regulariser.

    Args:
        data (Union[gpd.GeoDataFrame, str]): Input GeoDataFrame with polygon or multipolygon geometries,
            or a file path to the GeoDataFrame.
        parallel_threshold (float, optional): Distance threshold for merging nearly parallel adjacent edges
            during regularization. Defaults to 1.0.
        target_crs (Optional[Union[str, "pyproj.CRS"]], optional): Target Coordinate Reference System for
            processing. If None, uses the input GeoDataFrame's CRS. Processing is more reliable in a
            projected CRS. Defaults to None.
        simplify (bool, optional): If True, applies initial simplification to the geometry before
            regularization. Defaults to True.
        simplify_tolerance (float, optional): Tolerance for the initial simplification step (if `simplify`
            is True). Also used for geometry cleanup steps. Defaults to 0.5.
        allow_45_degree (bool, optional): If True, allows edges to be oriented at 45-degree angles relative
            to the main direction during regularization. Defaults to True.
        diagonal_threshold_reduction (float, optional): Reduction factor in degrees to reduce the likelihood
            of diagonal edges being created. Larger values reduce the likelihood of diagonal edges.
            Defaults to 15.
        allow_circles (bool, optional): If True, attempts to detect polygons that are nearly circular and
            replaces them with perfect circles. Defaults to True.
        circle_threshold (float, optional): Intersection over Union (IoU) threshold used for circle detection
            (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
        num_cores (int, optional): Number of CPU cores to use for parallel processing. If 1, processing is
            done sequentially. Defaults to 1.
        include_metadata (bool, optional): If True, includes metadata about the regularization process in the
            output GeoDataFrame. Defaults to False.
        output_path (Optional[str], optional): Path to save the output GeoDataFrame. If None, the output is
            not saved. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `to_file` method when saving the output.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with regularized polygon geometries. Original attributes are
        preserved. Geometries that failed processing might be dropped.

    Raises:
        ValueError: If the input data is not a GeoDataFrame or a file path, or if the input GeoDataFrame is empty.
    """
    try:
        from buildingregulariser import regularize_geodataframe
    except ImportError:
        install_package("buildingregulariser")
        from buildingregulariser import regularize_geodataframe

    if isinstance(data, str):
        data = gpd.read_file(data)
    elif not isinstance(data, gpd.GeoDataFrame):
        raise ValueError("Input data must be a GeoDataFrame or a file path.")

    # Check if the input data is empty
    if data.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    gdf = regularize_geodataframe(
        data,
        parallel_threshold=parallel_threshold,
        target_crs=target_crs,
        simplify=simplify,
        simplify_tolerance=simplify_tolerance,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        num_cores=num_cores,
        include_metadata=include_metadata,
    )

    if output_path:
        gdf.to_file(output_path, **kwargs)

    return gdf


def vector_to_geojson(
    filename: str, output: Optional[str] = None, **kwargs: Any
) -> str:
    """Converts a vector file to a geojson file.

    Args:
        filename (str): The vector file path.
        output (str, optional): The output geojson file path. Defaults to None.

    Returns:
        dict: The geojson dictionary.
    """

    if filename.startswith("http"):
        filename = download_file(filename)

    gdf = gpd.read_file(filename, **kwargs)
    if output is None:
        return gdf.__geo_interface__
    else:
        gdf.to_file(output, driver="GeoJSON")


def geojson_to_coords(
    geojson: str, src_crs: str = "epsg:4326", dst_crs: str = "epsg:4326"
) -> list:
    """Converts a geojson file or a dictionary of feature collection to a list of centroid coordinates.

    Args:
        geojson (str | dict): The geojson file path or a dictionary of feature collection.
        src_crs (str, optional): The source CRS. Defaults to "epsg:4326".
        dst_crs (str, optional): The destination CRS. Defaults to "epsg:4326".

    Returns:
        list: A list of centroid coordinates in the format of [[x1, y1], [x2, y2], ...]
    """

    import json
    import warnings

    warnings.filterwarnings("ignore")

    if isinstance(geojson, dict):
        geojson = json.dumps(geojson)
    gdf = gpd.read_file(geojson, driver="GeoJSON")
    centroids = gdf.geometry.centroid
    centroid_list = [[point.x, point.y] for point in centroids]
    if src_crs != dst_crs:
        centroid_list = transform_coords(
            [x[0] for x in centroid_list],
            [x[1] for x in centroid_list],
            src_crs,
            dst_crs,
        )
        centroid_list = [[x, y] for x, y in zip(centroid_list[0], centroid_list[1])]
    return centroid_list


def coords_to_xy(
    src_fp: str,
    coords: np.ndarray,
    coord_crs: str = "epsg:4326",
    return_out_of_bounds: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Converts a list or array of coordinates to pixel coordinates, i.e., (col, row) coordinates.

    Args:
        src_fp: The source raster file path.
        coords: A 2D or 3D array of coordinates. Can be of shape [[x1, y1], [x2, y2], ...]
                or [[[x1, y1]], [[x2, y2]], ...].
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        return_out_of_bounds: Whether to return out-of-bounds coordinates. Defaults to False.
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A 2D or 3D array of pixel coordinates in the same format as the input.
    """
    from rasterio.warp import transform as transform_coords

    out_of_bounds = []
    if isinstance(coords, np.ndarray):
        input_is_3d = coords.ndim == 3  # Check if the input is a 3D array
    else:
        input_is_3d = False

    # Flatten the 3D array to 2D if necessary
    if input_is_3d:
        original_shape = coords.shape  # Store the original shape
        coords = coords.reshape(-1, 2)  # Flatten to 2D

    # Convert ndarray to a list if necessary
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    xs, ys = zip(*coords)
    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height
        if coord_crs != src.crs:
            xs, ys = transform_coords(coord_crs, src.crs, xs, ys, **kwargs)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, **kwargs)

    result = [[col, row] for col, row in zip(cols, rows)]

    output = []

    for i, (x, y) in enumerate(result):
        if x >= 0 and y >= 0 and x < width and y < height:
            output.append([x, y])
        else:
            out_of_bounds.append(i)

    # Convert the output back to the original shape if input was 3D
    output = np.array(output)
    if input_is_3d:
        output = output.reshape(original_shape)

    # Handle cases where no valid pixel coordinates are found
    if len(output) == 0:
        print("No valid pixel coordinates found.")
    elif len(output) < len(coords):
        print("Some coordinates are out of the image boundary.")

    if return_out_of_bounds:
        return output, out_of_bounds
    else:
        return output


def boxes_to_vector(
    coords: Union[List[List[float]], np.ndarray],
    src_crs: str,
    dst_crs: str = "EPSG:4326",
    output: Optional[str] = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Convert a list of bounding box coordinates to vector data.

    Args:
        coords (list): A list of bounding box coordinates in the format [[left, top, right, bottom], [left, top, right, bottom], ...].
        src_crs (int or str): The EPSG code or proj4 string representing the source coordinate reference system (CRS) of the input coordinates.
        dst_crs (int or str, optional): The EPSG code or proj4 string representing the destination CRS to reproject the data (default is "EPSG:4326").
        output (str or None, optional): The full file path (including the directory and filename without the extension) where the vector data should be saved.
                                       If None (default), the function returns the GeoDataFrame without saving it to a file.
        **kwargs: Additional keyword arguments to pass to geopandas.GeoDataFrame.to_file() when saving the vector data.

    Returns:
        geopandas.GeoDataFrame or None: The GeoDataFrame with the converted vector data if output is None, otherwise None if the data is saved to a file.
    """

    from shapely.geometry import box

    # Create a list of Shapely Polygon objects based on the provided coordinates
    polygons = [box(*coord) for coord in coords]

    # Create a GeoDataFrame with the Shapely Polygon objects
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=src_crs)

    # Reproject the GeoDataFrame to the specified EPSG code
    gdf_reprojected = gdf.to_crs(dst_crs)

    if output is not None:
        gdf_reprojected.to_file(output, **kwargs)
    else:
        return gdf_reprojected


def rowcol_to_xy(
    src_fp: str,
    rows: Optional[List[int]] = None,
    cols: Optional[List[int]] = None,
    boxes: Optional[List[List[int]]] = None,
    zs: Optional[List[float]] = None,
    offset: str = "center",
    output: Optional[str] = None,
    dst_crs: str = "EPSG:4326",
    **kwargs: Any,
) -> Tuple[List[float], List[float]]:
    """Converts a list of (row, col) coordinates to (x, y) coordinates.

    Args:
        src_fp (str): The source raster file path.
        rows (list, optional): A list of row coordinates. Defaults to None.
        cols (list, optional): A list of col coordinates. Defaults to None.
        boxes (list, optional): A list of (row, col) coordinates in the format of [[left, top, right, bottom], [left, top, right, bottom], ...]
        zs: zs (list or float, optional): Height associated with coordinates. Primarily used for RPC based coordinate transformations.
        offset (str, optional): Determines if the returned coordinates are for the center of the pixel or for a corner.
        output (str, optional): The output vector file path. Defaults to None.
        dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.xy.

    Returns:
        A list of (x, y) coordinates.
    """

    if boxes is not None:
        rows = []
        cols = []

        for box in boxes:
            rows.append(box[1])
            rows.append(box[3])
            cols.append(box[0])
            cols.append(box[2])

    if rows is None or cols is None:
        raise ValueError("rows and cols must be provided.")

    with rasterio.open(src_fp) as src:
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, zs, offset, **kwargs)
        src_crs = src.crs

    if boxes is None:
        return [[x, y] for x, y in zip(xs, ys)]
    else:
        result = [[xs[i], ys[i + 1], xs[i + 1], ys[i]] for i in range(0, len(xs), 2)]

        if output is not None:
            boxes_to_vector(result, src_crs, dst_crs, output)
        else:
            return result


def bbox_to_xy(
    src_fp: str, coords: List[float], coord_crs: str = "epsg:4326", **kwargs: Any
) -> List[float]:
    """Converts a list of coordinates to pixel coordinates, i.e., (col, row) coordinates.
        Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
        While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright

    Args:
        src_fp (str): The source raster file path.
        coords (list): A list of coordinates in the format of [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...]
        coord_crs (str, optional): The coordinate CRS of the input coordinates. Defaults to "epsg:4326".

    Returns:
        list: A list of pixel coordinates in the format of [[minx, maxy, maxx, miny], ...] from top left to bottom right.
    """

    if isinstance(coords, str):
        gdf = gpd.read_file(coords)
        coords = gdf.geometry.bounds.values.tolist()
        if gdf.crs is not None:
            coord_crs = f"epsg:{gdf.crs.to_epsg()}"
    elif isinstance(coords, np.ndarray):
        coords = coords.tolist()
    if isinstance(coords, dict):
        import json

        geojson = json.dumps(coords)
        gdf = gpd.read_file(geojson, driver="GeoJSON")
        coords = gdf.geometry.bounds.values.tolist()

    elif not isinstance(coords, list):
        raise ValueError("coords must be a list of coordinates.")

    if not isinstance(coords[0], list):
        coords = [coords]

    new_coords = []

    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height

        for coord in coords:
            minx, miny, maxx, maxy = coord

            if coord_crs != src.crs:
                minx, miny = transform_coords(minx, miny, coord_crs, src.crs, **kwargs)
                maxx, maxy = transform_coords(maxx, maxy, coord_crs, src.crs, **kwargs)

                rows1, cols1 = rasterio.transform.rowcol(
                    src.transform, minx, miny, **kwargs
                )
                rows2, cols2 = rasterio.transform.rowcol(
                    src.transform, maxx, maxy, **kwargs
                )

                new_coords.append([cols1, rows1, cols2, rows2])

            else:
                new_coords.append([minx, miny, maxx, maxy])

    result = []

    for coord in new_coords:
        minx, miny, maxx, maxy = coord

        if (
            minx >= 0
            and miny >= 0
            and maxx >= 0
            and maxy >= 0
            and minx < width
            and miny < height
            and maxx < width
            and maxy < height
        ):
            # Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
            # While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright
            result.append([minx, maxy, maxx, miny])

    if len(result) == 0:
        print("No valid pixel coordinates found.")
        return None
    elif len(result) == 1:
        return result[0]
    elif len(result) < len(coords):
        print("Some coordinates are out of the image boundary.")

    return result


def geojson_to_xy(
    src_fp: str, geojson: str, coord_crs: str = "epsg:4326", **kwargs: Any
) -> List[List[float]]:
    """Converts a geojson file or a dictionary of feature collection to a list of pixel coordinates.

    Args:
        src_fp: The source raster file path.
        geojson: The geojson file path or a dictionary of feature collection.
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A list of pixel coordinates in the format of [[x1, y1], [x2, y2], ...]
    """
    with rasterio.open(src_fp) as src:
        src_crs = src.crs
    coords = geojson_to_coords(geojson, coord_crs, src_crs)
    return coords_to_xy(src_fp, coords, src_crs, **kwargs)


def write_colormap(
    image: Union[str, np.ndarray],
    colormap: Union[str, Dict],
    output: Optional[str] = None,
) -> Optional[str]:
    """Write a colormap to an image.

    Args:
        image: The image to write the colormap to.
        colormap: The colormap to write to the image.
        output: The output file path.
    """
    if isinstance(colormap, str):
        colormap = leafmap.get_image_colormap(colormap)
    leafmap.write_image_colormap(image, colormap, output)


def plot_performance_metrics(
    history_path: str,
    figsize: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Plot performance metrics from a training history object and return as DataFrame.

    This function loads training history, plots available metrics (loss, IoU, F1,
    precision, recall), optionally exports to CSV, and returns all metrics as a
    pandas DataFrame for further analysis.

    Args:
        history_path (str): Path to the saved training history (.pth file).
        figsize (Optional[Tuple[int, int]]): Figure size in inches. If None,
            automatically determined based on number of metrics.
        verbose (bool): Whether to print best and final metric values. Defaults to True.
        save_path (Optional[str]): Path to save the plot image. If None, plot is not saved.
        csv_path (Optional[str]): Path to export metrics as CSV. If None, CSV is not exported.
        kwargs (Optional[Dict]): Additional keyword arguments for plt.savefig().

    Returns:
        pd.DataFrame: DataFrame containing all metrics with columns for epoch and each metric.
            Columns include: 'epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1',
            'val_precision', 'val_recall' (depending on availability in history).

    Example:
        >>> df = plot_performance_metrics(
        ...     'training_history.pth',
        ...     save_path='metrics_plot.png',
        ...     csv_path='metrics.csv'
        ... )
        >>> print(df.head())
    """
    if kwargs is None:
        kwargs = {}
    history = torch.load(history_path)

    # Handle different key naming conventions
    train_loss_key = "train_losses" if "train_losses" in history else "train_loss"
    val_loss_key = "val_losses" if "val_losses" in history else "val_loss"
    val_iou_key = "val_ious" if "val_ious" in history else "val_iou"
    # Support both new (f1) and old (dice) key formats for backward compatibility
    val_f1_key = (
        "val_f1s"
        if "val_f1s" in history
        else ("val_dices" if "val_dices" in history else "val_dice")
    )
    # Add support for precision and recall
    val_precision_key = (
        "val_precisions" if "val_precisions" in history else "val_precision"
    )
    val_recall_key = "val_recalls" if "val_recalls" in history else "val_recall"

    # Collect available metrics for plotting
    available_metrics = []
    metric_info = {
        "Loss": (train_loss_key, val_loss_key, ["Train Loss", "Val Loss"]),
        "IoU": (val_iou_key, None, ["Val IoU"]),
        "F1": (val_f1_key, None, ["Val F1"]),
        "Precision": (val_precision_key, None, ["Val Precision"]),
        "Recall": (val_recall_key, None, ["Val Recall"]),
    }

    for metric_name, (key1, key2, labels) in metric_info.items():
        if key1 in history or (key2 and key2 in history):
            available_metrics.append((metric_name, key1, key2, labels))

    # Determine number of subplots and figure size
    n_plots = len(available_metrics)
    if figsize is None:
        figsize = (5 * n_plots, 5)

    # Create DataFrame for all metrics
    n_epochs = 0
    df_data = {}

    # Add epochs
    if "epochs" in history:
        df_data["epoch"] = history["epochs"]
        n_epochs = len(history["epochs"])
    elif train_loss_key in history:
        n_epochs = len(history[train_loss_key])
        df_data["epoch"] = list(range(1, n_epochs + 1))

    # Add all available metrics to DataFrame
    if train_loss_key in history:
        df_data["train_loss"] = history[train_loss_key]
    if val_loss_key in history:
        df_data["val_loss"] = history[val_loss_key]
    if val_iou_key in history:
        df_data["val_iou"] = history[val_iou_key]
    if val_f1_key in history:
        df_data["val_f1"] = history[val_f1_key]
    if val_precision_key in history:
        df_data["val_precision"] = history[val_precision_key]
    if val_recall_key in history:
        df_data["val_recall"] = history[val_recall_key]

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Export to CSV if requested
    if csv_path:
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"Metrics exported to: {csv_path}")

    # Create plots
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for idx, (metric_name, key1, key2, labels) in enumerate(available_metrics):
            ax = axes[idx]

            if metric_name == "Loss":
                # Special handling for loss (has both train and val)
                if key1 in history:
                    ax.plot(history[key1], label=labels[0])
                if key2 and key2 in history:
                    ax.plot(history[key2], label=labels[1])
            else:
                # Single metric plots
                if key1 in history:
                    ax.plot(history[key1], label=labels[0])

            ax.set_title(metric_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            if "dpi" not in kwargs:
                kwargs["dpi"] = 150
            if "bbox_inches" not in kwargs:
                kwargs["bbox_inches"] = "tight"
            plt.savefig(save_path, **kwargs)

        plt.show()

    # Print summary statistics
    if verbose:
        print("\n=== Performance Metrics Summary ===")
        if val_iou_key in history:
            print(
                f"IoU     - Best: {max(history[val_iou_key]):.4f} | Final: {history[val_iou_key][-1]:.4f}"
            )
        if val_f1_key in history:
            print(
                f"F1      - Best: {max(history[val_f1_key]):.4f} | Final: {history[val_f1_key][-1]:.4f}"
            )
        if val_precision_key in history:
            print(
                f"Precision - Best: {max(history[val_precision_key]):.4f} | Final: {history[val_precision_key][-1]:.4f}"
            )
        if val_recall_key in history:
            print(
                f"Recall  - Best: {max(history[val_recall_key]):.4f} | Final: {history[val_recall_key][-1]:.4f}"
            )
        if val_loss_key in history:
            print(
                f"Val Loss - Best: {min(history[val_loss_key]):.4f} | Final: {history[val_loss_key][-1]:.4f}"
            )
        print("===================================\n")

    return df


def get_device() -> torch.device:
    """
    Returns the best available device for deep learning in the order:
    CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def plot_prediction_comparison(
    original_image: Union[str, np.ndarray, Image.Image],
    prediction_image: Union[str, np.ndarray, Image.Image],
    ground_truth_image: Optional[Union[str, np.ndarray, Image.Image]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    prediction_colormap: str = "gray",
    ground_truth_colormap: str = "gray",
    original_colormap: Optional[str] = None,
    indexes: Optional[List[int]] = None,
    divider: Optional[float] = None,
) -> None:
    """Plot original image, prediction, and optional ground truth side by side.

    Supports input as file paths, NumPy arrays, or PIL Images. For multi-band
    images, selected channels can be specified via `indexes`. If the image data
    is not normalized (e.g., Sentinel-2 [0, 10000]), the `divider` can be used
    to scale values for visualization.

    Args:
        original_image (Union[str, np.ndarray, Image.Image]):
            Original input image as a file path, NumPy array, or PIL Image.
        prediction_image (Union[str, np.ndarray, Image.Image]):
            Predicted segmentation mask image.
        ground_truth_image (Optional[Union[str, np.ndarray, Image.Image]], optional):
            Ground truth mask image. Defaults to None.
        titles (Optional[List[str]], optional):
            List of titles for the subplots. If not provided, default titles are used.
        figsize (Tuple[int, int], optional):
            Size of the entire figure in inches. Defaults to (15, 5).
        save_path (Optional[str], optional):
            If specified, saves the figure to this path. Defaults to None.
        show_plot (bool, optional):
            Whether to display the figure using plt.show(). Defaults to True.
        prediction_colormap (str, optional):
            Colormap to use for the prediction mask. Defaults to "gray".
        ground_truth_colormap (str, optional):
            Colormap to use for the ground truth mask. Defaults to "gray".
        original_colormap (Optional[str], optional):
            Colormap to use for the original image if it's grayscale. Defaults to None.
        indexes (Optional[List[int]], optional):
            List of band/channel indexes (0-based for NumPy, 1-based for rasterio) to extract from the original image.
            Useful for multi-band imagery like Sentinel-2. Defaults to None.
        divider (Optional[float], optional):
            Value to divide the original image by for normalization (e.g., 10000 for reflectance). Defaults to None.

    Returns:
        matplotlib.figure.Figure:
            The generated matplotlib figure object.
    """

    def _load_image(img_input, indexes=None):
        """Helper function to load image from various input types."""
        if isinstance(img_input, str):
            if img_input.lower().endswith((".tif", ".tiff")):
                with rasterio.open(img_input) as src:
                    if indexes:
                        img = src.read(indexes)  # 1-based
                        img = (
                            np.transpose(img, (1, 2, 0)) if len(indexes) > 1 else img[0]
                        )
                    else:
                        img = src.read()
                        if img.shape[0] == 1:
                            img = img[0]
                        else:
                            img = np.transpose(img, (1, 2, 0))
            else:
                img = np.array(Image.open(img_input))
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input)
        elif isinstance(img_input, np.ndarray):
            img = img_input
            if indexes is not None and img.ndim == 3:
                img = img[:, :, indexes]
        else:
            raise ValueError(f"Unsupported image type: {type(img_input)}")
        return img

    # Load images
    original = _load_image(original_image, indexes=indexes)
    prediction = _load_image(prediction_image)
    ground_truth = (
        _load_image(ground_truth_image) if ground_truth_image is not None else None
    )

    # Apply divider normalization if requested
    if divider is not None and isinstance(original, np.ndarray) and original.ndim == 3:
        original = np.clip(original.astype(np.float32) / divider, 0, 1)

    # Determine layout
    num_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 2:
        axes = [axes[0], axes[1]]

    if titles is None:
        titles = ["Original Image", "Prediction"]
        if ground_truth is not None:
            titles.append("Ground Truth")

    # Plot original
    if original.ndim == 3 and original.shape[2] in [3, 4]:
        axes[0].imshow(original)
    else:
        axes[0].imshow(original, cmap=original_colormap)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(prediction, cmap=prediction_colormap)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap=ground_truth_colormap)
        axes[2].set_title(titles[2])
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


def get_raster_resolution(image_path: str) -> Tuple[float, float]:
    """Get pixel resolution from the raster using rasterio.

    Args:
        image_path: The path to the raster image.

    Returns:
        A tuple of (x resolution, y resolution).
    """
    with rasterio.open(image_path) as src:
        res = src.res
    return res


def stack_bands(
    input_files: List[str],
    output_file: str,
    resolution: Optional[float] = None,
    dtype: Optional[str] = None,  # e.g., "UInt16", "Float32"
    temp_vrt: str = "stack.vrt",
    overwrite: bool = False,
    compress: str = "DEFLATE",
    output_format: str = "COG",
    extra_gdal_translate_args: Optional[List[str]] = None,
) -> str:
    """
    Stack bands from multiple images into a single multi-band GeoTIFF.

    Args:
        input_files (List[str]): List of input image paths.
        output_file (str): Path to the output stacked image.
        resolution (float, optional): Output resolution. If None, inferred from first image.
        dtype (str, optional): Output data type (e.g., "UInt16", "Float32").
        temp_vrt (str): Temporary VRT filename.
        overwrite (bool): Whether to overwrite the output file.
        compress (str): Compression method.
        output_format (str): GDAL output format (default is "COG").
        extra_gdal_translate_args (List[str], optional): Extra arguments for gdal_translate.

    Returns:
        str: Path to the output file.
    """
    import leafmap

    if not input_files:
        raise ValueError("No input files provided.")
    elif isinstance(input_files, str):
        input_files = leafmap.find_files(input_files, ".tif")

    if os.path.exists(output_file) and not overwrite:
        print(f"Output file already exists: {output_file}")
        return output_file

    # Infer resolution if not provided
    if resolution is None:
        resolution_x, resolution_y = get_raster_resolution(input_files[0])
    else:
        resolution_x = resolution_y = resolution

    # Step 1: Build VRT
    vrt_cmd = ["gdalbuildvrt", "-separate", temp_vrt] + input_files
    subprocess.run(vrt_cmd, check=True)

    # Step 2: Translate VRT to output GeoTIFF
    translate_cmd = [
        "gdal_translate",
        "-tr",
        str(resolution_x),
        str(resolution_y),
        temp_vrt,
        output_file,
        "-of",
        output_format,
        "-co",
        f"COMPRESS={compress}",
    ]

    if dtype:
        translate_cmd.insert(1, "-ot")
        translate_cmd.insert(2, dtype)

    if extra_gdal_translate_args:
        translate_cmd += extra_gdal_translate_args

    subprocess.run(translate_cmd, check=True)

    # Step 3: Clean up VRT
    if os.path.exists(temp_vrt):
        os.remove(temp_vrt)

    return output_file


def empty_cache() -> None:
    """Empty the cache of the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def smooth_vector(
    vector_data: Union[str, gpd.GeoDataFrame],
    output_path: str = None,
    segment_length: float = None,
    smooth_iterations: int = 3,
    num_cores: int = 0,
    merge_collection: bool = True,
    merge_field: str = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Smooth a vector data using the smoothify library.
        See https://github.com/DPIRD-DMA/Smoothify for more details.

    Args:
        vector_data: The vector data to smooth.
        output_path: The path to save the smoothed vector data. If None, returns the smoothed vector data.
        segment_length: Resolution of the original raster data in map units. If None (default), automatically
            detects by finding the minimum segment length (from a data sample). Recommended to specify explicitly when known.
        smooth_iterations: The number of iterations to smooth the vector data.
        num_cores: Number of cores to use for parallel processing. If 0 (default), uses all available cores.
        merge_collection: Whether to merge/dissolve adjacent geometries in collections before smoothing.
        merge_field: Column name to use for dissolving geometries. Only valid when merge_collection=True.
            If None, dissolves all geometries together. If specified, dissolves geometries grouped by the column values.
        merge_multipolygons: Whether to merge adjacent polygons within MultiPolygons before smoothing
        preserve_area: Whether to restore original area after smoothing via buffering (applies to Polygons only)
        area_tolerance: Percentage of original area allowed as error (e.g., 0.01 = 0.01% error = 99.99% preservation).
            Only affects Polygons when preserve_area=True

    Returns:
        gpd.GeoDataFrame: The smoothed vector data.

    Examples:
        >>> import geoai
        >>> gdf = geoai.read_vector("path/to/vector.geojson")
        >>> smoothed_gdf = geoai.smooth_vector(gdf, smooth_iterations=3, output_path="path/to/smoothed_vector.geojson")
        >>> smoothed_gdf.head()
        >>> smoothed_gdf.explore()
    """
    try:
        from smoothify import smoothify
    except ImportError:
        install_package("smoothify")
        from smoothify import smoothify

    if isinstance(vector_data, str):
        vector_data = leafmap.read_vector(vector_data)

    smoothed_vector_data = smoothify(
        geom=vector_data,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        num_cores=num_cores,
        merge_collection=merge_collection,
        merge_field=merge_field,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
        **kwargs,
    )
    if output_path is not None:
        smoothed_vector_data.to_file(output_path)
    return smoothed_vector_data
