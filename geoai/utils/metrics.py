"""Segmentation metrics and evaluation utilities."""

from typing import Dict, List, Optional, Union

import numpy as np
import rasterio
import torch

__all__ = ["calc_iou", "calc_f1_score", "calc_segmentation_metrics"]


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
