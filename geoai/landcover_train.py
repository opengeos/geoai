"""
Landcover Classification Training Module

This module extends the base geoai training functionality with specialized
components for discrete landcover classification, including:
- Enhanced loss functions with boundary weighting
- Per-class frequency weighting for imbalanced datasets
- Configurable ignore_index handling
- Additional validation metrics

Key Features:
- Maintains full compatibility with base geoai workflow
- Adds optional advanced loss computation modes
- Provides flexible ignore_index configuration
- Optimized for multi-class landcover segmentation

Author: ValHab Project
Date: November 2025
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.

    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV.

    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        ignore_index: Specifies a target value that is ignored
        reduction: Specifies the reduction to apply to the output
        weight: Manual rescaling weight given to each class
    """

    def __init__(
        self, alpha=1.0, gamma=2.0, ignore_index=-100, reduction="mean", weight=None
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Forward pass of focal loss.

        Args:
            inputs: Predictions (N, C, H, W) where C = number of classes
            targets: Ground truth (N, H, W) with class indices

        Returns:
            Loss value
        """
        # Get class probabilities
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Get probability of true class
        p_t = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LandcoverCrossEntropyLoss(nn.Module):
    """
    Enhanced CrossEntropyLoss with optional ignore_index and class weights.

    This extends the standard CrossEntropyLoss with more flexible ignore_index
    handling, specifically designed for landcover classification tasks.

    Args:
        weight: Manual rescaling weight given to each class
        ignore_index: Specifies a target value that is ignored (default: None)
            - None: No values ignored (standard behavior)
            - int: Specific class index to ignore (e.g., 0 for background)
        reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index if ignore_index is not None else -100
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross entropy loss.

        Args:
            input: Predictions (N, C, H, W) where C = number of classes
            target: Ground truth (N, H, W) with class indices

        Returns:
            Loss value
        """
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


def landcover_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    smooth: float = 1e-6,
    mode: str = "mean",
    boundary_weight_map: Optional[torch.Tensor] = None,
) -> Union[float, Tuple[float, List[float], List[int]]]:
    """
    Calculate IoU for landcover classification with multiple weighting options.

    Supports three IoU calculation modes:
    1. "mean": Simple mean IoU across all classes
    2. "perclass_frequency": Weight by per-class pixel frequency
    3. "boundary_weighted": Weight by distance to class boundaries

    Args:
        pred: Predicted classes (N, H, W) or logits (N, C, H, W)
        target: Ground truth (N, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore (default: None)
        smooth: Smoothing factor to avoid division by zero
        mode: IoU calculation mode ("mean", "perclass_frequency", "boundary_weighted")
        boundary_weight_map: Optional boundary weights (N, H, W)

    Returns:
        If mode == "mean": float (mean IoU)
        If mode == "perclass_frequency": tuple (weighted IoU, per-class IoUs, class counts)
        If mode == "boundary_weighted": float (boundary-weighted IoU)
    """

    # Convert logits to class predictions if needed
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    # Ensure correct shape
    assert (
        pred.shape == target.shape
    ), f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = target != ignore_index
    else:
        valid_mask = torch.ones_like(target, dtype=torch.bool)

    # Simple mean IoU
    if mode == "mean":
        ious = []
        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue

            pred_cls = (pred == cls) & valid_mask
            target_cls = (target == cls) & valid_mask

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union > 0:
                iou = (intersection + smooth) / (union + smooth)
                ious.append(iou.item())

        return sum(ious) / len(ious) if ious else 0.0

    # Per-class frequency weighted IoU
    elif mode == "perclass_frequency":
        ious = []
        class_counts = []

        # Filter out ignore_index from target
        if ignore_index is not None:
            target_filtered = target[valid_mask]
            pred_filtered = pred[valid_mask]
        else:
            target_filtered = target.view(-1)
            pred_filtered = pred.view(-1)

        total_valid_pixels = target_filtered.numel()

        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue

            pred_cls = pred_filtered == cls
            target_cls = target_filtered == cls

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            class_pixel_count = target_cls.sum().item()

            if union > 0:
                iou = (intersection + smooth) / (union + smooth)
                ious.append(iou.item())
                class_counts.append(class_pixel_count)
            else:
                ious.append(0.0)
                class_counts.append(0)

        # Calculate frequency-weighted IoU
        if sum(class_counts) > 0:
            weights = [count / total_valid_pixels for count in class_counts]
            weighted_iou = sum(iou * weight for iou, weight in zip(ious, weights))
        else:
            weighted_iou = 0.0

        return weighted_iou, ious, class_counts

    # Boundary-weighted IoU
    elif mode == "boundary_weighted":
        if boundary_weight_map is None:
            raise ValueError("boundary_weight_map required for boundary_weighted mode")

        ious = []
        weights = []

        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue

            pred_cls = (pred == cls) & valid_mask
            target_cls = (target == cls) & valid_mask

            # Weight by boundary map
            weighted_intersection = (
                pred_cls & target_cls
            ).float() * boundary_weight_map
            weighted_union = (pred_cls | target_cls).float() * boundary_weight_map

            intersection_sum = weighted_intersection.sum()
            union_sum = weighted_union.sum()

            if union_sum > 0:
                iou = (intersection_sum + smooth) / (union_sum + smooth)
                weight = union_sum.item()
                ious.append(iou.item())
                weights.append(weight)

        if sum(weights) > 0:
            weighted_iou = sum(iou * w for iou, w in zip(ious, weights)) / sum(weights)
        else:
            weighted_iou = 0.0

        return weighted_iou

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'mean', 'perclass_frequency', or 'boundary_weighted'"
        )


def get_landcover_loss_function(
    loss_name: str = "crossentropy",
    num_classes: int = 2,
    ignore_index: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    use_class_weights: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Get loss function configured for landcover classification.

    Args:
        loss_name: Name of loss function ("crossentropy", "focal", "dice", "combo")
        num_classes: Number of classes
        ignore_index: Class index to ignore (default: None for no ignoring)
        class_weights: Manual class weights tensor
        use_class_weights: Whether to use class weights
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        device: Device to place loss function on

    Returns:
        Configured loss function
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_name = loss_name.lower()

    if loss_name == "crossentropy":
        weights = class_weights if use_class_weights else None
        if weights is not None:
            weights = weights.to(device)

        return LandcoverCrossEntropyLoss(
            weight=weights,
            ignore_index=ignore_index,
            reduction="mean",
        )

    elif loss_name == "focal":
        weights = class_weights if use_class_weights else None
        if weights is not None:
            weights = weights.to(device)

        # Use -100 as default ignore_index for compatibility
        idx = ignore_index if ignore_index is not None else -100

        return FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=idx,
            reduction="mean",
            weight=weights,
        )

    else:
        # Fall back to standard PyTorch loss
        weights = class_weights if use_class_weights else None
        if weights is not None:
            weights = weights.to(device)

        # Use -100 as default ignore_index for compatibility
        idx = ignore_index if ignore_index is not None else -100

        return nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=idx,
            reduction="mean",
        )


def compute_class_weights(
    labels_dir: str,
    num_classes: int,
    ignore_index: Optional[int] = None,
    custom_multipliers: Optional[Dict[int, float]] = None,
    max_weight: float = 50.0,
    use_inverse_frequency: bool = True,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets with optional custom multipliers and maximum weight cap.

    Args:
        labels_dir: Directory containing label files
        num_classes: Number of classes
        ignore_index: Class index to ignore when computing weights (default: None)
        custom_multipliers: Custom multipliers for specific classes after inverse frequency calculation.
            Format: {class_id: multiplier}
            Example: {1: 0.5, 7: 2.0} - reduce class 1 weight by half, double class 7 weight
        max_weight: Maximum allowed weight value to prevent extreme values (default: 50.0)
        use_inverse_frequency: Whether to compute inverse frequency weights.
            - True (default): Compute inverse frequency weights, then apply custom multipliers
            - False: Use uniform weights (1.0) for all classes, then apply custom multipliers

    Returns:
        Tensor of class weights (num_classes,) with custom adjustments and maximum weight cap applied
    """
    import os
    import rasterio
    from collections import Counter

    # Count pixels for each class
    class_counts = Counter()
    total_pixels = 0

    # Get all label files
    label_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    label_files = [
        os.path.join(labels_dir, f)
        for f in os.listdir(labels_dir)
        if f.lower().endswith(label_extensions)
    ]

    print(f"Computing class weights from {len(label_files)} label files...")

    for label_file in label_files:
        try:
            with rasterio.open(label_file) as src:
                label_data = src.read(1)
                for class_id in range(num_classes):
                    if ignore_index is not None and class_id == ignore_index:
                        continue
                    count = (label_data == class_id).sum()
                    class_counts[class_id] += int(count)
                    total_pixels += int(count)
        except Exception as e:
            print(f"Warning: Could not read {label_file}: {e}")
            continue

    if total_pixels == 0:
        raise ValueError("No valid pixels found in label files")

    # Initialize weights
    weights = torch.ones(num_classes)

    if use_inverse_frequency:
        # Compute inverse frequency weights
        for class_id in range(num_classes):
            if ignore_index is not None and class_id == ignore_index:
                weights[class_id] = 0.0
            elif class_counts[class_id] > 0:
                # Inverse frequency: total_pixels / class_pixels
                weights[class_id] = total_pixels / class_counts[class_id]
            else:
                weights[class_id] = 0.0

        # Normalize to have mean weight of 1.0
        non_zero_weights = weights[weights > 0]
        if len(non_zero_weights) > 0:
            weights = weights / non_zero_weights.mean()
    else:
        # Use uniform weights (all 1.0)
        for class_id in range(num_classes):
            if ignore_index is not None and class_id == ignore_index:
                weights[class_id] = 0.0

    # Apply custom multipliers if provided
    if custom_multipliers:
        print(f"\n🎯 Applying custom multipliers: {custom_multipliers}")
        for class_id, multiplier in custom_multipliers.items():
            if class_id < 0 or class_id >= num_classes:
                print(f"Warning: Invalid class_id {class_id}, skipping")
                continue

            original_weight = weights[class_id].item()
            weights[class_id] = weights[class_id] * multiplier
            print(
                f"  Class {class_id}: {original_weight:.4f} × {multiplier} = {weights[class_id].item():.4f}"
            )
    else:
        print("\nℹ️  No custom multipliers provided, using computed weights as-is")

    # Apply maximum weight cap to prevent extreme values
    weights_capped = False
    print(f"\n🔒 Applying maximum weight cap of {max_weight}...")
    for class_id in range(num_classes):
        if weights[class_id] > max_weight:
            print(
                f"  Class {class_id}: {weights[class_id].item():.4f} → {max_weight} (capped)"
            )
            weights[class_id] = max_weight
            weights_capped = True

    if not weights_capped:
        print("  No weights exceeded the cap")

    print(f"\nClass pixel counts: {dict(class_counts)}")
    print(f"\nFinal class weights:")
    for class_id in range(num_classes):
        pixel_count = class_counts.get(class_id, 0)
        percent = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0
        print(
            f"  Class {class_id}: weight={weights[class_id].item():.4f}, "
            f"pixels={pixel_count:,} ({percent:.2f}%)"
        )

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        print(f"\n⚠️  Note: Class {ignore_index} (ignore_index) has weight 0.0")

    return weights


def train_segmentation_landcover(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_format: str = "directory",
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    num_channels: int = 3,
    num_classes: int = 2,
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    print_freq: int = 10,
    verbose: bool = True,
    save_best_only: bool = True,
    plot_curves: bool = False,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    resume_training: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    resize_mode: str = "resize",
    num_workers: Optional[int] = None,
    loss_function: str = "crossentropy",
    ignore_index: Optional[int] = None,
    use_class_weights: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    custom_multipliers: Optional[Dict[int, float]] = None,
    max_class_weight: float = 50.0,
    use_inverse_frequency: bool = True,
    validation_iou_mode: str = "standard",
    boundary_alpha: float = 1.0,
    training_callback: Optional[callable] = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Train a semantic segmentation model with landcover-specific enhancements.

    This is a standalone version that wraps geoai.train.train_segmentation_model
    with landcover-specific loss functions, class weights, and metrics.

    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        output_dir: Directory to save model checkpoints and training history
        input_format: Data format ("directory", "COCO", "YOLO")
        architecture: Model architecture (default: "unet")
        encoder_name: Encoder backbone (default: "resnet34")
        encoder_weights: Pretrained weights ("imagenet" or None)
        num_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        batch_size: Training batch size (default: 8)
        num_epochs: Number of training epochs (default: 50)
        learning_rate: Initial learning rate (default: 0.001)
        weight_decay: Weight decay for optimizer (default: 1e-4)
        seed: Random seed for reproducibility (default: 42)
        val_split: Validation split ratio (default: 0.2)
        print_freq: Frequency of training progress prints (default: 10)
        verbose: Enable verbose output (default: True)
        save_best_only: Only save best model checkpoint (default: True)
        plot_curves: Plot training curves at end (default: False)
        device: Torch device (auto-detected if None)
        checkpoint_path: Path to checkpoint for resuming training
        resume_training: Whether to resume from checkpoint (default: False)
        target_size: Target size for resizing images (H, W) or None
        resize_mode: How to resize ("resize", "crop", or "pad")
        num_workers: Number of dataloader workers (default: auto)
        loss_function: Loss function name ("crossentropy", "focal")
        ignore_index: Class index to ignore (0 for background, None to include all)
        use_class_weights: Whether to compute and use class weights (default: False)
        focal_alpha: Focal loss alpha parameter (default: 1.0)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        custom_multipliers: Custom class weight multipliers {class_id: multiplier}
        max_class_weight: Maximum allowed class weight (default: 50.0)
        use_inverse_frequency: Use inverse frequency for weights (default: True)
        validation_iou_mode: IoU calculation mode for validation (default: "standard")
            - "standard": Unweighted mean IoU (all classes equal importance)
            - "perclass_frequency": Frequency-weighted IoU (classes weighted by pixel count)
            - "boundary_weighted": Boundary-distance weighted IoU (wIoU, focus on edges)
        boundary_alpha: Boundary importance factor for wIoU mode (default: 1.0)
            Higher values = more focus on boundaries (0.01-100 range)
        training_callback: Optional callback function for automatic metric tracking
        **kwargs: Additional arguments passed to base training function

    Returns:
        Trained model

    Example:
        >>> from landcover_train import train_segmentation_landcover
        >>>
        >>> model = train_segmentation_landcover(
        ...     images_dir="tiles/images",
        ...     labels_dir="tiles/labels",
        ...     output_dir="models/landcover_001",
        ...     num_classes=5,
        ...     loss_function="focal",
        ...     ignore_index=0,  # Ignore background
        ...     use_class_weights=True,
        ...     custom_multipliers={1: 1.5, 4: 0.8},  # Boost class 1, reduce class 4
        ...     max_class_weight=50.0,
        ...     use_inverse_frequency=True,  # Use inverse frequency weighting
        ...     validation_iou_mode="boundary_weighted",  # Focus on boundaries
        ...     boundary_alpha=2.0,  # Moderate boundary emphasis
        ... )
    """

    # Import geoai training function
    try:
        from geoai.train import train_segmentation_model
    except ImportError:
        raise ImportError("geoai package not found. Install with: pip install geoai-py")

    # Convert ignore_index to format expected by base function
    # Base function uses Union[int, bool], we use Optional[int]
    ignore_idx_param = ignore_index if ignore_index is not None else False

    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPUTING CLASS WEIGHTS")
            print("=" * 60)

        class_weights = compute_class_weights(
            labels_dir=labels_dir,
            num_classes=num_classes,
            ignore_index=ignore_index if ignore_index is not None else -100,
            custom_multipliers=custom_multipliers,
            max_weight=max_class_weight,
            use_inverse_frequency=use_inverse_frequency,
        )

        if verbose:
            print("=" * 60 + "\n")

    # Call base training function with enhanced parameters
    model = train_segmentation_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        input_format=input_format,
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=num_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        val_split=val_split,
        print_freq=print_freq,
        verbose=verbose,
        save_best_only=save_best_only,
        plot_curves=plot_curves,
        device=device,
        checkpoint_path=checkpoint_path,
        resume_training=resume_training,
        target_size=target_size,
        resize_mode=resize_mode,
        num_workers=num_workers,
        loss_function=loss_function,
        ignore_index=ignore_idx_param,
        use_class_weights=use_class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        custom_multipliers=custom_multipliers,
        max_class_weight=max_class_weight,
        use_inverse_frequency=use_inverse_frequency,
        validation_iou_mode=validation_iou_mode,
        boundary_alpha=boundary_alpha,
        training_callback=training_callback,
        **kwargs,
    )

    return model


# Export main functions
__all__ = [
    "FocalLoss",
    "LandcoverCrossEntropyLoss",
    "landcover_iou",
    "get_landcover_loss_function",
    "compute_class_weights",
    "train_segmentation_landcover",
]
