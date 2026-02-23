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
        ignore_index: Specifies a target value that is ignored.
            - False: No values ignored (standard behavior)
            - int: Specific class index to ignore (e.g., 0 for background)
        reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Union[int, bool] = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        # Convert ignore_index: int stays as-is, False becomes -100 (PyTorch default)
        self.ignore_index = ignore_index if isinstance(ignore_index, int) else -100
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
    ignore_index: Union[int, bool] = False,
    smooth: float = 1e-6,
    mode: str = "mean",
    boundary_weight_map: Optional[torch.Tensor] = None,
    background_class: Optional[int] = None,
) -> Union[float, Tuple[float, List[float], List[int]]]:
    """
    Calculate IoU for landcover classification with multiple weighting options.

    Supports four IoU calculation modes:
    1. "mean": Simple mean IoU across all classes
    2. "perclass_frequency": Weight by per-class pixel frequency
    3. "boundary_weighted": Weight by distance to class boundaries
    4. "sparse_labels": For incomplete ground truth - only penalize FP where GT is positive
       (does NOT penalize predictions in unlabeled/background areas)

    Args:
        pred: Predicted classes (N, H, W) or logits (N, C, H, W)
        target: Ground truth (N, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore (default: None)
        smooth: Smoothing factor to avoid division by zero
        mode: IoU calculation mode ("mean", "perclass_frequency", "boundary_weighted", "sparse_labels")
        boundary_weight_map: Optional boundary weights (N, H, W)
        background_class: Background/unlabeled class for sparse_labels mode (default: 0)

    Returns:
        If mode == "mean": float (mean IoU)
        If mode == "perclass_frequency": tuple (weighted IoU, per-class IoUs, class counts)
        If mode == "boundary_weighted": float (boundary-weighted IoU)
        If mode == "sparse_labels": tuple (sparse IoU, per-class IoUs, per-class recall, per-class precision)
    """

    # Convert logits to class predictions if needed
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    # Ensure correct shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    # Create mask for valid pixels
    # Handle ignore_index: int means specific class, False means don't ignore
    if isinstance(ignore_index, int):
        valid_mask = target != ignore_index
    else:
        # ignore_index is False or any other non-int value - don't ignore anything
        valid_mask = torch.ones_like(target, dtype=torch.bool)

    # Simple mean IoU
    if mode == "mean":
        ious = []
        for cls in range(num_classes):
            if isinstance(ignore_index, int) and cls == ignore_index:
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
        if isinstance(ignore_index, int):
            target_filtered = target[valid_mask]
            pred_filtered = pred[valid_mask]
        else:
            target_filtered = target.view(-1)
            pred_filtered = pred.view(-1)

        total_valid_pixels = target_filtered.numel()

        for cls in range(num_classes):
            if isinstance(ignore_index, int) and cls == ignore_index:
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
            if isinstance(ignore_index, int) and cls == ignore_index:
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

    # Sparse labels IoU - for incomplete ground truth
    # Key insight: background (0) means "unlabeled", not "definitely not this class"
    # So we DON'T penalize predictions in background areas
    elif mode == "sparse_labels":
        # Default background class is 0 if not specified
        bg_class = background_class if background_class is not None else 0

        ious = []
        recalls = []
        precisions = []
        per_class_ious = []

        # Mask for labeled pixels (ground truth is NOT background)
        labeled_mask = target != bg_class
        if isinstance(ignore_index, int):
            labeled_mask = labeled_mask & (target != ignore_index)

        for cls in range(num_classes):
            # Skip background class and ignore_index
            if cls == bg_class:
                per_class_ious.append(0.0)
                recalls.append(0.0)
                precisions.append(0.0)
                continue
            if isinstance(ignore_index, int) and cls == ignore_index:
                per_class_ious.append(0.0)
                recalls.append(0.0)
                precisions.append(0.0)
                continue

            # Where prediction says this class
            pred_cls = pred == cls
            # Where ground truth says this class
            target_cls = target == cls

            # TRUE POSITIVE: Prediction matches target (both say this class)
            tp = (pred_cls & target_cls).sum().float()

            # FALSE NEGATIVE: Target says this class but prediction doesn't
            fn = (target_cls & ~pred_cls).sum().float()

            # FALSE POSITIVE (SPARSE VERSION):
            # Prediction says this class, target says DIFFERENT class (but NOT background)
            # Key: We don't count predictions in background as FP!
            fp_sparse = (pred_cls & ~target_cls & labeled_mask).sum().float()

            # Standard IoU but with sparse FP definition
            # Union = TP + FN + FP_sparse
            union_sparse = tp + fn + fp_sparse

            if union_sparse > 0:
                iou = (tp + smooth) / (union_sparse + smooth)
                ious.append(iou.item())
                per_class_ious.append(iou.item())
            else:
                per_class_ious.append(0.0)

            # Also compute recall and precision for diagnostic purposes
            # Recall: Of all true positives, how many did we find?
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
                recalls.append(recall.item())
            else:
                recalls.append(0.0)

            # Precision (sparse): Of predictions in labeled areas, how many are correct?
            if (tp + fp_sparse) > 0:
                precision = tp / (tp + fp_sparse)
                precisions.append(precision.item())
            else:
                precisions.append(0.0)

        # Mean IoU across classes (excluding background)
        mean_sparse_iou = sum(ious) / len(ious) if ious else 0.0

        return mean_sparse_iou, per_class_ious, recalls, precisions

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'mean', 'perclass_frequency', 'boundary_weighted', or 'sparse_labels'"
        )


def get_landcover_loss_function(
    loss_name: str = "crossentropy",
    num_classes: int = 2,
    ignore_index: Union[int, bool] = -100,
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
        ignore_index: Class index to ignore, or False to not ignore any class.
            - If int: pixels with this label value will be ignored during training
            - If False: no pixels will be ignored (all pixels contribute to loss)
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

        # Convert ignore_index: int stays as-is, False becomes -100 (PyTorch default)
        idx = ignore_index if isinstance(ignore_index, int) else -100

        return LandcoverCrossEntropyLoss(
            weight=weights,
            ignore_index=idx,
            reduction="mean",
        )

    elif loss_name == "focal":
        weights = class_weights if use_class_weights else None
        if weights is not None:
            weights = weights.to(device)

        # Convert ignore_index: int stays as-is, False becomes -100 (PyTorch default)
        idx = ignore_index if isinstance(ignore_index, int) else -100

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

        # Convert ignore_index: int stays as-is, False becomes -100 (PyTorch default)
        idx = ignore_index if isinstance(ignore_index, int) else -100

        return nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=idx,
            reduction="mean",
        )


def compute_class_weights(
    labels_dir: str,
    num_classes: int,
    ignore_index: Union[int, bool] = -100,
    custom_multipliers: Optional[Dict[int, float]] = None,
    max_weight: float = 50.0,
    use_inverse_frequency: bool = True,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets with optional custom multipliers and maximum weight cap.

    Args:
        labels_dir: Directory containing label files
        num_classes: Number of classes
        ignore_index: Class index to ignore when computing weights.
            - If int: specific class to ignore (pixels will be excluded from weight calc)
            - If False: no class ignored (all classes contribute to weights)
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
                    if isinstance(ignore_index, int) and class_id == ignore_index:
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
            if isinstance(ignore_index, int) and class_id == ignore_index:
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
            if isinstance(ignore_index, int) and class_id == ignore_index:
                weights[class_id] = 0.0

    # Apply custom multipliers if provided
    if custom_multipliers:
        print(f"\nApplying custom multipliers: {custom_multipliers}")
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
        print("\nNo custom multipliers provided, using computed weights as-is")

    # Apply maximum weight cap to prevent extreme values
    weights_capped = False
    print(f"\nApplying maximum weight cap of {max_weight}...")
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

    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        print(f"\nNote: Class {ignore_index} (ignore_index) has weight 0.0")

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
    ignore_index: Union[int, bool] = 0,
    use_class_weights: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    custom_multipliers: Optional[Dict[int, float]] = None,
    max_class_weight: float = 50.0,
    use_inverse_frequency: bool = True,
    validation_iou_mode: str = "standard",
    boundary_alpha: float = 1.0,
    background_class: int = 0,
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
        ignore_index: Class index to ignore during training. (default: 0)
            - If int: pixels with this label value will be ignored during training
            - If False: no pixels will be ignored (all pixels contribute to loss)
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
            - "sparse_labels": For incomplete ground truth - predictions in background
              areas are NOT penalized. Uses custom training loop with sparse IoU for
              model selection. BEST FOR INCOMPLETE/SPARSE HABITAT MASKS.
        boundary_alpha: Boundary importance factor for wIoU mode (default: 1.0)
            Higher values = more focus on boundaries (0.01-100 range)
        background_class: Class ID for background/unlabeled pixels in sparse_labels mode
            (default: 0). Predictions in this class area are NOT counted as false positives.
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

    # Convert ignore_index to format expected by PyTorch loss functions
    if isinstance(ignore_index, bool) and ignore_index is False:
        ignore_idx_for_loss = -100  # PyTorch default (effectively no ignoring)
    elif isinstance(ignore_index, int):
        ignore_idx_for_loss = ignore_index
    else:
        ignore_idx_for_loss = -100

    # Compute class weights if requested
    class_weights_tensor = None
    if use_class_weights:
        if verbose:
            print("\n" + "=" * 60)
            print("COMPUTING CLASS WEIGHTS")
            print("=" * 60)

        class_weights_tensor = compute_class_weights(
            labels_dir=labels_dir,
            num_classes=num_classes,
            ignore_index=ignore_index,
            custom_multipliers=custom_multipliers,
            max_weight=max_class_weight,
            use_inverse_frequency=use_inverse_frequency,
        )

        if verbose:
            print("=" * 60 + "\n")

    # Create custom loss function using landcover-specific implementation
    # This ensures ignore_index and class_weights are properly used
    import torch

    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if class_weights_tensor is not None:
        class_weights_tensor = class_weights_tensor.to(device)

    # Create the loss function with proper ignore_index support
    criterion = get_landcover_loss_function(
        loss_name=loss_function,
        num_classes=num_classes,
        ignore_index=ignore_idx_for_loss,
        class_weights=class_weights_tensor,
        use_class_weights=use_class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        device=device,
    )

    if verbose:
        print(
            f"Created {loss_function} loss function with ignore_index={ignore_idx_for_loss}"
        )
        if use_class_weights:
            print(f"Class weights applied: {class_weights_tensor}")

    # ==========================================================================
    # ALL MODES: Use custom training loop with landcover_iou for model selection
    # ==========================================================================
    # This ensures ALL IoU modes work correctly, not just sparse_labels
    # The base geoai training function ignores validation_iou_mode parameter

    if verbose:
        mode_descriptions = {
            "standard": "STANDARD (unweighted mean IoU)",
            "perclass_frequency": "PER-CLASS FREQUENCY-WEIGHTED IoU",
            "boundary_weighted": f"BOUNDARY-WEIGHTED IoU (wIoU, α={boundary_alpha})",
            "sparse_labels": f"SPARSE LABELS IoU (bg={background_class} ignored)",
        }
        print("\n" + "=" * 60)
        print(
            f"CUSTOM TRAINING LOOP: {mode_descriptions.get(validation_iou_mode, validation_iou_mode)}"
        )
        print("=" * 60)
        if validation_iou_mode == "sparse_labels":
            print(
                f"Background class: {background_class} (predictions here NOT penalized)"
            )
        elif validation_iou_mode == "boundary_weighted":
            print(
                f"Boundary alpha: {boundary_alpha} (higher = more focus on boundaries)"
            )
        elif validation_iou_mode == "perclass_frequency":
            print("Classes weighted by pixel frequency in dataset")
        print(f"Using {validation_iou_mode} IoU for model selection during training")
        print("=" * 60 + "\n")

    model = _train_with_custom_iou(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
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
        num_workers=num_workers,
        criterion=criterion,
        validation_iou_mode=validation_iou_mode,
        boundary_alpha=boundary_alpha,
        background_class=background_class,
        ignore_index=(
            ignore_idx_for_loss
            if isinstance(ignore_idx_for_loss, int) and ignore_idx_for_loss != -100
            else False
        ),
        training_callback=training_callback,
        **kwargs,
    )
    return model


def _compute_boundary_weight_map(
    target: torch.Tensor,
    alpha: float = 1.0,
    num_classes: int = None,
) -> torch.Tensor:
    """
    Compute boundary weight map for boundary-weighted IoU.

    Pixels near class boundaries get higher weight.
    Uses distance transform from scipy.

    Args:
        target: Ground truth tensor (N, H, W)
        alpha: Weight decay rate (higher = sharper boundary focus)
        num_classes: Number of classes (for edge detection)

    Returns:
        Weight map (N, H, W) with values in [0, 1]
    """
    import numpy as np
    from scipy import ndimage

    batch_size = target.shape[0]
    weight_maps = []

    for b in range(batch_size):
        label = target[b].numpy()

        # Find boundaries using gradient magnitude
        # Boundaries are where adjacent pixels have different classes
        gradient_x = np.abs(np.diff(label, axis=1, prepend=label[:, :1]))
        gradient_y = np.abs(np.diff(label, axis=0, prepend=label[:1, :]))
        boundary = (gradient_x > 0) | (gradient_y > 0)

        # Compute distance transform from boundaries
        # Distance is 0 at boundary, increases away from boundary
        if boundary.any():
            distance = ndimage.distance_transform_edt(~boundary)
            # Normalize distance to [0, 1]
            max_dist = distance.max()
            if max_dist > 0:
                distance = distance / max_dist
            # Compute weight: exp(-alpha * distance)
            # At boundary (distance=0): weight = 1.0
            # Far from boundary: weight approaches 0
            weight = np.exp(-alpha * distance)
        else:
            # No boundaries found - uniform weight
            weight = np.ones_like(label, dtype=np.float32)

        weight_maps.append(torch.from_numpy(weight.astype(np.float32)))

    return torch.stack(weight_maps, dim=0)


def _train_with_custom_iou(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    architecture: str,
    encoder_name: str,
    encoder_weights: Optional[str],
    num_channels: int,
    num_classes: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    val_split: float,
    print_freq: int,
    verbose: bool,
    save_best_only: bool,
    plot_curves: bool,
    device: torch.device,
    checkpoint_path: Optional[str],
    resume_training: bool,
    target_size: Optional[Tuple[int, int]],
    num_workers: Optional[int],
    criterion: nn.Module,
    validation_iou_mode: str,
    boundary_alpha: float,
    background_class: int,
    ignore_index: Union[int, bool],
    training_callback: Optional[callable],
    **kwargs,
) -> torch.nn.Module:
    """
    Internal training function with custom IoU modes for model selection.

    This implements a custom training loop that uses landcover_iou with
    the specified mode for validation and model selection, instead of
    the standard geoai IoU which only supports mean IoU.

    Supports all landcover_iou modes:
    - "standard": Simple unweighted mean IoU
    - "perclass_frequency": Classes weighted by pixel frequency
    - "boundary_weighted": Pixels near boundaries weighted more (wIoU)
    - "sparse_labels": FP only counted where ground truth is positive
    """
    import os
    import platform
    import rasterio
    import numpy as np
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    # Try to import segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch not found. Install with: pip install segmentation-models-pytorch"
        )

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get image and label files
    image_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(image_extensions)
        ]
    )
    label_files = sorted(
        [
            os.path.join(labels_dir, f)
            for f in os.listdir(labels_dir)
            if f.lower().endswith(image_extensions)
        ]
    )

    # Ensure matching files
    if len(image_files) != len(label_files):
        print("Warning: Number of image files and label files don't match!")
        basenames = [os.path.basename(f) for f in image_files]
        label_files = [
            os.path.join(labels_dir, os.path.basename(f))
            for f in image_files
            if os.path.exists(os.path.join(labels_dir, os.path.basename(f)))
        ]
        image_files = [
            f
            for f, b in zip(image_files, basenames)
            if os.path.exists(os.path.join(labels_dir, b))
        ]
        print(f"Using {len(image_files)} matching files")

    print(f"Found {len(image_files)} image files and {len(label_files)} label files")

    if len(image_files) == 0:
        raise FileNotFoundError("No matching image and label files found")

    # Split data into train and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=val_split, random_state=seed
    )

    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)} images")

    # Simple dataset class for sparse labels training
    class SparseLabelsDataset(Dataset):
        def __init__(self, image_files, label_files, num_channels, target_size=None):
            self.image_files = image_files
            self.label_files = label_files
            self.num_channels = num_channels
            self.target_size = target_size

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            # Load image
            img_path = self.image_files[idx]
            if img_path.lower().endswith((".tif", ".tiff")):
                with rasterio.open(img_path) as src:
                    image = src.read()[: self.num_channels]  # (C, H, W)
            else:
                with Image.open(img_path) as img:
                    image = np.array(img)
                    if image.ndim == 2:
                        image = np.expand_dims(image, 0)
                    elif image.ndim == 3:
                        image = np.transpose(image, (2, 0, 1))
                    image = image[: self.num_channels]

            # Load label
            label_path = self.label_files[idx]
            if label_path.lower().endswith((".tif", ".tiff")):
                with rasterio.open(label_path) as src:
                    label = src.read(1)  # (H, W)
            else:
                with Image.open(label_path) as lbl:
                    label = np.array(lbl)

            # Normalize image to 0-1
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0

            # Resize if needed
            if self.target_size is not None:
                from PIL import Image as PILImage

                # Resize image
                c, h, w = image.shape
                img_pil = PILImage.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                img_pil = img_pil.resize(
                    (self.target_size[1], self.target_size[0]), PILImage.BILINEAR
                )
                image = np.array(img_pil).astype(np.float32) / 255.0
                if image.ndim == 2:
                    image = np.expand_dims(image, 0)
                else:
                    image = image.transpose(2, 0, 1)

                # Resize label
                lbl_pil = PILImage.fromarray(label.astype(np.uint8))
                lbl_pil = lbl_pil.resize(
                    (self.target_size[1], self.target_size[0]), PILImage.NEAREST
                )
                label = np.array(lbl_pil)

            return torch.from_numpy(image), torch.from_numpy(label.astype(np.int64))

    # Create datasets
    train_dataset = SparseLabelsDataset(
        train_imgs, train_labels, num_channels, target_size
    )
    val_dataset = SparseLabelsDataset(val_imgs, val_labels, num_channels, target_size)

    # Create data loaders
    if num_workers is None:
        num_workers = 0 if platform.system() in ["Darwin", "Windows"] else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test data loader
    print("Testing data loader...")
    try:
        next(iter(train_loader))
        print("Data loader test passed.")
    except RuntimeError as e:
        if "stack expects each tensor to be equal size" in str(e):
            raise RuntimeError(
                "Images have different sizes and cannot be batched together. "
                "Please set target_size parameter to standardize image dimensions. "
                f"Example: target_size=(512, 512). Original error: {str(e)}"
            ) from e
        else:
            raise

    # Initialize model using segmentation_models_pytorch
    arch_map = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "linknet": smp.Linknet,
        "manet": smp.MAnet,
        "pan": smp.PAN,
    }

    if architecture.lower() not in arch_map:
        raise ValueError(
            f"Unknown architecture: {architecture}. Available: {list(arch_map.keys())}"
        )

    model = arch_map[architecture.lower()](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=num_channels,
        classes=num_classes,
        activation=None,
    )
    model.to(device)

    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    # ===== PERFORMANCE OPTIMIZATIONS =====
    # Enable cuDNN auto-tuner for optimal conv algorithms (fixed input size)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled")

    # Setup mixed precision training (AMP) for ~2x speedup
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    print(f"Starting training with {architecture} + {encoder_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training state
    best_iou = 0.0
    start_epoch = 0
    train_losses = []
    val_losses = []
    val_ious = []

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if resume_training:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint.get("epoch", 0) + 1
                    best_iou = checkpoint.get("best_iou", 0.0)
                    print(
                        f"Resuming from epoch {start_epoch}, best IoU: {best_iou:.4f}"
                    )
        else:
            model.load_state_dict(checkpoint)

    # Training loop
    mode_labels = {
        "standard": "STANDARD (unweighted mean)",
        "mean": "STANDARD (unweighted mean)",
        "perclass_frequency": "PER-CLASS FREQUENCY-WEIGHTED",
        "boundary_weighted": f"BOUNDARY-WEIGHTED (wIoU, α={boundary_alpha})",
        "sparse_labels": f"SPARSE LABELS (bg={background_class} ignored)",
    }
    print(
        f"\nStarting training with {mode_labels.get(validation_iou_mode, validation_iou_mode)} IoU..."
    )

    for epoch in range(start_epoch, num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Non-blocking transfers overlap CPU→GPU copy with computation
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # set_to_none=True is faster than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            # Scaled backward pass for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            if verbose and batch_idx % print_freq == 0:
                print(
                    f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)

        # ===== VALIDATION PHASE WITH CUSTOM IoU =====
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                # Non-blocking transfers
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Mixed precision inference
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Collect predictions and targets for custom IoU
                # Keep argmax on GPU, only move final result to CPU
                preds = torch.argmax(outputs, dim=1).cpu()
                all_preds.append(preds)
                all_targets.append(targets.cpu())

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate IoU based on validation_iou_mode
        # Map "standard" to "mean" for landcover_iou
        iou_mode = "mean" if validation_iou_mode == "standard" else validation_iou_mode

        if iou_mode == "mean":
            # Standard mean IoU
            val_iou = landcover_iou(
                pred=all_preds,
                target=all_targets,
                num_classes=num_classes,
                ignore_index=ignore_index,
                mode="mean",
            )
            iou_display = f"Val IoU: {val_iou:.4f}"

        elif iou_mode == "perclass_frequency":
            # Per-class frequency weighted IoU
            val_iou, per_class_ious, class_counts = landcover_iou(
                pred=all_preds,
                target=all_targets,
                num_classes=num_classes,
                ignore_index=ignore_index,
                mode="perclass_frequency",
            )
            iou_display = f"Val wIoU: {val_iou:.4f} (freq-weighted)"

        elif iou_mode == "boundary_weighted":
            # Boundary-weighted IoU - compute boundary weight map
            boundary_weight_map = _compute_boundary_weight_map(
                all_targets, alpha=boundary_alpha, num_classes=num_classes
            )
            val_iou = landcover_iou(
                pred=all_preds,
                target=all_targets,
                num_classes=num_classes,
                ignore_index=ignore_index,
                mode="boundary_weighted",
                boundary_weight_map=boundary_weight_map,
            )
            iou_display = f"Val wIoU: {val_iou:.4f} (boundary, α={boundary_alpha})"

        elif iou_mode == "sparse_labels":
            # Sparse labels IoU - FP only in labeled areas
            val_iou, per_class_ious, recalls, precisions = landcover_iou(
                pred=all_preds,
                target=all_targets,
                num_classes=num_classes,
                ignore_index=ignore_index,
                mode="sparse_labels",
                background_class=background_class,
            )
            iou_display = (
                f"Val Sparse IoU: {val_iou:.4f} (bg={background_class} ignored)"
            )

        else:
            raise ValueError(f"Unknown validation_iou_mode: {validation_iou_mode}")

        val_ious.append(val_iou)

        # Update learning rate based on IoU
        lr_scheduler.step(val_iou)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"{iou_display}"
        )

        # Call training callback if provided
        if training_callback is not None:
            try:
                # Determine if this is the best epoch
                is_best = val_iou > best_iou

                # Call callback with positional arguments
                training_callback(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_iou=val_iou,
                    val_dice=0.0,  # Dice score not computed in this loop
                    is_best=is_best,
                )
            except Exception as e:
                print(f"Warning: Training callback error: {e}")

        # Save best model based on validation IoU
        if val_iou > best_iou:
            best_iou = val_iou
            print(f"New best model! {iou_display}")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # Save checkpoint every 10 epochs if not save_best_only
        if not save_best_only and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_iou": best_iou,
                    "architecture": architecture,
                    "encoder_name": encoder_name,
                    "num_channels": num_channels,
                    "num_classes": num_classes,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_ious": val_ious,
                    "validation_iou_mode": validation_iou_mode,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Mode-agnostic display labels
    mode_labels = {
        "standard": "Standard IoU",
        "mean": "Mean IoU",
        "perclass_frequency": "Frequency-Weighted IoU",
        "boundary_weighted": "Boundary-Weighted IoU",
        "sparse_labels": "Sparse Labels IoU",
    }
    iou_label = mode_labels.get(validation_iou_mode, validation_iou_mode)

    print(f"\nTraining complete! Best {iou_label}: {best_iou:.4f}")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pth')}")

    # Plot training curves if requested
    if plot_curves:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss plot
            axes[0].plot(train_losses, label="Train Loss")
            axes[0].plot(val_losses, label="Val Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training and Validation Loss")
            axes[0].legend()

            # IoU plot
            axes[1].plot(val_ious, label=f"Val {iou_label}", color="green")
            axes[1].axhline(
                y=best_iou, color="r", linestyle="--", label=f"Best: {best_iou:.4f}"
            )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel(iou_label)
            axes[1].set_title(f"Validation {iou_label}")
            axes[1].legend()

            plt.tight_layout()
            plot_filename = f"training_curves_{validation_iou_mode}.png"
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=150)
            plt.show()
            print(
                f"Training curves saved to: {os.path.join(output_dir, plot_filename)}"
            )
        except Exception as e:
            print(f"Warning: Could not plot training curves: {e}")

    # Load best model weights
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model ({iou_label}: {best_iou:.4f})")

    return model


def evaluate_sparse_iou(
    model: torch.nn.Module,
    images_dir: str,
    labels_dir: str,
    num_classes: int,
    num_channels: int = 3,
    batch_size: int = 8,
    background_class: int = 0,
    ignore_index: Union[int, bool] = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model using sparse labels IoU.

    This function is designed for incomplete/sparse ground truth where
    background (0) means "unlabeled" rather than "definitely not this class".
    Predictions in background areas are NOT penalized as false positives.

    Use this for post-training evaluation when your training masks are incomplete.

    Args:
        model: Trained segmentation model
        images_dir: Directory containing validation images
        labels_dir: Directory containing validation labels
        num_classes: Number of classes
        num_channels: Number of input channels (default: 3)
        batch_size: Batch size for evaluation (default: 8)
        background_class: Class ID for background/unlabeled pixels (default: 0)
        ignore_index: Class to ignore during evaluation.
            - If int: specific class index to ignore
            - If False: no class ignored (default)
        device: Torch device (auto-detected if None)
        verbose: Print detailed results (default: True)

    Returns:
        Dictionary containing:
        - 'mean_sparse_iou': Mean IoU across all non-background classes
        - 'per_class_iou': Dict of class_id -> IoU
        - 'per_class_recall': Dict of class_id -> recall (sensitivity)
        - 'per_class_precision': Dict of class_id -> precision
        - 'mean_recall': Mean recall across classes
        - 'mean_precision': Mean precision across classes

    Example:
        >>> model = torch.load("best_model.pth")
        >>> results = evaluate_sparse_iou(
        ...     model=model,
        ...     images_dir="tiles/images",
        ...     labels_dir="tiles/labels",
        ...     num_classes=18,
        ...     background_class=0,
        ... )
        >>> print(f"Sparse IoU: {results['mean_sparse_iou']:.4f}")
    """
    import os
    import rasterio
    from torch.utils.data import DataLoader, Dataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Get all image and label files
    image_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(image_extensions)
        ]
    )
    label_files = sorted(
        [
            os.path.join(labels_dir, f)
            for f in os.listdir(labels_dir)
            if f.lower().endswith(image_extensions)
        ]
    )

    if len(image_files) != len(label_files):
        raise ValueError(
            f"Mismatch: {len(image_files)} images vs {len(label_files)} labels"
        )

    if verbose:
        print(f"\n{'='*60}")
        print("SPARSE LABELS IoU EVALUATION")
        print(f"{'='*60}")
        print(f"Evaluating {len(image_files)} image-label pairs")
        print(f"Background class: {background_class} (predictions here NOT penalized)")
        print(f"Number of classes: {num_classes}")

    # Accumulate predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (img_path, label_path) in enumerate(zip(image_files, label_files)):
            # Load image
            with rasterio.open(img_path) as src:
                image = src.read()[:num_channels]  # (C, H, W)

            # Load label
            with rasterio.open(label_path) as src:
                label = src.read(1)  # (H, W)

            # Normalize image to 0-1 range
            image = image.astype("float32")
            if image.max() > 1.0:
                image = image / 255.0

            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            label_tensor = torch.from_numpy(label).unsqueeze(0)

            # Get prediction
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).cpu()

            all_preds.append(pred)
            all_targets.append(label_tensor)

            if verbose and (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(image_files)} tiles...")

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate sparse IoU
    mean_iou, per_class_ious, recalls, precisions = landcover_iou(
        pred=all_preds,
        target=all_targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        mode="sparse_labels",
        background_class=background_class,
    )

    # Build results dictionary
    per_class_iou_dict = {}
    per_class_recall_dict = {}
    per_class_precision_dict = {}

    class_idx = 0
    for cls in range(num_classes):
        if cls == background_class:
            continue
        if isinstance(ignore_index, int) and cls == ignore_index:
            continue

        per_class_iou_dict[cls] = per_class_ious[cls]
        per_class_recall_dict[cls] = (
            recalls[class_idx] if class_idx < len(recalls) else 0.0
        )
        per_class_precision_dict[cls] = (
            precisions[class_idx] if class_idx < len(precisions) else 0.0
        )
        class_idx += 1

    # Calculate means (excluding background)
    valid_recalls = [r for r in recalls if r > 0]
    valid_precisions = [p for p in precisions if p > 0]
    mean_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
    mean_precision = (
        sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0.0
    )

    results = {
        "mean_sparse_iou": mean_iou,
        "per_class_iou": per_class_iou_dict,
        "per_class_recall": per_class_recall_dict,
        "per_class_precision": per_class_precision_dict,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
    }

    if verbose:
        print("\nSPARSE LABELS IoU RESULTS:")
        print(f"   (Predictions in background areas NOT counted as false positives)")
        print(f"\n   {'Class':<8} {'IoU':>8} {'Recall':>8} {'Precision':>10}")
        print(f"   {'-'*36}")
        for cls in sorted(per_class_iou_dict.keys()):
            iou = per_class_iou_dict.get(cls, 0.0)
            recall = per_class_recall_dict.get(cls, 0.0)
            precision = per_class_precision_dict.get(cls, 0.0)
            print(f"   {cls:<8} {iou:>8.4f} {recall:>8.4f} {precision:>10.4f}")

        print(f"   {'-'*36}")
        print(
            f"   {'MEAN':<8} {mean_iou:>8.4f} {mean_recall:>8.4f} {mean_precision:>10.4f}"
        )
        print("\nSparse IoU evaluation complete!")
        print(f"{'='*60}\n")

    return results


# Export main functions
__all__ = [
    "FocalLoss",
    "LandcoverCrossEntropyLoss",
    "landcover_iou",
    "get_landcover_loss_function",
    "compute_class_weights",
    "train_segmentation_landcover",
    "evaluate_sparse_iou",
]
