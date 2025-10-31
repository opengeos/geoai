#!/usr/bin/env python3
"""
Example script demonstrating the new FocalLoss and class weighting functionality for landcover classification.

This script shows how to:
1. Use FocalLoss for handling class imbalance
2. Compute class weights from label files
3. Use get_loss_function helper
4. Apply min_feature_ratio filtering in tile generation
"""

import torch
import numpy as np
from geoai.utils import FocalLoss, get_loss_function, compute_class_weights


def example_focal_loss():
    """Demonstrate FocalLoss usage."""
    print("=" * 60)
    print("Example 1: Using FocalLoss for Class Imbalance")
    print("=" * 60)
    
    # Create sample data
    batch_size, num_classes, height, width = 2, 5, 64, 64
    inputs = torch.randn(batch_size, num_classes, height, width)
    
    # Simulate imbalanced labels (class 0 is dominant)
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)
    # Add some minority class pixels
    targets[:, 10:20, 10:20] = 1  # Some class 1
    targets[:, 30:35, 30:35] = 2  # Some class 2
    
    # Standard CrossEntropy Loss
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(inputs, targets)
    print(f"\n📊 Standard CrossEntropy Loss: {ce_loss.item():.4f}")
    
    # Focal Loss (focuses on hard examples)
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    focal_loss = focal_loss_fn(inputs, targets)
    print(f"🎯 Focal Loss (alpha=1.0, gamma=2.0): {focal_loss.item():.4f}")
    
    # Focal Loss with higher gamma (more focus on hard examples)
    focal_loss_fn_high = FocalLoss(alpha=1.0, gamma=3.0)
    focal_loss_high = focal_loss_fn_high(inputs, targets)
    print(f"🎯 Focal Loss (alpha=1.0, gamma=3.0): {focal_loss_high.item():.4f}")
    
    print("\n✅ Focal loss helps focus training on hard-to-classify pixels!")


def example_get_loss_function():
    """Demonstrate get_loss_function helper."""
    print("\n" + "=" * 60)
    print("Example 2: Using get_loss_function Helper")
    print("=" * 60)
    
    num_classes = 5
    
    # Get CrossEntropy loss
    print("\n🎯 Creating CrossEntropy Loss:")
    ce_loss = get_loss_function("crossentropy", num_classes=num_classes)
    
    # Get Focal loss with custom parameters
    print("\n🎯 Creating Focal Loss with custom parameters:")
    focal_loss = get_loss_function(
        "focal", 
        num_classes=num_classes,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    # Get loss with class weights
    class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Boost rare classes
    print("\n🎯 Creating Focal Loss with class weights:")
    weighted_loss = get_loss_function(
        "focal",
        num_classes=num_classes,
        use_class_weights=True,
        class_weights=class_weights,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    print("\n✅ get_loss_function makes it easy to select and configure loss functions!")


def example_ignore_index():
    """Demonstrate flexible ignore_index handling."""
    print("\n" + "=" * 60)
    print("Example 3: Flexible Ignore Index Handling")
    print("=" * 60)
    
    batch_size, num_classes, height, width = 1, 4, 32, 32
    inputs = torch.randn(batch_size, num_classes, height, width)
    
    # Create targets with some pixels marked as ignore (255)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    targets[:, 0:5, 0:5] = 255  # Mark some pixels as ignore
    
    # Standard usage: ignore index 255
    print("\n🎯 With ignore_index=255:")
    loss_fn_ignore = get_loss_function("focal", ignore_index=255, num_classes=num_classes)
    loss_ignore = loss_fn_ignore(inputs, targets)
    print(f"   Loss (ignoring 255): {loss_ignore.item():.4f}")
    
    # Don't ignore any pixels
    print("\n🎯 With ignore_index=False (no pixels ignored):")
    loss_fn_no_ignore = get_loss_function("focal", ignore_index=False, num_classes=num_classes)
    # Use targets without the ignore value
    targets_valid = targets.clone()
    targets_valid[targets_valid == 255] = 0  # Replace ignore with valid class
    loss_no_ignore = loss_fn_no_ignore(inputs, targets_valid)
    print(f"   Loss (all pixels): {loss_no_ignore.item():.4f}")
    
    print("\n✅ Flexible ignore_index allows handling of uncertain/unlabeled pixels!")


def example_class_weights_computation():
    """Demonstrate compute_class_weights function (conceptual)."""
    print("\n" + "=" * 60)
    print("Example 4: Computing Class Weights (Conceptual)")
    print("=" * 60)
    
    print("\n📝 The compute_class_weights function can:")
    print("   1. Automatically compute inverse frequency weights")
    print("   2. Apply custom multipliers to boost/reduce specific classes")
    print("   3. Cap maximum weights to prevent instability")
    print("   4. Support pure custom weights without inverse frequency")
    
    print("\n💡 Example usage:")
    print("""
    # Compute inverse frequency weights from label files
    weights = compute_class_weights(
        labels_dir="path/to/labels/",
        num_classes=5,
        ignore_index=-100
    )
    
    # Apply custom multipliers to boost rare classes
    weights = compute_class_weights(
        labels_dir="path/to/labels/",
        num_classes=5,
        custom_multipliers={4: 2.0},  # Double weight for class 4
        max_weight=50.0  # Cap weights at 50
    )
    
    # Use pure custom weights
    weights = compute_class_weights(
        labels_dir="path/to/labels/",
        num_classes=5,
        custom_multipliers={0: 0.5, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
        use_inverse_frequency=False
    )
    """)
    
    print("\n✅ compute_class_weights handles class imbalance automatically!")


def example_min_feature_ratio():
    """Demonstrate min_feature_ratio parameter (conceptual)."""
    print("\n" + "=" * 60)
    print("Example 5: Enhanced Tile Filtering with min_feature_ratio")
    print("=" * 60)
    
    print("\n📝 The enhanced export_geotiff_tiles function now supports:")
    print("   - min_feature_ratio: Filter out background-heavy tiles")
    print("   - Improved training balance by focusing on content-rich tiles")
    
    print("\n💡 Example usage:")
    print("""
    from geoai.utils import export_geotiff_tiles
    
    # Basic usage - skip completely empty tiles
    export_geotiff_tiles(
        in_raster="image.tif",
        out_folder="tiles/",
        in_class_data="labels.tif",
        skip_empty_tiles=True
    )
    
    # Advanced usage - also skip tiles with < 10% feature content
    export_geotiff_tiles(
        in_raster="image.tif",
        out_folder="tiles/",
        in_class_data="labels.tif",
        skip_empty_tiles=True,
        min_feature_ratio=0.1  # Require at least 10% labeled pixels
    )
    
    # Results in better training balance:
    # - Skips empty tiles: 150
    # - Skips background-heavy tiles: 320
    # - Generated tiles: 530 (all content-rich!)
    """)
    
    print("\n✅ min_feature_ratio improves training efficiency and balance!")


def main():
    """Run all examples."""
    print("\n" + "🚀" * 30)
    print("GeoAI Landcover Training Enhancements - Examples")
    print("🚀" * 30 + "\n")
    
    # Run examples
    example_focal_loss()
    example_get_loss_function()
    example_ignore_index()
    example_class_weights_computation()
    example_min_feature_ratio()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print("\nThese enhancements enable:")
    print("  🎯 Better handling of class imbalance with FocalLoss")
    print("  ⚖️  Automatic class weight computation")
    print("  🎨 Flexible ignore_index for uncertain pixels")
    print("  🔍 Smart tile filtering with min_feature_ratio")
    print("\nPerfect for discrete landcover classification with sparse, imbalanced data!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
