#!/usr/bin/env python3
"""Test the full wetland model pipeline: train, resume, predict, visualize."""

import sys

sys.path.insert(0, "/home/qiusheng/Documents/GitHub/geoai")

import os
import numpy as np
import rasterio
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import geoai

print(f"GeoAI version: {geoai.__version__}")

images = list(Path("wetland_training_data/images").glob("*.tif"))
print(f"Training tiles available: {len(images)}")

# ── Step 1: Train from scratch (5 epochs) ──────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Training from scratch (5 epochs)")
print("=" * 60)

results1 = geoai.train_wetland_model(
    dataset_dir="wetland_training_data",
    output_dir="wetland_model_output",
    prithvi_model="Prithvi-EO-2.0-300M-TL",
    batch_size=2,
    max_epochs=5,
    learning_rate=1e-4,
    val_split=0.2,
    freeze_backbone_epochs=1,  # unfreeze backbone early for short runs
)
print(f"\n✅ Step 1 complete!")
print(f"  Best model: {results1['best_model_path']}")
print(f"  Checkpoint: {results1['checkpoint_path']}")
print(f"  Resumed from: {results1.get('resumed_from')}")

# Verify last.ckpt exists for resume
last_ckpt = Path("wetland_model_output/checkpoints/last.ckpt")
assert last_ckpt.exists(), f"last.ckpt not found at {last_ckpt}"
print(f"  last.ckpt exists: {last_ckpt}")

# ── Step 2: Resume training (5 → 10 epochs) ────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Resume training (5 → 10 epochs)")
print("=" * 60)

results2 = geoai.train_wetland_model(
    dataset_dir="wetland_training_data",
    output_dir="wetland_model_output",
    prithvi_model="Prithvi-EO-2.0-300M-TL",
    batch_size=2,
    max_epochs=10,
    learning_rate=1e-4,
    val_split=0.2,
    freeze_backbone_epochs=1,
    resume_from="last",
)
print(f"\n✅ Step 2 complete!")
print(f"  Best model: {results2['best_model_path']}")
print(f"  Checkpoint: {results2['checkpoint_path']}")
print(f"  Resumed from: {results2.get('resumed_from')}")

# ── Step 3: Run inference ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Running inference")
print("=" * 60)

naip_files = list(Path("wetland_data_cache/naip").glob("*.tif"))
test_raster = str(naip_files[0])
output_pred = "wetland_prediction_demo.tif"

print(f"  Test raster: {test_raster}")
print(f"  Output: {output_pred}")

result_path = geoai.predict_wetlands_large_image(
    model_path=results2["best_model_path"],
    input_raster=test_raster,
    output_path=output_pred,
    tile_size=512,
    overlap=64,
)
print(f"  Prediction saved: {result_path}")

# ── Step 4: Validate prediction is not empty ────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Validating prediction")
print("=" * 60)

with rasterio.open(output_pred) as src:
    pred_data = src.read(1)
    unique_classes = np.unique(pred_data)
    total_pixels = pred_data.size
    class_names = {
        0: "Background",
        1: "Freshwater Emergent",
        2: "Freshwater Forested",
        3: "Freshwater Pond",
        4: "Estuarine",
        5: "Other Wetland",
    }
    print(f"  Prediction shape: {pred_data.shape}")
    print(f"  Unique classes found: {unique_classes}")
    print(f"  Class distribution:")
    for cls in unique_classes:
        count = np.sum(pred_data == cls)
        pct = count / total_pixels * 100
        name = class_names.get(cls, f"Unknown({cls})")
        print(f"    {cls} ({name}): {count:,} pixels ({pct:.1f}%)")

    non_background = np.sum(pred_data > 0)
    print(
        f"\n  Non-background pixels: {non_background:,} ({non_background/total_pixels*100:.1f}%)"
    )

    if non_background > 0:
        print("\n✅ PREDICTION IS NOT EMPTY — wetland classes detected!")
    else:
        print("\n❌ PREDICTION IS EMPTY — all background!")

# ── Step 5: Test visualization ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Testing visualization")
print("=" * 60)

try:
    viz_map = geoai.visualize_wetland_predictions(
        prediction_path=output_pred,
        naip_path=test_raster,
        center=[46.95, -99.15],
    )
    print(f"  ✅ Visualization created successfully (type: {type(viz_map).__name__})")
except Exception as e:
    print(f"  ❌ Visualization failed: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
