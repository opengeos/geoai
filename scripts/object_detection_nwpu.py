"""End-to-end multi-class object detection example using NWPU-VHR-10.

This script demonstrates the complete workflow for multi-class object
detection on remote sensing imagery using the NWPU-VHR-10 benchmark
dataset. It downloads the dataset, prepares train/val splits, trains a
Mask R-CNN model, evaluates it with COCO metrics, and runs inference.

Usage:
    python scripts/object_detection_nwpu.py

Requirements:
    pip install geoai-py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script usage

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import geoai

# ============================================================
# Step 1: Download NWPU-VHR-10 dataset
# ============================================================
print("=" * 60)
print("Step 1: Downloading NWPU-VHR-10 dataset")
print("=" * 60)

data_dir = geoai.download_nwpu_vhr10()

print(f"Dataset directory: {data_dir}")
print(f"Contents: {os.listdir(data_dir)}")

print(f"\nNWPU-VHR-10 Classes:")
for i, name in enumerate(geoai.NWPU_VHR10_CLASSES):
    print(f"  {i}: {name}")

# ============================================================
# Step 2: Prepare dataset (train/val split)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Preparing dataset")
print("=" * 60)

splits = geoai.prepare_nwpu_vhr10(data_dir, val_split=0.2, seed=42)

print(f"Images directory: {splits['images_dir']}")
print(f"Number of classes: {splits['num_classes']}")
print(f"Class names: {splits['class_names']}")
print(f"Training images: {len(splits['train_image_ids'])}")
print(f"Validation images: {len(splits['val_image_ids'])}")

# ============================================================
# Step 3: Visualize sample annotations
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Visualizing sample annotations")
print("=" * 60)

with open(splits["annotations_path"], "r") as f:
    coco_data = json.load(f)

sample_images = coco_data["images"][:4]
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
cmap = plt.cm.get_cmap("tab10", 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

for ax_idx, img_info in enumerate(sample_images):
    img_path = os.path.join(splits["images_dir"], img_info["file_name"])
    img = Image.open(img_path)
    axes[ax_idx].imshow(img)
    axes[ax_idx].set_title(img_info["file_name"], fontsize=10)
    axes[ax_idx].axis("off")

    img_anns = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
    ]
    for ann in img_anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        color = cmap(cat_id % 10)
        rect = plt.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        axes[ax_idx].add_patch(rect)
        axes[ax_idx].text(
            x,
            y - 3,
            categories.get(cat_id, str(cat_id)),
            color="white",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )

plt.tight_layout()
plt.savefig("nwpu_sample_annotations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Sample annotations saved to nwpu_sample_annotations.png")

# ============================================================
# Step 4: Train multi-class detection model
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Training multi-class detection model")
print("=" * 60)

output_dir = "nwpu_output"

model_path = geoai.train_multiclass_detector(
    images_dir=splits["images_dir"],
    annotations_path=splits["train_annotations"],
    output_dir=output_dir,
    class_names=splits["class_names"],
    num_channels=3,
    batch_size=4,
    num_epochs=20,
    learning_rate=0.005,
    val_split=0.15,
    seed=42,
    pretrained=True,
    verbose=True,
)

# ============================================================
# Step 5: Plot training metrics
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Plotting training metrics")
print("=" * 60)

history_path = os.path.join(output_dir, "training_history.pth")
if os.path.exists(history_path):
    history = torch.load(history_path, weights_only=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epochs"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epochs"], history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    axes[1].plot(history["epochs"], history["val_iou"], label="Val IoU", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU")
    axes[1].legend()

    axes[2].plot(
        history["epochs"], history["lr"], label="Learning Rate", color="orange"
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("nwpu_training_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training metrics saved to nwpu_training_metrics.png")

# ============================================================
# Step 6: Evaluate model with COCO metrics
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Evaluating model")
print("=" * 60)

metrics = geoai.evaluate_multiclass_detector(
    model_path=model_path,
    images_dir=splits["images_dir"],
    annotations_path=splits["val_annotations"],
    num_classes=splits["num_classes"],
    class_names=splits["class_names"][1:],  # Exclude background
    batch_size=4,
)

# ============================================================
# Step 7: Run inference on a validation image
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Running inference")
print("=" * 60)

with open(splits["val_annotations"], "r") as f:
    val_data = json.load(f)

test_img_info = val_data["images"][0]
test_img_path = os.path.join(splits["images_dir"], test_img_info["file_name"])
print(f"Test image: {test_img_path}")

output_raster = "nwpu_detection_output.tif"

result_path, inference_time, detections = geoai.multiclass_detection(
    input_path=test_img_path,
    output_path=output_raster,
    model_path=model_path,
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    num_channels=3,
)

print(f"\nInference time: {inference_time:.2f}s")
print(f"Total detections: {len(detections)}")

# ============================================================
# Step 8: Visualize detections
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Visualizing detections")
print("=" * 60)

geoai.visualize_multiclass_detections(
    image_path=test_img_path,
    detections=detections,
    class_names=splits["class_names"],
    confidence_threshold=0.5,
    figsize=(12, 10),
    output_path="nwpu_detections.png",
)
print("Detection visualization saved to nwpu_detections.png")

# Clean up
if os.path.exists(output_raster):
    os.remove(output_raster)

print("\n" + "=" * 60)
print("Done! Multi-class object detection pipeline complete.")
print("=" * 60)
