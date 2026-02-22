"""Training script for building detection model using the WHU Building Dataset.

This script downloads the WHU Building Dataset, trains a ConvNeXt-Base + UNet++
semantic segmentation model, evaluates it, and pushes the trained model to
HuggingFace Hub.

WHU Building Dataset:
    - Aerial imagery at 0.3m resolution
    - 512x512 RGB tiles with binary building masks
    - Train: 4,736 tiles, Val: 1,036 tiles, Test: 2,416 tiles
    - Reference: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

Usage:
    python scripts/train_whu_building.py

Requirements:
    pip install geoai-py timm segmentation-models-pytorch lightning
"""

import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

import geoai


def download_and_prepare_whu_dataset(data_dir="whu_building_dataset"):
    """Download and prepare the WHU Building Dataset.

    Downloads the raw dataset from Source Cooperative, then converts
    PNG images to GeoTIFF and remaps labels from 0/255 to 0/1.

    Args:
        data_dir: Directory to store the prepared dataset.

    Returns:
        str: Path to the prepared dataset directory.
    """
    url = "https://data.source.coop/giswqs/opengeos/geoai/whu_building_dataset.zip"
    print("Downloading WHU Building Dataset...")
    raw_path = geoai.download_file(url)
    print(f"Dataset downloaded to: {raw_path}")

    # Find the extracted directory containing train/val/test
    raw_dir = raw_path
    if os.path.isdir(os.path.join(raw_path, "WHU")):
        raw_dir = os.path.join(raw_path, "WHU")

    print("Preprocessing dataset (PNG → TIF, remapping labels)...")
    prepare_whu_from_raw(raw_dir, data_dir)
    return data_dir


def prepare_whu_from_raw(raw_dir, output_dir="whu_building_dataset"):
    """Prepare WHU dataset from raw PNG files to TIF format.

    Converts PNG images to TIF and remaps labels from 0/255 to 0/1.

    Args:
        raw_dir: Path to the raw WHU dataset (containing train/val/test dirs
            with image/ and label/ subdirectories of PNG files).
        output_dir: Output directory for the prepared dataset.

    Returns:
        str: Path to the prepared dataset directory.
    """
    import rasterio
    from rasterio.transform import from_bounds

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        images_dir = os.path.join(output_dir, split, "images")
        labels_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Find image directory (handle both 'Image'/'image' naming conventions)
        img_dir_name = None
        for candidate in ["Image", "image", "images"]:
            if os.path.isdir(os.path.join(split_dir, candidate)):
                img_dir_name = candidate
                break
        if img_dir_name is None:
            print(f"  No image directory found in {split_dir}, skipping")
            continue

        # Find mask/label directory
        lbl_dir_name = None
        for candidate in ["Mask", "mask", "label", "labels"]:
            if os.path.isdir(os.path.join(split_dir, candidate)):
                lbl_dir_name = candidate
                break
        if lbl_dir_name is None:
            print(f"  No label directory found in {split_dir}, skipping")
            continue

        # Process images
        image_files = sorted(
            glob.glob(os.path.join(split_dir, img_dir_name, "*.png"))
            + glob.glob(os.path.join(split_dir, img_dir_name, "*.tif"))
        )
        label_files = sorted(
            glob.glob(os.path.join(split_dir, lbl_dir_name, "*.png"))
            + glob.glob(os.path.join(split_dir, lbl_dir_name, "*.tif"))
        )

        print(
            f"Processing {split}: {len(image_files)} images, {len(label_files)} labels"
        )

        for img_path, lbl_path in tqdm(
            zip(image_files, label_files), total=len(image_files), desc=split
        ):
            basename = os.path.splitext(os.path.basename(img_path))[0] + ".tif"

            # Convert image PNG to TIF
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

            img_out = os.path.join(images_dir, basename)
            with rasterio.open(
                img_out,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=3,
                dtype="uint8",
                compress="lzw",
            ) as dst:
                for band in range(3):
                    dst.write(img[:, :, band], band + 1)

            # Convert label PNG to TIF with remapping (255 -> 1)
            lbl = np.array(Image.open(lbl_path))
            if lbl.ndim == 3:
                lbl = lbl[:, :, 0]
            # Remap: 255 -> 1, everything else -> 0
            lbl = (lbl > 127).astype(np.uint8)

            lbl_out = os.path.join(labels_dir, basename)
            with rasterio.open(
                lbl_out,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="uint8",
                compress="lzw",
            ) as dst:
                dst.write(lbl, 1)

    print(f"Dataset prepared at: {output_dir}")
    return output_dir


def train_model(
    data_dir,
    output_dir="whu_output",
    encoder_name="tu-convnext_base",
    architecture="unetplusplus",
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-4,
):
    """Train the building detection model.

    Args:
        data_dir: Path to the prepared WHU dataset.
        output_dir: Directory to save model outputs.
        encoder_name: Encoder backbone name.
        architecture: Segmentation architecture.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.

    Returns:
        str: Path to the trained model checkpoint.
    """
    train_images = os.path.join(data_dir, "train", "images")
    train_labels = os.path.join(data_dir, "train", "labels")

    if not os.path.exists(train_images):
        raise FileNotFoundError(
            f"Training images not found at {train_images}. "
            "Run with --prepare-raw to convert raw WHU data first."
        )

    print(f"Training images: {len(os.listdir(train_images))} tiles")
    print(f"Training labels: {len(os.listdir(train_labels))} tiles")

    geoai.train_timm_segmentation_model(
        images_dir=train_images,
        labels_dir=train_labels,
        output_dir=output_dir,
        encoder_name=encoder_name,
        architecture=architecture,
        encoder_weights="imagenet",
        num_channels=3,
        num_classes=2,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        val_split=0.2,
        verbose=True,
    )

    model_path = os.path.join(output_dir, "models", "last.ckpt")
    print(f"Model saved to: {model_path}")
    return model_path


def evaluate_model(
    data_dir,
    model_path,
    output_dir="whu_eval",
    encoder_name="tu-convnext_base",
    architecture="unetplusplus",
):
    """Evaluate the trained model on the test split.

    Args:
        data_dir: Path to the prepared WHU dataset.
        model_path: Path to the trained model checkpoint.
        output_dir: Directory to save evaluation outputs.
        encoder_name: Encoder backbone name.
        architecture: Segmentation architecture.
    """
    import rasterio

    test_images = os.path.join(data_dir, "test", "images")
    test_labels = os.path.join(data_dir, "test", "labels")

    if not os.path.exists(test_images):
        print("Test split not found, skipping evaluation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(test_images, "*.tif")))
    label_files = sorted(glob.glob(os.path.join(test_labels, "*.tif")))

    print(f"Evaluating on {len(image_files)} test tiles...")

    total_intersection = 0
    total_union = 0

    for img_path, lbl_path in tqdm(
        zip(image_files, label_files), total=len(image_files), desc="Evaluating"
    ):
        pred_path = os.path.join(output_dir, os.path.basename(img_path))

        geoai.timm_semantic_segmentation(
            input_path=img_path,
            output_path=pred_path,
            model_path=model_path,
            encoder_name=encoder_name,
            architecture=architecture,
            num_channels=3,
            num_classes=2,
            window_size=512,
            overlap=0,
            batch_size=1,
            quiet=True,
        )

        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        with rasterio.open(lbl_path) as src:
            label = src.read(1)

        pred_binary = (pred > 0).astype(np.uint8)
        label_binary = (label > 0).astype(np.uint8)

        intersection = np.logical_and(pred_binary, label_binary).sum()
        union = np.logical_or(pred_binary, label_binary).sum()

        total_intersection += intersection
        total_union += union

    iou = total_intersection / max(total_union, 1)
    print(f"\nTest IoU: {iou:.4f}")
    print(f"Test Dice: {2 * iou / (1 + iou):.4f}")


def push_to_hub(
    model_path,
    repo_id="giswqs/whu-building-unetplusplus-convnext-base",
    encoder_name="tu-convnext_base",
    architecture="unetplusplus",
):
    """Push the trained model to HuggingFace Hub.

    Args:
        model_path: Path to the trained model checkpoint.
        repo_id: HuggingFace repository ID.
        encoder_name: Encoder backbone name.
        architecture: Segmentation architecture.

    Returns:
        str: URL of the uploaded model.
    """
    url = geoai.push_timm_model_to_hub(
        model_path=model_path,
        repo_id=repo_id,
        encoder_name=encoder_name,
        architecture=architecture,
        num_channels=3,
        num_classes=2,
        commit_message="Upload WHU building detection model (ConvNeXt-Base + UNet++)",
        private=False,
    )
    print(f"Model uploaded to: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Train building detection model on WHU Building Dataset"
    )
    parser.add_argument(
        "--prepare-raw",
        type=str,
        default=None,
        help="Path to raw WHU dataset (PNG format) to convert to TIF.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="whu_building_dataset",
        help="Path to prepared dataset or output directory for preparation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="whu_output",
        help="Directory for model training outputs.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="tu-convnext_base",
        help="Encoder backbone (default: tu-convnext_base).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="unetplusplus",
        help="Segmentation architecture (default: unetplusplus).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="giswqs/whu-building-unetplusplus-convnext-base",
        help="HuggingFace repository ID for model upload.",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on test set after training."
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (useful if model already trained).",
    )

    args = parser.parse_args()

    # Step 1: Prepare dataset
    if args.prepare_raw:
        print("Preparing WHU dataset from raw PNG files...")
        prepare_whu_from_raw(args.prepare_raw, args.data_dir)
    elif not os.path.exists(args.data_dir):
        print("Downloading pre-processed WHU dataset from HuggingFace...")
        download_and_prepare_whu_dataset(args.data_dir)

    # Step 2: Train model
    if not args.skip_training:
        model_path = train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            encoder_name=args.encoder,
            architecture=args.architecture,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )
    else:
        model_path = os.path.join(args.output_dir, "models", "last.ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run training first."
            )

    # Step 3: Evaluate
    if args.evaluate:
        evaluate_model(
            data_dir=args.data_dir,
            model_path=model_path,
            output_dir=os.path.join(args.output_dir, "eval"),
            encoder_name=args.encoder,
            architecture=args.architecture,
        )

    # Step 4: Push to HuggingFace Hub
    if args.push_to_hub:
        push_to_hub(
            model_path=model_path,
            repo_id=args.repo_id,
            encoder_name=args.encoder,
            architecture=args.architecture,
        )

    print("Done!")


if __name__ == "__main__":
    main()
