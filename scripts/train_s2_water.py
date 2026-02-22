"""Training script for surface water detection using the Earth Surface Water Dataset.

This script downloads the Earth Surface Water Dataset (Sentinel-2), tiles the
variable-sized scenes, trains a EfficientNet-B4 + UNet++ semantic segmentation
model, evaluates it, and optionally pushes the model and dataset to
HuggingFace Hub.

Earth Surface Water Dataset:
    - Sentinel-2 L2A imagery with 6 spectral bands (B2, B3, B4, B8, B11, B12)
    - Variable-sized scenes (868x764 to 1351x1246 pixels)
    - Train: 64 scenes, Val: 31 scenes
    - Binary labels: 0 = background, 1 = water
    - Source: https://zenodo.org/records/5205674
    - Reference: https://doi.org/10.1016/j.jag.2021.102472

Usage:
    python scripts/train_s2_water.py
    python scripts/train_s2_water.py --push-to-hub --push-dataset

Requirements:
    pip install geoai-py timm segmentation-models-pytorch lightning huggingface-hub
"""

import argparse
import glob
import os
import tempfile

import numpy as np
from tqdm import tqdm

import geoai


def download_dataset(data_dir="s2_water_dataset"):
    """Download the Earth Surface Water Dataset.

    Downloads the dataset from HuggingFace and extracts it.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        str: Path to the dataset directory containing tra_scene/, tra_truth/,
            val_scene/, val_truth/.
    """
    url = "https://huggingface.co/datasets/giswqs/s2-water-dataset/resolve/main/dset-s2.zip"
    print("Downloading Earth Surface Water Dataset...")
    raw_path = geoai.download_file(url)
    print(f"Dataset downloaded to: {raw_path}")

    # Find the extracted directory containing tra_scene/val_scene
    dataset_dir = raw_path
    if os.path.isdir(os.path.join(raw_path, "dset-s2")):
        dataset_dir = os.path.join(raw_path, "dset-s2")

    return dataset_dir


def prepare_tiles(
    data_dir,
    tiles_dir="s2_water_tiles",
    tile_size=512,
    stride=128,
):
    """Tile variable-sized Sentinel-2 scenes into fixed-size training tiles.

    Args:
        data_dir: Path to the dataset directory with tra_scene/, tra_truth/,
            val_scene/, val_truth/.
        tiles_dir: Output directory for tiled data.
        tile_size: Tile size in pixels (square).
        stride: Step size between tiles.

    Returns:
        str: Path to the tiles directory.
    """
    os.makedirs(tiles_dir, exist_ok=True)

    splits = {
        "train": ("tra_scene", "tra_truth"),
        "val": ("val_scene", "val_truth"),
    }

    for split_name, (scene_dir, truth_dir) in splits.items():
        images_folder = os.path.join(data_dir, scene_dir)
        masks_folder = os.path.join(data_dir, truth_dir)

        if not os.path.exists(images_folder):
            print(f"Skipping {split_name}: {images_folder} not found")
            continue

        output_folder = os.path.join(tiles_dir, split_name)
        print(f"Tiling {split_name} split (tile_size={tile_size}, stride={stride})...")

        result = geoai.export_geotiff_tiles_batch(
            images_folder=images_folder,
            masks_folder=masks_folder,
            output_folder=output_folder,
            tile_size=tile_size,
            stride=stride,
            quiet=True,
        )

        num_tiles = result.get("total_tiles", "unknown")
        print(f"  {split_name}: {num_tiles} tiles created at {output_folder}")

    print(f"Tiling complete. Tiles saved to: {tiles_dir}")
    return tiles_dir


def train_model(
    tiles_dir,
    output_dir="s2_water_output",
    encoder_name="efficientnet-b4",
    architecture="unetplusplus",
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-4,
):
    """Train the surface water detection model.

    Args:
        tiles_dir: Path to the tiled dataset.
        output_dir: Directory to save model outputs.
        encoder_name: Encoder backbone name.
        architecture: Segmentation architecture.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.

    Returns:
        str: Path to the trained model checkpoint.
    """
    train_images = os.path.join(tiles_dir, "train", "images")
    train_labels = os.path.join(tiles_dir, "train", "masks")

    if not os.path.exists(train_images):
        raise FileNotFoundError(
            f"Training images not found at {train_images}. "
            "Run tiling step first (remove --skip-tiling flag)."
        )

    num_images = len(os.listdir(train_images))
    num_labels = len(os.listdir(train_labels))
    print(f"Training images: {num_images} tiles")
    print(f"Training labels: {num_labels} tiles")

    geoai.train_timm_segmentation_model(
        images_dir=train_images,
        labels_dir=train_labels,
        output_dir=output_dir,
        encoder_name=encoder_name,
        architecture=architecture,
        encoder_weights="imagenet",
        num_channels=6,
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
    output_dir="s2_water_eval",
    encoder_name="efficientnet-b4",
    architecture="unetplusplus",
):
    """Evaluate the trained model on the validation split.

    Runs inference on each validation scene and computes IoU and Dice scores
    against ground truth masks.

    Args:
        data_dir: Path to the dataset directory.
        model_path: Path to the trained model checkpoint.
        output_dir: Directory to save evaluation outputs.
        encoder_name: Encoder backbone name.
        architecture: Segmentation architecture.
    """
    import rasterio

    val_images = os.path.join(data_dir, "val_scene")
    val_labels = os.path.join(data_dir, "val_truth")

    if not os.path.exists(val_images):
        print("Validation split not found, skipping evaluation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(val_images, "*.tif")))
    label_files = sorted(glob.glob(os.path.join(val_labels, "*.tif")))

    print(f"Evaluating on {len(image_files)} validation scenes...")

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
            num_channels=6,
            num_classes=2,
            window_size=512,
            overlap=256,
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
    print(f"\nValidation IoU: {iou:.4f}")
    print(f"Validation Dice: {2 * iou / (1 + iou):.4f}")


def push_to_hub(
    model_path,
    repo_id="giswqs/s2-water-unetplusplus-efficientnet-b4",
    encoder_name="efficientnet-b4",
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
        num_channels=6,
        num_classes=2,
        commit_message="Upload S2 water detection model (EfficientNet-B4 + UNet++)",
        private=False,
    )
    print(f"Model uploaded to: {url}")
    return url


def push_dataset_to_hub(data_dir, repo_id="giswqs/s2-water-dataset"):
    """Push the Earth Surface Water Dataset to HuggingFace Hub.

    Creates a dataset repository and uploads the scene/truth directories
    along with a dataset card.

    Args:
        data_dir: Path to the dataset directory containing tra_scene/,
            tra_truth/, val_scene/, val_truth/.
        repo_id: HuggingFace dataset repository ID.

    Returns:
        str: URL of the uploaded dataset.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create dataset repo
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Dataset repository created: {repo_id}")

    # Create dataset card
    readme_content = """---
license: cc-by-4.0
task_categories:
  - image-segmentation
tags:
  - remote-sensing
  - sentinel-2
  - water-detection
  - semantic-segmentation
  - earth-observation
pretty_name: Earth Surface Water Dataset (Sentinel-2)
size_categories:
  - n<1K
---

# Earth Surface Water Dataset (Sentinel-2)

A dataset for training deep learning models to detect surface water from
Sentinel-2 satellite imagery.

## Dataset Description

- **Source**: [Zenodo](https://zenodo.org/records/5205674)
- **Paper**: [WatNet: A deep learning model for surface water detection](https://doi.org/10.1016/j.jag.2021.102472)
- **Author**: Xin Luo (Shenzhen University)
- **Code**: [WatNet GitHub](https://github.com/xinluo2018/WatNet)
- **License**: CC-BY-4.0

## Dataset Structure

```
├── tra_scene/    # 64 training Sentinel-2 scenes (6-band GeoTIFF, uint16)
├── tra_truth/    # 64 training ground truth masks (binary GeoTIFF, uint8)
├── val_scene/    # 31 validation Sentinel-2 scenes
└── val_truth/    # 31 validation ground truth masks
```

## Sentinel-2 Bands

The 6 bands included are:
| Band | Name | Wavelength (nm) | Resolution |
|------|------|-----------------|------------|
| B2   | Blue | 490             | 10m        |
| B3   | Green| 560             | 10m        |
| B4   | Red  | 665             | 10m        |
| B8   | NIR  | 842             | 10m        |
| B11  | SWIR1| 1610            | 20m        |
| B12  | SWIR2| 2190            | 20m        |

## Labels

Binary classification:
- **0**: Background (non-water)
- **1**: Water

## Image Properties

- **Format**: GeoTIFF
- **Image dtype**: uint16
- **Label dtype**: uint8
- **Dimensions**: Variable (868×764 to 1351×1246 pixels)
- **Total**: 95 scene-label pairs (64 train + 31 validation)

## Citation

```bibtex
@ARTICLE{Luo2021-te,
  title     = "{An applicable and automatic method for earth surface water
               mapping based on multispectral images}",
  author    = "Luo, Xin and Tong, Xiaohua and Hu, Zhongwen",
  journal   = "International Journal of Applied Earth Observation and
               Geoinformation",
  publisher = "Elsevier BV",
  volume    =  103,
  pages     =  102472,
  year      =  2021,
  url       = "http://dx.doi.org/10.1016/j.jag.2021.102472",
  doi       = "10.1016/j.jag.2021.102472",
  issn      = "1569-8432,1872-826X",
}
```
"""

    # Write README to a temp file and upload
    readme_path = os.path.join(tempfile.gettempdir(), "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    # Upload dataset directories
    print("Uploading dataset files (this may take a while)...")
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload Earth Surface Water Dataset (Sentinel-2)",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Dataset uploaded to: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Train surface water detection model on Earth Surface Water Dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="s2_water_dataset",
        help="Path to the dataset directory (with tra_scene/, val_scene/, etc.).",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="s2_water_tiles",
        help="Directory for tiled training data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="s2_water_output",
        help="Directory for model training outputs.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride between tiles (default: 128).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="efficientnet-b4",
        help="Encoder backbone (default: efficientnet-b4).",
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
        default="giswqs/s2-water-unetplusplus-efficientnet-b4",
        help="HuggingFace repository ID for model upload.",
    )
    parser.add_argument(
        "--push-dataset",
        action="store_true",
        help="Push dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="giswqs/s2-water-dataset",
        help="HuggingFace repository ID for dataset upload.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on validation set after training.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (useful if model already trained).",
    )
    parser.add_argument(
        "--skip-tiling",
        action="store_true",
        help="Skip tiling (useful if tiles already created).",
    )

    args = parser.parse_args()

    # Step 1: Download dataset if not present
    if not os.path.exists(args.data_dir):
        print("Downloading dataset...")
        args.data_dir = download_dataset(args.data_dir)

    # Step 2: Push dataset to HuggingFace Hub
    if args.push_dataset:
        push_dataset_to_hub(
            data_dir=args.data_dir,
            repo_id=args.dataset_repo_id,
        )

    # Step 3: Tile scenes into fixed-size tiles
    if not args.skip_tiling:
        prepare_tiles(
            data_dir=args.data_dir,
            tiles_dir=args.tiles_dir,
            tile_size=args.tile_size,
            stride=args.stride,
        )

    # Step 4: Train model
    if not args.skip_training:
        model_path = train_model(
            tiles_dir=args.tiles_dir,
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

    # Step 5: Evaluate
    if args.evaluate:
        evaluate_model(
            data_dir=args.data_dir,
            model_path=model_path,
            output_dir=os.path.join(args.output_dir, "eval"),
            encoder_name=args.encoder,
            architecture=args.architecture,
        )

    # Step 6: Push model to HuggingFace Hub
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
