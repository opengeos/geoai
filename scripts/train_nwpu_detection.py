"""Training script for multi-class object detection using NWPU-VHR-10.

This script downloads the NWPU-VHR-10 dataset, trains a Mask R-CNN model
for 10-class object detection, evaluates it, and optionally pushes the
trained model to HuggingFace Hub.

NWPU-VHR-10 Dataset:
    - 800 VHR remote sensing images (650 positive, 150 negative)
    - 10 object classes: airplane, ship, storage_tank, baseball_diamond,
      tennis_court, basketball_court, ground_track_field, harbor, bridge,
      vehicle
    - 3,775 annotated instances in COCO format
    - Reference: https://github.com/chaozhong2010/VHR-10_dataset_coco

Usage:
    python scripts/train_nwpu_detection.py
    python scripts/train_nwpu_detection.py --epochs 50 --batch-size 8
    python scripts/train_nwpu_detection.py --evaluate
    python scripts/train_nwpu_detection.py --push-to-hub --repo-id user/model-name

Requirements:
    pip install geoai-py
"""

import argparse
import os

import geoai


def download_dataset(data_dir="NWPU-VHR-10"):
    """Download and prepare the NWPU-VHR-10 dataset.

    Args:
        data_dir: Directory name for the dataset.

    Returns:
        dict: Dataset split information including paths and class names.
    """
    print("Downloading NWPU-VHR-10 dataset...")
    dataset_path = geoai.download_nwpu_vhr10(output_dir=data_dir)
    print(f"Dataset downloaded to: {dataset_path}")

    print("\nPreparing train/val splits...")
    splits = geoai.prepare_nwpu_vhr10(dataset_path, val_split=0.2, seed=42)
    return splits


def train_model(
    splits,
    output_dir="nwpu_output",
    batch_size=4,
    epochs=50,
    lr=0.005,
    num_workers=None,
):
    """Train a multi-class Mask R-CNN model on NWPU-VHR-10.

    Args:
        splits: Dataset split information from prepare_nwpu_vhr10.
        output_dir: Directory for model outputs.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        num_workers: Number of data loading workers.

    Returns:
        str: Path to the best model checkpoint.
    """
    print(f"\nTraining multi-class detector for {epochs} epochs...")
    print(f"  Classes: {splits['class_names'][1:]}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    model_path = geoai.train_multiclass_detector(
        images_dir=splits["images_dir"],
        annotations_path=splits["train_annotations"],
        output_dir=output_dir,
        class_names=splits["class_names"],
        num_channels=3,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=lr,
        val_split=0.15,
        seed=42,
        pretrained=True,
        num_workers=num_workers,
        verbose=True,
    )

    print(f"\nBest model saved to: {model_path}")
    return model_path


def evaluate_model(splits, model_path, output_dir="nwpu_output"):
    """Evaluate the trained model on the validation set.

    Args:
        splits: Dataset split information.
        model_path: Path to trained model weights.
        output_dir: Output directory for evaluation results.

    Returns:
        dict: Evaluation metrics.
    """
    print("\nEvaluating model on validation set...")

    metrics = geoai.evaluate_multiclass_detector(
        model_path=model_path,
        images_dir=splits["images_dir"],
        annotations_path=splits["val_annotations"],
        num_classes=splits["num_classes"],
        class_names=splits["class_names"][1:],  # Exclude background
        batch_size=4,
    )

    # Save metrics
    import json

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    return metrics


def push_to_hub(model_path, repo_id, output_dir="nwpu_output"):
    """Push the trained model to HuggingFace Hub.

    Args:
        model_path: Path to trained model weights.
        repo_id: HuggingFace Hub repository ID.
        output_dir: Directory containing model artifacts.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required for pushing to Hub.")
        print("Install with: pip install huggingface_hub")
        return

    print(f"\nPushing model to HuggingFace Hub: {repo_id}")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    # Upload model weights
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.pth",
        repo_id=repo_id,
        commit_message="Upload NWPU-VHR-10 Mask R-CNN model",
    )

    # Upload class info
    class_info_path = os.path.join(output_dir, "class_info.json")
    if os.path.exists(class_info_path):
        api.upload_file(
            path_or_fileobj=class_info_path,
            path_in_repo="class_info.json",
            repo_id=repo_id,
            commit_message="Upload class information",
        )

    # Upload training summary
    summary_path = os.path.join(output_dir, "training_summary.txt")
    if os.path.exists(summary_path):
        api.upload_file(
            path_or_fileobj=summary_path,
            path_in_repo="training_summary.txt",
            repo_id=repo_id,
            commit_message="Upload training summary",
        )

    # Upload eval metrics if available
    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    if os.path.exists(metrics_path):
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo="eval_metrics.json",
            repo_id=repo_id,
            commit_message="Upload evaluation metrics",
        )

    print(f"Model pushed to: https://huggingface.co/{repo_id}")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train multi-class object detection on NWPU-VHR-10"
    )
    parser.add_argument(
        "--data-dir",
        default="NWPU-VHR-10",
        help="Directory for the dataset (default: NWPU-VHR-10)",
    )
    parser.add_argument(
        "--output-dir",
        default="nwpu_output",
        help="Directory for model outputs (default: nwpu_output)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Initial learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: auto)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model after training",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        default="giswqs/nwpu-vhr10-maskrcnn",
        help="HuggingFace Hub repo ID (default: giswqs/nwpu-vhr10-maskrcnn)",
    )

    args = parser.parse_args()

    # Step 1: Download and prepare dataset
    splits = download_dataset(args.data_dir)

    # Step 2: Train model
    model_path = train_model(
        splits=splits,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
    )

    # Step 3: Evaluate (optional)
    if args.evaluate:
        evaluate_model(splits, model_path, args.output_dir)

    # Step 4: Push to Hub (optional)
    if args.push_to_hub:
        push_to_hub(model_path, args.repo_id, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
