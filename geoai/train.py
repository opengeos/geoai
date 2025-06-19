import glob
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.utils.data
import torchvision

# import torchvision.transforms as transforms
from rasterio.windows import Window
from skimage import measure
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from .utils import download_model_from_hf


# Additional imports for semantic segmentation
try:
    import segmentation_models_pytorch as smp
    from torch.nn import functional as F

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


def get_instance_segmentation_model(num_classes=2, num_channels=3, pretrained=True):
    """
    Get Mask R-CNN model with custom input channels and output classes.

    Args:
        num_classes (int): Number of output classes (including background).
        num_channels (int): Number of input channels (3 for RGB, 4 for RGBN).
        pretrained (bool): Whether to use pretrained backbone.

    Returns:
        torch.nn.Module: Mask R-CNN model with specified input channels and output classes.

    Raises:
        ValueError: If num_channels is less than 3.
    """
    # Validate num_channels
    if num_channels < 3:
        raise ValueError("num_channels must be at least 3")

    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        progress=True,
        weights=(
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        ),
    )

    # Modify transform if num_channels is different from 3
    if num_channels != 3:
        # Get the transform
        transform = model.transform

        # Default values are [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]
        # Calculate means and stds for additional channels
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]

        # Extend them to num_channels (use the mean value for additional channels)
        mean_of_means = sum(rgb_mean) / len(rgb_mean)
        mean_of_stds = sum(rgb_std) / len(rgb_std)

        # Create new lists with appropriate length
        transform.image_mean = rgb_mean + [mean_of_means] * (num_channels - 3)
        transform.image_std = rgb_std + [mean_of_stds] * (num_channels - 3)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Modify the first layer if num_channels is different from 3
    if num_channels != 3:
        original_layer = model.backbone.body.conv1
        model.backbone.body.conv1 = torch.nn.Conv2d(
            num_channels,
            original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=original_layer.bias is not None,
        )

        # Copy weights from the original 3 channels to the new layer
        with torch.no_grad():
            # Copy the weights for the first 3 channels
            model.backbone.body.conv1.weight[:, :3, :, :] = original_layer.weight

            # Initialize additional channels with the mean of the first 3 channels
            mean_weight = original_layer.weight.mean(dim=1, keepdim=True)
            for i in range(3, num_channels):
                model.backbone.body.conv1.weight[:, i : i + 1, :, :] = mean_weight

            # Copy bias if it exists
            if original_layer.bias is not None:
                model.backbone.body.conv1.bias = original_layer.bias

    return model


class ObjectDetectionDataset(Dataset):
    """Dataset for object detection from GeoTIFF images and labels."""

    def __init__(self, image_paths, label_paths, transforms=None, num_channels=None):
        """
        Initialize dataset.

        Args:
            image_paths (list): List of paths to image GeoTIFF files.
            label_paths (list): List of paths to label GeoTIFF files.
            transforms (callable, optional): Transformations to apply to images and masks.
            num_channels (int, optional): Number of channels to use from images. If None,
                auto-detected from the first image.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms

        # Auto-detect the number of channels if not specified
        if num_channels is None:
            with rasterio.open(self.image_paths[0]) as src:
                self.num_channels = src.count
        else:
            self.num_channels = num_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        with rasterio.open(self.image_paths[idx]) as src:
            # Read as [C, H, W] format
            image = src.read().astype(np.float32)

            # Normalize image to [0, 1] range
            image = image / 255.0

            # Handle different number of channels
            if image.shape[0] > self.num_channels:
                image = image[
                    : self.num_channels
                ]  # Keep only first 4 bands if more exist
            elif image.shape[0] < self.num_channels:
                # Pad with zeros if less than 4 bands
                padded = np.zeros(
                    (self.num_channels, image.shape[1], image.shape[2]),
                    dtype=np.float32,
                )
                padded[: image.shape[0]] = image
                image = padded

            # Convert to CHW tensor
            image = torch.as_tensor(image, dtype=torch.float32)

        # Load label mask
        with rasterio.open(self.label_paths[idx]) as src:
            label_mask = src.read(1)
            binary_mask = (label_mask > 0).astype(np.uint8)

        # Find all building instances using connected components
        labeled_mask, num_instances = measure.label(
            binary_mask, return_num=True, connectivity=2
        )

        # Create list to hold masks for each building instance
        masks = []
        boxes = []
        labels = []

        for i in range(1, num_instances + 1):
            # Create mask for this instance
            instance_mask = (labeled_mask == i).astype(np.uint8)

            # Calculate area and filter out tiny instances (noise)
            area = instance_mask.sum()
            if area < 10:  # Minimum area threshold
                continue

            # Find bounding box coordinates
            pos = np.where(instance_mask)
            if len(pos[0]) == 0:  # Skip if mask is empty
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            # Add small padding to ensure the mask is within the box
            xmin = max(0, xmin - 1)
            ymin = max(0, ymin - 1)
            xmax = min(binary_mask.shape[1] - 1, xmax + 1)
            ymax = min(binary_mask.shape[0] - 1, ymax + 1)

            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(instance_mask)
            labels.append(1)  # 1 for building class

        # Handle case with no valid instances
        if len(boxes) == 0:
            # Create a dummy target with minimal required fields
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0), dtype=torch.int64),
                "masks": torch.zeros(
                    (0, binary_mask.shape[0], binary_mask.shape[1]), dtype=torch.uint8
                ),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0), dtype=torch.float32),
                "iscrowd": torch.zeros((0), dtype=torch.int64),
            }
        else:
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

            # Calculate area of boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Prepare target dictionary
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([idx]),
                "area": area,
                "iscrowd": torch.zeros_like(labels),  # Assume no crowd instances
            }

        # Apply transforms if specified
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class Compose:
    """Custom compose transform that works with image and target."""

    def __init__(self, transforms):
        """
        Initialize compose transform.

        Args:
            transforms (list): List of transforms to apply.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert numpy.ndarray to tensor."""

    def __call__(self, image, target):
        """
        Apply transform to image and target.

        Args:
            image (torch.Tensor): Input image.
            target (dict): Target annotations.

        Returns:
            tuple: Transformed image and target.
        """
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip transform."""

    def __init__(self, prob=0.5):
        """
        Initialize random horizontal flip.

        Args:
            prob (float): Probability of applying the flip.
        """
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = torch.flip(image, dims=[2])  # Flip along width dimension

            # Flip masks
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], dims=[2])

            # Update boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                width = image.shape[2]
                boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
                target["boxes"] = boxes

        return image, target


def get_transform(train):
    """
    Get transforms for data augmentation.

    Args:
        train (bool): Whether to include training-specific transforms.

    Returns:
        Compose: Composed transforms.
    """
    transforms = []
    transforms.append(ToTensor())

    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


def collate_fn(batch):
    """
    Custom collate function for batching samples.

    Args:
        batch (list): List of (image, target) tuples.

    Returns:
        tuple: Tuple of images and targets.
    """
    return tuple(zip(*batch))


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq=10, verbose=True
):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        print_freq (int): How often to print progress.
        verbose (bool): Whether to print detailed progress.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0

    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Track loss
        total_loss += losses.item()

        # Print progress
        if i % print_freq == 0:
            elapsed_time = time.time() - start_time
            if verbose:
                print(
                    f"Epoch: {epoch}, Batch: {i}/{len(data_loader)}, Loss: {losses.item():.4f}, Time: {elapsed_time:.2f}s"
                )
            start_time = time.time()

    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to evaluate on.

    Returns:
        dict: Evaluation metrics including loss and IoU.
    """
    model.eval()

    # Initialize metrics
    total_loss = 0
    iou_scores = []

    with torch.no_grad():
        for images, targets in data_loader:
            # Move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # During evaluation, Mask R-CNN directly returns predictions, not losses
            # So we'll only get loss when we provide targets explicitly
            if len(targets) > 0:
                try:
                    # Try to get loss dict (this works in some implementations)
                    loss_dict = model(images, targets)
                    if isinstance(loss_dict, dict):
                        losses = sum(loss for loss in loss_dict.values())
                        total_loss += losses.item()
                except Exception as e:
                    print(f"Warning: Could not compute loss during evaluation: {e}")
                    # If we can't compute loss, we'll just focus on IoU
                    pass

            # Get predictions
            outputs = model(images)

            # Calculate IoU for each image
            for i, output in enumerate(outputs):
                if len(output["masks"]) == 0 or len(targets[i]["masks"]) == 0:
                    continue

                # Convert predicted masks to binary (threshold at 0.5)
                pred_masks = (output["masks"].squeeze(1) > 0.5).float()

                # Combine all instance masks into a single binary mask
                pred_combined = (
                    torch.max(pred_masks, dim=0)[0]
                    if pred_masks.shape[0] > 0
                    else torch.zeros_like(targets[i]["masks"][0])
                )
                target_combined = (
                    torch.max(targets[i]["masks"], dim=0)[0]
                    if targets[i]["masks"].shape[0] > 0
                    else torch.zeros_like(pred_combined)
                )

                # Calculate IoU
                intersection = (pred_combined * target_combined).sum().item()
                union = ((pred_combined + target_combined) > 0).sum().item()

                if union > 0:
                    iou = intersection / union
                    iou_scores.append(iou)

    # Calculate metrics
    avg_loss = total_loss / len(data_loader) if total_loss > 0 else float("inf")
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    return {"loss": avg_loss, "IoU": avg_iou}


def visualize_predictions(model, dataset, device, num_samples=5, output_dir=None):
    """
    Visualize model predictions.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): Dataset to visualize.
        device (torch.device): Device to run inference on.
        num_samples (int): Number of samples to visualize.
        output_dir (str, optional): Directory to save visualizations. If None,
            visualizations are displayed but not saved.
    """
    model.eval()

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        # Get image and target
        image, target = dataset[idx]

        # Convert to device and add batch dimension
        image = image.to(device)
        image_batch = [image]

        # Get prediction
        with torch.no_grad():
            output = model(image_batch)[0]

        # Convert image from CHW to HWC for display (first 3 bands as RGB)
        rgb_image = image[:3].cpu().numpy()
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        rgb_image = np.clip(rgb_image, 0, 1)  # Ensure values are in [0,1]

        # Create binary ground truth mask (combine all instances)
        gt_masks = target["masks"].cpu().numpy()
        gt_combined = (
            np.max(gt_masks, axis=0)
            if len(gt_masks) > 0
            else np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
        )

        # Create binary prediction mask (combine all instances with score > 0.5)
        pred_masks = output["masks"].cpu().numpy()
        pred_scores = output["scores"].cpu().numpy()
        high_conf_indices = pred_scores > 0.5

        pred_combined = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        if np.any(high_conf_indices):
            for mask in pred_masks[high_conf_indices]:
                # Apply threshold to each predicted mask
                binary_mask = (mask[0] > 0.5).astype(np.float32)
                # Combine with existing masks
                pred_combined = np.maximum(pred_combined, binary_mask)

        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Show RGB image
        axs[0].imshow(rgb_image)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        # Show prediction
        axs[1].imshow(pred_combined, cmap="viridis")
        axs[1].set_title(f"Predicted Buildings: {np.sum(high_conf_indices)} instances")
        axs[1].axis("off")

        # Show ground truth
        axs[2].imshow(gt_combined, cmap="viridis")
        axs[2].set_title(f"Ground Truth: {len(gt_masks)} instances")
        axs[2].axis("off")

        plt.tight_layout()

        # Save or show
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"prediction_{idx}.png"))
            plt.close()
        else:
            plt.show()


def train_MaskRCNN_model(
    images_dir,
    labels_dir,
    output_dir,
    num_channels=3,
    model=None,
    pretrained=True,
    pretrained_model_path=None,
    batch_size=4,
    num_epochs=10,
    learning_rate=0.005,
    seed=42,
    val_split=0.2,
    visualize=False,
    resume_training=False,
    print_freq=10,
    verbose=True,
):
    """Train and evaluate Mask R-CNN model for instance segmentation.

    This function trains a Mask R-CNN model for instance segmentation using the
    provided dataset. It supports loading a pretrained model to either initialize
    the backbone or to continue training from a specific checkpoint.

    Args:
        images_dir (str): Directory containing image GeoTIFF files.
        labels_dir (str): Directory containing label GeoTIFF files.
        output_dir (str): Directory to save model checkpoints and results.
        num_channels (int, optional): Number of input channels. If None, auto-detected.
            Defaults to 3.
        model (torch.nn.Module, optional): Predefined model. If None, a new model is created.
        pretrained (bool): Whether to use pretrained backbone. This is ignored if
            pretrained_model_path is provided. Defaults to True.
        pretrained_model_path (str, optional): Path to a .pth file to load as a
            pretrained model for continued training. Defaults to None.
        batch_size (int): Batch size for training. Defaults to 4.
        num_epochs (int): Number of training epochs. Defaults to 10.
        learning_rate (float): Initial learning rate. Defaults to 0.005.
        seed (int): Random seed for reproducibility. Defaults to 42.
        val_split (float): Fraction of data to use for validation (0-1). Defaults to 0.2.
        visualize (bool): Whether to generate visualizations of model predictions.
            Defaults to False.
        resume_training (bool): If True and pretrained_model_path is provided,
            will try to load optimizer and scheduler states as well. Defaults to False.
        print_freq (int): Frequency of printing training progress. Defaults to 10.
        verbose (bool): If True, prints detailed training progress. Defaults to True.
    Returns:
        None: Model weights are saved to output_dir.

    Raises:
        FileNotFoundError: If pretrained_model_path is provided but file doesn't exist.
        RuntimeError: If there's an issue loading the pretrained model.
    """

    import datetime

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Get all image and label files
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".tif")
        ]
    )
    label_files = sorted(
        [
            os.path.join(labels_dir, f)
            for f in os.listdir(labels_dir)
            if f.endswith(".tif")
        ]
    )

    print(f"Found {len(image_files)} image files and {len(label_files)} label files")

    # Ensure matching files
    if len(image_files) != len(label_files):
        print("Warning: Number of image files and label files don't match!")
        # Find matching files by basename
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

    # Split data into train and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=val_split, random_state=seed
    )

    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)} images")

    # Create datasets
    train_dataset = ObjectDetectionDataset(
        train_imgs, train_labels, transforms=get_transform(train=True)
    )
    val_dataset = ObjectDetectionDataset(
        val_imgs, val_labels, transforms=get_transform(train=False)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Initialize model (2 classes: background and building)
    if model is None:
        model = get_instance_segmentation_model(
            num_classes=2, num_channels=num_channels, pretrained=pretrained
        )
    model.to(device)

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )

    # Set up learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # Initialize training variables
    start_epoch = 0
    best_iou = 0

    # Load pretrained model if provided
    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(
                f"Pretrained model file not found: {pretrained_model_path}"
            )

        print(f"Loading pretrained model from: {pretrained_model_path}")
        try:
            # Check if it's a full checkpoint or just model weights
            checkpoint = torch.load(pretrained_model_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # It's a checkpoint with extra information
                model.load_state_dict(checkpoint["model_state_dict"])

                if resume_training:
                    # Resume from checkpoint
                    start_epoch = checkpoint.get("epoch", 0) + 1
                    best_iou = checkpoint.get("best_iou", 0)

                    if "optimizer_state_dict" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                    if "scheduler_state_dict" in checkpoint:
                        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    print(f"Resuming training from epoch {start_epoch}")
                    print(f"Previous best IoU: {best_iou:.4f}")
            else:
                # Assume it's just the model weights
                model.load_state_dict(checkpoint)

            print("Pretrained model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {str(e)}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq, verbose
        )

        # Update learning rate
        lr_scheduler.step()

        # Evaluate
        eval_metrics = evaluate(model, val_loader, device)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {eval_metrics['loss']:.4f}, Val IoU: {eval_metrics['IoU']:.4f}"
        )

        # Save best model
        if eval_metrics["IoU"] > best_iou:
            best_iou = eval_metrics["IoU"]
            print(f"Saving best model with IoU: {best_iou:.4f}")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_iou": best_iou,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # Save full checkpoint of final state
    torch.save(
        {
            "epoch": num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_iou": best_iou,
        },
        os.path.join(output_dir, "final_checkpoint.pth"),
    )

    # Load best model for evaluation and visualization
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))

    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)
    print(
        f"Final Evaluation - Loss: {final_metrics['loss']:.4f}, IoU: {final_metrics['IoU']:.4f}"
    )

    # Visualize results
    if visualize:
        print("Generating visualizations...")
        visualize_predictions(
            model,
            val_dataset,
            device,
            num_samples=5,
            output_dir=os.path.join(output_dir, "visualizations"),
        )

    # Save training summary
    with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
        f.write(
            f"Training completed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Best validation IoU: {best_iou:.4f}\n")
        f.write(f"Final validation IoU: {final_metrics['IoU']:.4f}\n")
        f.write(f"Final validation loss: {final_metrics['loss']:.4f}\n")

        if pretrained_model_path:
            f.write(f"Started from pretrained model: {pretrained_model_path}\n")
            if resume_training:
                f.write(f"Resumed training from epoch {start_epoch}\n")

    print(f"Training complete! Trained model saved to {output_dir}")


def inference_on_geotiff(
    model,
    geotiff_path,
    output_path,
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    num_channels=3,
    device=None,
    **kwargs,
):
    """
    Perform inference on a large GeoTIFF using a sliding window approach with improved blending.

    Args:
        model (torch.nn.Module): Trained model for inference.
        geotiff_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save output mask GeoTIFF.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        confidence_threshold (float): Confidence threshold for predictions (0-1).
        batch_size (int): Batch size for inference.
        num_channels (int): Number of channels to use from the input image.
        device (torch.device, optional): Device to run inference on. If None, uses CUDA if available.
        **kwargs: Additional arguments.

    Returns:
        tuple: Tuple containing output path and inference time in seconds.
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # Put model in evaluation mode
    model.to(device)
    model.eval()

    # Open the GeoTIFF
    with rasterio.open(geotiff_path) as src:
        # Read metadata
        meta = src.meta
        height = src.height
        width = src.width

        # Update metadata for output raster
        out_meta = meta.copy()
        out_meta.update(
            {"count": 1, "dtype": "uint8"}  # Single band for mask  # Binary mask
        )

        # We'll use two arrays:
        # 1. For accumulating predictions
        pred_accumulator = np.zeros((height, width), dtype=np.float32)
        # 2. For tracking how many predictions contribute to each pixel
        count_accumulator = np.zeros((height, width), dtype=np.float32)

        # Calculate the number of windows needed to cover the entire image
        steps_y = math.ceil((height - overlap) / (window_size - overlap))
        steps_x = math.ceil((width - overlap) / (window_size - overlap))

        # Ensure we cover the entire image
        last_y = height - window_size
        last_x = width - window_size

        total_windows = steps_y * steps_x
        print(
            f"Processing {total_windows} windows with size {window_size}x{window_size} and overlap {overlap}..."
        )

        # Create progress bar
        pbar = tqdm(total=total_windows)

        # Process in batches
        batch_inputs = []
        batch_positions = []
        batch_count = 0

        start_time = time.time()

        # Slide window over the image - make sure we cover the entire image
        for i in range(steps_y + 1):  # +1 to ensure we reach the edge
            y = min(i * (window_size - overlap), last_y)
            y = max(0, y)  # Prevent negative indices

            if y > last_y and i > 0:  # Skip if we've already covered the entire height
                continue

            for j in range(steps_x + 1):  # +1 to ensure we reach the edge
                x = min(j * (window_size - overlap), last_x)
                x = max(0, x)  # Prevent negative indices

                if (
                    x > last_x and j > 0
                ):  # Skip if we've already covered the entire width
                    continue

                # Read window
                window = src.read(window=Window(x, y, window_size, window_size))

                # Check if window is valid
                if window.shape[1] != window_size or window.shape[2] != window_size:
                    # This can happen at image edges - adjust window size
                    current_height = window.shape[1]
                    current_width = window.shape[2]
                    if current_height == 0 or current_width == 0:
                        continue  # Skip empty windows
                else:
                    current_height = window_size
                    current_width = window_size

                # Normalize and prepare input
                image = window.astype(np.float32) / 255.0

                # Handle different number of bands
                if image.shape[0] > num_channels:
                    image = image[:num_channels]
                elif image.shape[0] < num_channels:
                    padded = np.zeros(
                        (num_channels, current_height, current_width), dtype=np.float32
                    )
                    padded[: image.shape[0]] = image
                    image = padded

                # Convert to tensor
                image_tensor = torch.tensor(image, device=device)

                # Add to batch
                batch_inputs.append(image_tensor)
                batch_positions.append((y, x, current_height, current_width))
                batch_count += 1

                # Process batch when it reaches the batch size or at the end
                if batch_count == batch_size or (i == steps_y and j == steps_x):
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(batch_inputs)

                    # Process each output in the batch
                    for idx, output in enumerate(outputs):
                        y_pos, x_pos, h, w = batch_positions[idx]

                        # Create weight matrix that gives higher weight to center pixels
                        # This helps with smooth blending at boundaries
                        y_grid, x_grid = np.mgrid[0:h, 0:w]

                        # Calculate distance from each edge
                        dist_from_left = x_grid
                        dist_from_right = w - x_grid - 1
                        dist_from_top = y_grid
                        dist_from_bottom = h - y_grid - 1

                        # Combine distances (minimum distance to any edge)
                        edge_distance = np.minimum.reduce(
                            [
                                dist_from_left,
                                dist_from_right,
                                dist_from_top,
                                dist_from_bottom,
                            ]
                        )

                        # Convert to weight (higher weight for center pixels)
                        # Normalize to [0, 1]
                        edge_distance = np.minimum(edge_distance, overlap / 2)
                        weight = edge_distance / (overlap / 2)

                        # Get masks for predictions above threshold
                        if len(output["scores"]) > 0:
                            # Get all instances that meet confidence threshold
                            keep = output["scores"] > confidence_threshold
                            masks = output["masks"][keep].squeeze(1)

                            # Combine all instances into one mask
                            if len(masks) > 0:
                                combined_mask = torch.max(masks, dim=0)[0] > 0.5
                                combined_mask = (
                                    combined_mask.cpu().numpy().astype(np.float32)
                                )

                                # Apply weight to prediction
                                weighted_pred = combined_mask * weight

                                # Add to accumulators
                                pred_accumulator[
                                    y_pos : y_pos + h, x_pos : x_pos + w
                                ] += weighted_pred
                                count_accumulator[
                                    y_pos : y_pos + h, x_pos : x_pos + w
                                ] += weight

                    # Reset batch
                    batch_inputs = []
                    batch_positions = []
                    batch_count = 0

                    # Update progress bar
                    pbar.update(len(outputs))

        # Close progress bar
        pbar.close()

        # Calculate final mask by dividing accumulated predictions by counts
        # Handle division by zero
        mask = np.zeros((height, width), dtype=np.uint8)
        valid_pixels = count_accumulator > 0
        if np.any(valid_pixels):
            # Average predictions where we have data
            mask[valid_pixels] = (
                pred_accumulator[valid_pixels] / count_accumulator[valid_pixels] > 0.5
            ).astype(np.uint8)

        # Record time
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")

        # Save output
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mask, 1)

        print(f"Saved prediction to {output_path}")

        return output_path, inference_time


def object_detection(
    input_path,
    output_path,
    model_path,
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    num_channels=3,
    model=None,
    pretrained=True,
    device=None,
    **kwargs,
):
    """
    Perform object detection on a GeoTIFF using a pre-trained Mask R-CNN model.

    Args:
        input_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save output mask GeoTIFF.
        model_path (str): Path to trained model weights.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        confidence_threshold (float): Confidence threshold for predictions (0-1).
        batch_size (int): Batch size for inference.
        num_channels (int): Number of channels in the input image and model.
        model (torch.nn.Module, optional): Predefined model. If None, a new model is created.
        pretrained (bool): Whether to use pretrained backbone for model loading.
        device (torch.device, optional): Device to run inference on. If None, uses CUDA if available.
        **kwargs: Additional arguments passed to inference_on_geotiff.

    Returns:
        None: Output mask is saved to output_path.
    """
    # Load your trained model
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if model is None:
        model = get_instance_segmentation_model(
            num_classes=2, num_channels=num_channels, pretrained=pretrained
        )

    if not os.path.exists(model_path):
        try:
            model_path = download_model_from_hf(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    inference_on_geotiff(
        model=model,
        geotiff_path=input_path,
        output_path=output_path,
        window_size=window_size,  # Adjust based on your model and memory
        overlap=overlap,  # Overlap to avoid edge artifacts
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,  # Adjust based on your GPU memory
        num_channels=num_channels,
        device=device,
        **kwargs,
    )


def object_detection_batch(
    input_paths,
    output_dir,
    model_path,
    filenames=None,
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    model=None,
    num_channels=3,
    pretrained=True,
    device=None,
    **kwargs,
):
    """
    Perform object detection on a GeoTIFF using a pre-trained Mask R-CNN model.

    Args:
        input_paths (str or list): Path(s) to input GeoTIFF file(s). If a directory is provided,
            all .tif files in that directory will be processed.
        output_dir (str): Directory to save output mask GeoTIFF files.
        model_path (str): Path to trained model weights.
        filenames (list, optional): List of output filenames. If None, defaults to
            "<input_filename>_mask.tif" for each input file.
            If provided, must match the number of input files.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        confidence_threshold (float): Confidence threshold for predictions (0-1).
        batch_size (int): Batch size for inference.
        num_channels (int): Number of channels in the input image and model.
        model (torch.nn.Module, optional): Predefined model. If None, a new model is created.
        pretrained (bool): Whether to use pretrained backbone for model loading.
        device (torch.device, optional): Device to run inference on. If None, uses CUDA if available.
        **kwargs: Additional arguments passed to inference_on_geotiff.

    Returns:
        None: Output mask is saved to output_path.
    """
    # Load your trained model
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if model is None:
        model = get_instance_segmentation_model(
            num_classes=2, num_channels=num_channels, pretrained=pretrained
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        try:
            model_path = download_model_from_hf(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if isinstance(input_paths, str) and (not input_paths.endswith(".tif")):
        files = glob.glob(os.path.join(input_paths, "*.tif"))
        files.sort()
    elif isinstance(input_paths, str):
        files = [input_paths]

    if filenames is None:
        filenames = [
            os.path.join(output_dir, os.path.basename(f).replace(".tif", "_mask.tif"))
            for f in files
        ]
    else:
        if len(filenames) != len(files):
            raise ValueError("Number of filenames must match number of input files.")

    for index, file in enumerate(files):
        print(f"Processing file {index + 1}/{len(files)}: {file}")
        inference_on_geotiff(
            model=model,
            geotiff_path=file,
            output_path=filenames[index],
            window_size=window_size,  # Adjust based on your model and memory
            overlap=overlap,  # Overlap to avoid edge artifacts
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,  # Adjust based on your GPU memory
            num_channels=num_channels,
            device=device,
            **kwargs,
        )


class SemanticSegmentationDataset(Dataset):
    """Dataset for semantic segmentation from GeoTIFF images and labels."""

    def __init__(self, image_paths, label_paths, transforms=None, num_channels=None):
        """
        Initialize dataset for semantic segmentation.

        Args:
            image_paths (list): List of paths to image GeoTIFF files.
            label_paths (list): List of paths to label GeoTIFF files.
            transforms (callable, optional): Transformations to apply to images and masks.
            num_channels (int, optional): Number of channels to use from images. If None,
                auto-detected from the first image.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms

        # Auto-detect the number of channels if not specified
        if num_channels is None:
            with rasterio.open(self.image_paths[0]) as src:
                self.num_channels = src.count
        else:
            self.num_channels = num_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        with rasterio.open(self.image_paths[idx]) as src:
            # Read as [C, H, W] format
            image = src.read().astype(np.float32)

            # Normalize image to [0, 1] range
            image = image / 255.0

            # Handle different number of channels
            if image.shape[0] > self.num_channels:
                image = image[: self.num_channels]  # Keep only specified bands
            elif image.shape[0] < self.num_channels:
                # Pad with zeros if less than specified bands
                padded = np.zeros(
                    (self.num_channels, image.shape[1], image.shape[2]),
                    dtype=np.float32,
                )
                padded[: image.shape[0]] = image
                image = padded

            # Convert to CHW tensor
            image = torch.as_tensor(image, dtype=torch.float32)

        # Load label mask
        with rasterio.open(self.label_paths[idx]) as src:
            label_mask = src.read(1).astype(np.int64)
            # Keep original class values for multi-class segmentation
            # No conversion to binary - preserve all class labels

        # Convert to tensor
        mask = torch.as_tensor(label_mask, dtype=torch.long)

        # Apply transforms if specified
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask


class SemanticTransforms:
    """Custom transforms for semantic segmentation."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class SemanticToTensor:
    """Convert numpy.ndarray to tensor for semantic segmentation."""

    def __call__(self, image, mask):
        return image, mask


class SemanticRandomHorizontalFlip:
    """Random horizontal flip transform for semantic segmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            # Flip image and mask along width dimension
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        return image, mask


def get_semantic_transform(train):
    """
    Get transforms for semantic segmentation data augmentation.

    Args:
        train (bool): Whether to include training-specific transforms.

    Returns:
        SemanticTransforms: Composed transforms.
    """
    transforms = []
    transforms.append(SemanticToTensor())

    if train:
        transforms.append(SemanticRandomHorizontalFlip(0.5))

    return SemanticTransforms(transforms)


def get_smp_model(
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation=None,
    **kwargs,
):
    """
    Get a segmentation model from segmentation-models-pytorch using the generic create_model function.

    Args:
        architecture (str): Model architecture (e.g., 'unet', 'deeplabv3', 'deeplabv3plus', 'fpn',
            'pspnet', 'linknet', 'manet', 'pan', 'upernet', etc.). Case insensitive.
        encoder_name (str): Encoder backbone name (e.g., 'resnet34', 'efficientnet-b0', 'mit_b0', etc.).
        encoder_weights (str): Encoder weights ('imagenet' or None).
        in_channels (int): Number of input channels.
        classes (int): Number of output classes.
        activation (str): Activation function for output layer.
        **kwargs: Additional arguments passed to smp.create_model().

    Returns:
        torch.nn.Module: Segmentation model.

    Note:
        This function uses smp.create_model() which supports all architectures available in
        segmentation-models-pytorch, making it future-proof for new model additions.
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation-models-pytorch is not installed. "
            "Please install it with: pip install segmentation-models-pytorch"
        )

    try:
        # Use the generic create_model function - supports all SMP architectures
        model = smp.create_model(
            arch=architecture,  # Case insensitive
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

        # Apply activation if specified (note: activation is handled differently in create_model)
        if activation is not None:
            import warnings

            warnings.warn(
                "The 'activation' parameter is deprecated when using smp.create_model(). "
                "Apply activation manually after model creation if needed.",
                DeprecationWarning,
                stacklevel=2,
            )

        return model

    except Exception as e:
        # Provide helpful error message
        available_archs = []
        try:
            # Try to get available architectures from smp
            if hasattr(smp, "get_available_models"):
                available_archs = smp.get_available_models()
            else:
                available_archs = [
                    "unet",
                    "unetplusplus",
                    "manet",
                    "linknet",
                    "fpn",
                    "pspnet",
                    "deeplabv3",
                    "deeplabv3plus",
                    "pan",
                    "upernet",
                ]
        except:
            available_archs = [
                "unet",
                "fpn",
                "deeplabv3plus",
                "pspnet",
                "linknet",
                "manet",
            ]

        raise ValueError(
            f"Failed to create model with architecture '{architecture}' and encoder '{encoder_name}'. "
            f"Error: {str(e)}. "
            f"Available architectures include: {', '.join(available_archs)}. "
            f"Please check the segmentation-models-pytorch documentation for supported combinations."
        )


def dice_coefficient(pred, target, smooth=1e-6, num_classes=None):
    """
    Calculate Dice coefficient for segmentation (binary or multi-class).

    Args:
        pred (torch.Tensor): Predicted mask (probabilities or logits) with shape [C, H, W] or [H, W].
        target (torch.Tensor): Ground truth mask with shape [H, W].
        smooth (float): Smoothing factor to avoid division by zero.
        num_classes (int, optional): Number of classes. If None, auto-detected.

    Returns:
        float: Mean Dice coefficient across all classes.
    """
    # Convert predictions to class predictions
    if pred.dim() == 3:  # [C, H, W] format
        pred = torch.softmax(pred, dim=0)
        pred_classes = torch.argmax(pred, dim=0)
    elif pred.dim() == 2:  # [H, W] format
        pred_classes = pred
    else:
        raise ValueError(f"Unexpected prediction dimensions: {pred.shape}")

    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = max(pred_classes.max().item(), target.max().item()) + 1

    # Calculate Dice for each class and average
    dice_scores = []
    for class_id in range(num_classes):
        pred_class = (pred_classes == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union > 0:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def iou_coefficient(pred, target, smooth=1e-6, num_classes=None):
    """
    Calculate IoU coefficient for segmentation (binary or multi-class).

    Args:
        pred (torch.Tensor): Predicted mask (probabilities or logits) with shape [C, H, W] or [H, W].
        target (torch.Tensor): Ground truth mask with shape [H, W].
        smooth (float): Smoothing factor to avoid division by zero.
        num_classes (int, optional): Number of classes. If None, auto-detected.

    Returns:
        float: Mean IoU coefficient across all classes.
    """
    # Convert predictions to class predictions
    if pred.dim() == 3:  # [C, H, W] format
        pred = torch.softmax(pred, dim=0)
        pred_classes = torch.argmax(pred, dim=0)
    elif pred.dim() == 2:  # [H, W] format
        pred_classes = pred
    else:
        raise ValueError(f"Unexpected prediction dimensions: {pred.shape}")

    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = max(pred_classes.max().item(), target.max().item()) + 1

    # Calculate IoU for each class and average
    iou_scores = []
    for class_id in range(num_classes):
        pred_class = (pred_classes == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection

        if union > 0:
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())

    return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0


def train_semantic_one_epoch(
    model, optimizer, data_loader, device, epoch, criterion, print_freq=10, verbose=True
):
    """
    Train the semantic segmentation model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        criterion: Loss function.
        print_freq (int): How often to print progress.
        verbose (bool): Whether to print detailed progress.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        # Move images and targets to device
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Print progress
        if i % print_freq == 0:
            elapsed_time = time.time() - start_time
            if verbose:
                print(
                    f"Epoch: {epoch}, Batch: {i}/{num_batches}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s"
                )
            start_time = time.time()

    # Calculate average loss
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_semantic(model, data_loader, device, criterion, num_classes=2):
    """
    Evaluate the semantic segmentation model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to evaluate on.
        criterion: Loss function.
        num_classes (int): Number of classes for evaluation metrics.

    Returns:
        dict: Evaluation metrics including loss, IoU, and Dice.
    """
    model.eval()

    total_loss = 0
    dice_scores = []
    iou_scores = []
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in data_loader:
            # Move to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate metrics for each sample in the batch
            for pred, target in zip(outputs, targets):
                dice = dice_coefficient(pred, target, num_classes=num_classes)
                iou = iou_coefficient(pred, target, num_classes=num_classes)
                dice_scores.append(dice)
                iou_scores.append(iou)

    # Calculate metrics
    avg_loss = total_loss / num_batches
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    return {"loss": avg_loss, "Dice": avg_dice, "IoU": avg_iou}


def train_segmentation_model(
    images_dir,
    labels_dir,
    output_dir,
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=8,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-4,
    seed=42,
    val_split=0.2,
    print_freq=10,
    verbose=True,
    save_best_only=True,
    plot_curves=False,
    **kwargs,
):
    """
    Train a semantic segmentation model for object detection using segmentation-models-pytorch.

    This function trains a semantic segmentation model for object detection (e.g., building detection)
    using models from the segmentation-models-pytorch library. Unlike instance segmentation (Mask R-CNN),
    this approach treats the task as pixel-level binary classification.

    Args:
        images_dir (str): Directory containing image GeoTIFF files.
        labels_dir (str): Directory containing label GeoTIFF files.
        output_dir (str): Directory to save model checkpoints and results.
        architecture (str): Model architecture ('unet', 'deeplabv3', 'deeplabv3plus', 'fpn',
            'pspnet', 'linknet', 'manet'). Defaults to 'unet'.
        encoder_name (str): Encoder backbone name (e.g., 'resnet34', 'resnet50', 'efficientnet-b0').
            Defaults to 'resnet34'.
        encoder_weights (str): Encoder pretrained weights ('imagenet' or None). Defaults to 'imagenet'.
        num_channels (int): Number of input channels. Defaults to 3.
        num_classes (int): Number of output classes (typically 2 for binary segmentation). Defaults to 2.
        batch_size (int): Batch size for training. Defaults to 8.
        num_epochs (int): Number of training epochs. Defaults to 50.
        learning_rate (float): Initial learning rate. Defaults to 0.001.
        weight_decay (float): Weight decay for optimizer. Defaults to 1e-4.
        seed (int): Random seed for reproducibility. Defaults to 42.
        val_split (float): Fraction of data to use for validation (0-1). Defaults to 0.2.
        print_freq (int): Frequency of printing training progress. Defaults to 10.
        verbose (bool): If True, prints detailed training progress. Defaults to True.
        save_best_only (bool): If True, only saves the best model. Otherwise saves all checkpoints.
            Defaults to True.
        plot_curves (bool): If True, plots training curves. Defaults to False.
        **kwargs: Additional arguments passed to smp.create_model().
    Returns:
        None: Model weights are saved to output_dir.

    Raises:
        ImportError: If segmentation-models-pytorch is not installed.
        FileNotFoundError: If input directories don't exist or contain no matching files.
    """
    import datetime

    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation-models-pytorch is not installed. "
            "Please install it with: pip install segmentation-models-pytorch"
        )

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Get all image and label files
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".tif")
        ]
    )
    label_files = sorted(
        [
            os.path.join(labels_dir, f)
            for f in os.listdir(labels_dir)
            if f.endswith(".tif")
        ]
    )

    print(f"Found {len(image_files)} image files and {len(label_files)} label files")

    # Ensure matching files
    if len(image_files) != len(label_files):
        print("Warning: Number of image files and label files don't match!")
        # Find matching files by basename
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

    if len(image_files) == 0:
        raise FileNotFoundError("No matching image and label files found")

    # Split data into train and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=val_split, random_state=seed
    )

    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)} images")

    # Create datasets
    train_dataset = SemanticSegmentationDataset(
        train_imgs,
        train_labels,
        transforms=get_semantic_transform(train=True),
        num_channels=num_channels,
    )
    val_dataset = SemanticSegmentationDataset(
        val_imgs,
        val_labels,
        transforms=get_semantic_transform(train=False),
        num_channels=num_channels,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = get_smp_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=num_channels,
        classes=num_classes,
        activation=None,  # We'll apply softmax later
        **kwargs,
    )
    model.to(device)

    # Set up loss function (CrossEntropyLoss for multi-class, can also use DiceLoss)
    criterion = torch.nn.CrossEntropyLoss()

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Set up learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize tracking variables
    best_iou = 0
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []

    print(f"Starting training with {architecture} + {encoder_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train_semantic_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            criterion,
            print_freq,
            verbose,
        )
        train_losses.append(train_loss)

        # Evaluate on validation set
        eval_metrics = evaluate_semantic(
            model, val_loader, device, criterion, num_classes=num_classes
        )
        val_losses.append(eval_metrics["loss"])
        val_ious.append(eval_metrics["IoU"])
        val_dices.append(eval_metrics["Dice"])

        # Update learning rate
        lr_scheduler.step(eval_metrics["loss"])

        # Print metrics
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {eval_metrics['loss']:.4f}, "
            f"Val IoU: {eval_metrics['IoU']:.4f}, "
            f"Val Dice: {eval_metrics['Dice']:.4f}"
        )

        # Save best model
        if eval_metrics["IoU"] > best_iou:
            best_iou = eval_metrics["IoU"]
            print(f"Saving best model with IoU: {best_iou:.4f}")
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # Save checkpoint every 10 epochs (if not save_best_only)
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
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ious": val_ious,
        "val_dices": val_dices,
    }
    torch.save(history, os.path.join(output_dir, "training_history.pth"))

    # Save training summary
    with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
        f.write(
            f"Training completed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Architecture: {architecture}\n")
        f.write(f"Encoder: {encoder_name}\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Best validation IoU: {best_iou:.4f}\n")
        f.write(f"Final validation IoU: {val_ious[-1]:.4f}\n")
        f.write(f"Final validation Dice: {val_dices[-1]:.4f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")

    print(f"Training complete! Best IoU: {best_iou:.4f}")
    print(f"Models saved to {output_dir}")

    # Plot training curves
    if plot_curves:
        try:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(val_ious, label="Val IoU")
            plt.title("IoU Score")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(val_dices, label="Val Dice")
            plt.title("Dice Score")
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "training_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
            print(
                f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}"
            )
            plt.close()
        except Exception as e:
            print(f"Could not save training curves: {e}")


def semantic_inference_on_geotiff(
    model,
    geotiff_path,
    output_path,
    window_size=512,
    overlap=256,
    batch_size=4,
    num_channels=3,
    num_classes=2,
    device=None,
    **kwargs,
):
    """
    Perform semantic segmentation inference on a large GeoTIFF using a sliding window approach.

    Args:
        model (torch.nn.Module): Trained semantic segmentation model.
        geotiff_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save output mask GeoTIFF.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        batch_size (int): Batch size for inference.
        num_channels (int): Number of channels to use from the input image.
        num_classes (int): Number of classes in the model output.
        device (torch.device, optional): Device to run inference on.
        **kwargs: Additional arguments.

    Returns:
        tuple: Tuple containing output path and inference time in seconds.
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # Put model in evaluation mode
    model.to(device)
    model.eval()

    # Open the GeoTIFF
    with rasterio.open(geotiff_path) as src:
        # Read metadata
        meta = src.meta
        height = src.height
        width = src.width

        # Update metadata for output raster
        out_meta = meta.copy()
        out_meta.update({"count": 1, "dtype": "uint8"})

        # Initialize accumulator arrays for multi-class probability blending
        # We'll accumulate probabilities for each class and then take argmax
        prob_accumulator = np.zeros((num_classes, height, width), dtype=np.float32)
        count_accumulator = np.zeros((height, width), dtype=np.float32)

        # Calculate steps
        steps_y = math.ceil((height - overlap) / (window_size - overlap))
        steps_x = math.ceil((width - overlap) / (window_size - overlap))
        last_y = height - window_size
        last_x = width - window_size

        total_windows = steps_y * steps_x
        print(f"Processing {total_windows} windows...")

        pbar = tqdm(total=total_windows)
        batch_inputs = []
        batch_positions = []
        batch_count = 0

        start_time = time.time()

        for i in range(steps_y + 1):
            y = min(i * (window_size - overlap), last_y)
            y = max(0, y)

            if y > last_y and i > 0:
                continue

            for j in range(steps_x + 1):
                x = min(j * (window_size - overlap), last_x)
                x = max(0, x)

                if x > last_x and j > 0:
                    continue

                # Read window
                window = src.read(window=Window(x, y, window_size, window_size))

                if window.shape[1] == 0 or window.shape[2] == 0:
                    continue

                current_height = window.shape[1]
                current_width = window.shape[2]

                # Normalize and prepare input
                image = window.astype(np.float32) / 255.0

                # Handle different number of bands
                if image.shape[0] > num_channels:
                    image = image[:num_channels]
                elif image.shape[0] < num_channels:
                    padded = np.zeros(
                        (num_channels, current_height, current_width), dtype=np.float32
                    )
                    padded[: image.shape[0]] = image
                    image = padded

                # Convert to tensor
                image_tensor = torch.tensor(image, device=device)

                # Add to batch
                batch_inputs.append(image_tensor)
                batch_positions.append((y, x, current_height, current_width))
                batch_count += 1

                # Process batch
                if batch_count == batch_size or (i == steps_y and j == steps_x):
                    with torch.no_grad():
                        batch_tensor = torch.stack(batch_inputs)
                        outputs = model(batch_tensor)

                        # Apply softmax to get class probabilities
                        probs = torch.softmax(outputs, dim=1)

                    # Process each output in the batch
                    for idx, prob in enumerate(probs):
                        y_pos, x_pos, h, w = batch_positions[idx]

                        # Create weight matrix for blending
                        y_grid, x_grid = np.mgrid[0:h, 0:w]
                        dist_from_left = x_grid
                        dist_from_right = w - x_grid - 1
                        dist_from_top = y_grid
                        dist_from_bottom = h - y_grid - 1

                        edge_distance = np.minimum.reduce(
                            [
                                dist_from_left,
                                dist_from_right,
                                dist_from_top,
                                dist_from_bottom,
                            ]
                        )
                        edge_distance = np.minimum(edge_distance, overlap / 2)

                        # Avoid zero weights - use minimum weight of 0.1
                        weight = np.maximum(edge_distance / (overlap / 2), 0.1)

                        # For non-overlapping windows, use uniform weight
                        if overlap == 0:
                            weight = np.ones_like(weight)

                        # Convert probabilities to numpy [C, H, W]
                        prob_np = prob.cpu().numpy()

                        # Accumulate weighted probabilities for each class
                        y_slice = slice(y_pos, y_pos + h)
                        x_slice = slice(x_pos, x_pos + w)

                        # Add weighted probabilities for each class
                        for class_idx in range(num_classes):
                            prob_accumulator[class_idx, y_slice, x_slice] += (
                                prob_np[class_idx] * weight
                            )

                        # Update weight accumulator
                        count_accumulator[y_slice, x_slice] += weight

                    # Reset batch
                    batch_inputs = []
                    batch_positions = []
                    batch_count = 0
                    pbar.update(len(probs))

        pbar.close()

        # Calculate final mask by taking argmax of accumulated probabilities
        mask = np.zeros((height, width), dtype=np.uint8)
        valid_pixels = count_accumulator > 0

        if np.any(valid_pixels):
            # Normalize accumulated probabilities by weights
            normalized_probs = np.zeros_like(prob_accumulator)
            for class_idx in range(num_classes):
                normalized_probs[class_idx, valid_pixels] = (
                    prob_accumulator[class_idx, valid_pixels]
                    / count_accumulator[valid_pixels]
                )

            # Take argmax to get final class predictions
            mask[valid_pixels] = np.argmax(
                normalized_probs[:, valid_pixels], axis=0
            ).astype(np.uint8)

            # Check class distribution in predictions (summary only)
            unique_classes, class_counts = np.unique(
                mask[valid_pixels], return_counts=True
            )
            bg_ratio = np.sum(mask == 0) / mask.size
            print(
                f"Predicted classes: {len(unique_classes)} classes, Background: {bg_ratio:.1%}"
            )

        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")

        # Save output
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mask, 1)

        print(f"Saved prediction to {output_path}")

        return output_path, inference_time


def semantic_segmentation(
    input_path,
    output_path,
    model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
    device=None,
    **kwargs,
):
    """
    Perform semantic segmentation on a GeoTIFF using a trained model.

    Args:
        input_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save output mask GeoTIFF.
        model_path (str): Path to trained model weights.
        architecture (str): Model architecture used for training.
        encoder_name (str): Encoder backbone name used for training.
        num_channels (int): Number of channels in the input image and model.
        num_classes (int): Number of classes in the model.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to run inference on.
        **kwargs: Additional arguments.

    Returns:
        None: Output mask is saved to output_path.
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # Load model
    model = get_smp_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,  # We're loading trained weights
        in_channels=num_channels,
        classes=num_classes,
        activation=None,
    )

    if not os.path.exists(model_path):
        try:
            model_path = download_model_from_hf(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    semantic_inference_on_geotiff(
        model=model,
        geotiff_path=input_path,
        output_path=output_path,
        window_size=window_size,
        overlap=overlap,
        batch_size=batch_size,
        num_channels=num_channels,
        num_classes=num_classes,
        device=device,
        **kwargs,
    )
