"""Image classification operations.

Wraps geoai classification functions: TIMM-based training,
prediction, and evaluation for CLI consumption.
"""

import os
from typing import Any, Dict, List, Optional


def train_classifier(
    train_dir: str,
    val_dir: Optional[str] = None,
    output_dir: str = "./classification_output",
    model_name: str = "resnet50",
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    in_channels: int = 3,
) -> Dict[str, Any]:
    """Train an image classification model.

    Args:
        train_dir: Training data directory (subdirectories are class names).
        val_dir: Optional validation data directory.
        output_dir: Output directory for model and metrics.
        model_name: TIMM model architecture name.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        image_size: Input image size (square).
        in_channels: Number of input channels.

    Returns:
        Result dict with model path and training metrics.

    Raises:
        FileNotFoundError: If training directory does not exist.
    """
    train_dir = os.path.abspath(train_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if val_dir:
        val_dir = os.path.abspath(val_dir)
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    os.makedirs(output_dir, exist_ok=True)

    from geoai import train_image_classifier

    kwargs = {
        "train_dir": train_dir,
        "output_dir": output_dir,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "in_channels": in_channels,
    }
    if val_dir:
        kwargs["val_dir"] = val_dir

    result = train_image_classifier(**kwargs)

    output = {
        "train_dir": train_dir,
        "output_dir": output_dir,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    if isinstance(result, dict):
        output.update(result)
    elif result is not None:
        output["result"] = str(result)

    return output


def predict_classification(
    image_path: str,
    model_path: str,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    image_size: int = 224,
    in_channels: int = 3,
) -> Dict[str, Any]:
    """Classify an image with a trained model.

    Args:
        image_path: Path to the input image.
        model_path: Path to the trained model checkpoint.
        num_classes: Number of classes.
        class_names: Optional list of class names.
        image_size: Expected input image size.
        in_channels: Number of input channels.

    Returns:
        Result dict with predicted class and probabilities.

    Raises:
        FileNotFoundError: If input files do not exist.
    """
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from geoai.recognize import predict_with_timm

    kwargs = {
        "image_size": image_size,
        "in_channels": in_channels,
    }
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if class_names is not None:
        kwargs["class_names"] = class_names

    result = predict_with_timm(
        image_path=image_path,
        model_path=model_path,
        **kwargs,
    )

    output = {
        "image": image_path,
        "model": model_path,
    }

    if isinstance(result, dict):
        output.update(result)
    elif isinstance(result, (list, tuple)):
        output["predictions"] = list(result)
    else:
        output["result"] = str(result)

    return output
