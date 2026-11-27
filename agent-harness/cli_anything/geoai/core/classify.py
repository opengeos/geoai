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
        val_dir: Optional validation data directory (validated if given;
            geoai.recognize.train_image_classifier splits data_dir internally).
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
        raise ValueError(
            "Separate val_dir is not supported by geoai.recognize.train_image_classifier; "
            "place all class subdirectories under train_dir instead."
        )

    os.makedirs(output_dir, exist_ok=True)

    from geoai.recognize import train_image_classifier

    result = train_image_classifier(
        data_dir=train_dir,
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        in_channels=in_channels,
    )

    output: Dict[str, Any] = {
        "train_dir": train_dir,
        "output_dir": output_dir,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    if isinstance(result, dict):
        if "checkpoint_path" in result:
            output["checkpoint_path"] = result["checkpoint_path"]
        if "class_names" in result:
            output["class_names"] = result["class_names"]
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

    from geoai.timm_train import TimmClassifier, predict_with_timm

    load_kwargs: Dict[str, Any] = {"in_channels": in_channels}
    if num_classes is not None:
        load_kwargs["num_classes"] = num_classes

    model = TimmClassifier.load_from_checkpoint(model_path, **load_kwargs)
    result = predict_with_timm(
        model=model,
        image_paths=[image_path],
        return_probabilities=True,
    )

    output: Dict[str, Any] = {
        "image": image_path,
        "model": model_path,
        "image_size": image_size,
    }

    if isinstance(result, tuple) and len(result) == 2:
        preds, probs = result
        pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        prob_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        output["predictions"] = pred_list
        output["probabilities"] = prob_list
        if class_names and pred_list:
            idx = int(pred_list[0])
            if 0 <= idx < len(class_names):
                output["predicted_class"] = class_names[idx]
    elif hasattr(result, "tolist"):
        output["predictions"] = result.tolist()
    else:
        output["result"] = str(result)

    return output
