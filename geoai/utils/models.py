"""Model inspection and loading utilities."""

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

logger = logging.getLogger(__name__)

__all__ = ["inspect_pth_file", "try_common_architectures"]


def inspect_pth_file(pth_path: str) -> None:
    """
    Inspect a PyTorch .pth model file to determine its architecture.

    Logs detailed information about the model's layer structure,
    architecture indicators, and output classes.

    Args:
        pth_path: Path to the .pth file to inspect

    Raises:
        FileNotFoundError: If *pth_path* does not exist.
    """
    # Check if file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")

    # Load the checkpoint
    try:
        checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))
        logger.info("=" * 50)
        logger.info("Inspecting model file: %s", pth_path)
        logger.info("=" * 50)

        # Check if it's a state_dict or a complete model
        if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                logger.info("Found 'state_dict' key in the checkpoint.")
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                logger.info("Found 'model_state_dict' key in the checkpoint.")
                state_dict = checkpoint["model_state_dict"]
            else:
                logger.info("Assuming file contains a direct state_dict.")
                state_dict = checkpoint

            # Print the keys in the checkpoint
            logger.info("Checkpoint contains the following keys:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    logger.info(
                        "- %s (dictionary with %d items)", key, len(checkpoint[key])
                    )
                elif isinstance(checkpoint[key], (torch.Tensor, list, tuple)):
                    logger.info(
                        "- %s (shape/size: %s)",
                        key,
                        (
                            len(checkpoint[key])
                            if isinstance(checkpoint[key], (list, tuple))
                            else checkpoint[key].shape
                        ),
                    )
                else:
                    logger.info("- %s (%s)", key, type(checkpoint[key]).__name__)

            # Try to infer the model architecture from the state_dict keys
            logger.info("Analyzing model architecture from state_dict...")

            # Extract layer keys for analysis
            layer_keys = list(state_dict.keys())

            # Print the first few layer keys to understand naming pattern
            logger.info("First 10 layer names in state_dict:")
            for i, key in enumerate(layer_keys[:10]):
                shape = state_dict[key].shape
                logger.info("- %s (shape: %s)", key, shape)

            # Look for architecture indicators in the keys
            architecture_indicators = {
                "conv": 0,
                "bn": 0,
                "layer": 0,
                "fc": 0,
                "backbone": 0,
                "encoder": 0,
                "decoder": 0,
                "unet": 0,
                "resnet": 0,
                "classifier": 0,
                "deeplab": 0,
                "fcn": 0,
            }

            for key in layer_keys:
                for indicator in architecture_indicators:
                    if indicator in key.lower():
                        architecture_indicators[indicator] += 1

            logger.info("Architecture indicators found in layer names:")
            for indicator, count in architecture_indicators.items():
                if count > 0:
                    logger.info("- '%s' appears %d times", indicator, count)

            # Count total parameters
            total_params = sum(p.numel() for p in state_dict.values())
            logger.info("Total parameters: %s", f"{total_params:,}")

            # Try to load the model with different architectures
            logger.info("Attempting to match with common architectures...")

            # Try to identify if it's a segmentation model
            if any("out" in k or "classifier" in k for k in layer_keys):
                logger.info("Model appears to be a segmentation model.")

                # Check if it might be a UNet
                if (
                    architecture_indicators["encoder"] > 0
                    and architecture_indicators["decoder"] > 0
                ):
                    logger.info(
                        "Architecture seems to be a UNet-based model with encoder-decoder structure."
                    )
                # Check for FCN or DeepLab indicators
                elif architecture_indicators["fcn"] > 0:
                    logger.info(
                        "Architecture seems to be FCN-based (Fully Convolutional Network)."
                    )
                elif architecture_indicators["deeplab"] > 0:
                    logger.info("Architecture seems to be DeepLab-based.")
                elif architecture_indicators["backbone"] > 0:
                    logger.info(
                        "Model has a backbone architecture, likely a modern segmentation model."
                    )

            # Try to infer output classes from the final layer
            output_layer_keys = [
                k for k in layer_keys if "classifier" in k or k.endswith(".out.weight")
            ]
            if output_layer_keys:
                output_shape = state_dict[output_layer_keys[0]].shape
                if len(output_shape) >= 2:
                    num_classes = output_shape[0]
                    logger.info("Model likely has %d output classes.", num_classes)

            logger.info("SUMMARY:")
            if architecture_indicators["unet"] > 0:
                arch_desc = "a UNet architecture."
            elif architecture_indicators["fcn"] > 0:
                arch_desc = "an FCN architecture."
            elif architecture_indicators["deeplab"] > 0:
                arch_desc = "a DeepLab architecture."
            elif architecture_indicators["resnet"] > 0:
                arch_desc = "ResNet-based."
            else:
                arch_desc = "a custom architecture."
            logger.info("The model appears to be %s", arch_desc)

            # Try to load with common models
            try_common_architectures(state_dict)

        else:
            logger.info(
                "The file contains an entire model object rather than just a state dictionary."
            )
            # If it's a complete model, we can directly examine its architecture
            logger.info("%s", checkpoint)

    except (OSError, RuntimeError) as e:
        logger.error("Error loading the model file: %s", str(e))


def try_common_architectures(state_dict: Dict[str, Any]) -> Optional[str]:
    """
    Try to load the state_dict into common architectures to see which one fits.

    Args:
        state_dict: The model's state dictionary
    """
    import torchinfo

    # Test models and their initializations
    models_to_try = {
        "FCN-ResNet50": lambda: fcn_resnet50(num_classes=9),
        "DeepLabV3-ResNet50": lambda: deeplabv3_resnet50(num_classes=9),
    }

    logger.info("Trying to load state_dict into common architectures:")

    for name, model_fn in models_to_try.items():
        try:
            model = model_fn()
            # Sometimes state_dict keys have 'model.' prefix
            if all(k.startswith("model.") for k in state_dict.keys()):
                cleaned_state_dict = {k[6:]: v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            logger.info(
                "- %s: Successfully loaded (may have missing or unexpected keys)",
                name,
            )

            # Generate model summary
            logger.info("Summary of %s architecture:", name)
            summary = torchinfo.summary(model, input_size=(1, 3, 224, 224), verbose=0)
            logger.info("%s", summary)

        except (RuntimeError, ValueError) as e:
            logger.info("- %s: Failed to load - %s", name, str(e))
