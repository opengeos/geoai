"""Model inspection and loading utilities."""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

__all__ = ["inspect_pth_file", "try_common_architectures"]


def inspect_pth_file(pth_path: str) -> Dict[str, Any]:
    """
    Inspect a PyTorch .pth model file to determine its architecture.

    Args:
        pth_path: Path to the .pth file to inspect

    Returns:
        Information about the model architecture
    """
    # Check if file exists
    if not os.path.exists(pth_path):
        print(f"Error: File {pth_path} not found")
        return

    # Load the checkpoint
    try:
        checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))
        print(f"\n{'='*50}")
        print(f"Inspecting model file: {pth_path}")
        print(f"{'='*50}\n")

        # Check if it's a state_dict or a complete model
        if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                print("Found 'state_dict' key in the checkpoint.")
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                print("Found 'model_state_dict' key in the checkpoint.")
                state_dict = checkpoint["model_state_dict"]
            else:
                print("Assuming file contains a direct state_dict.")
                state_dict = checkpoint

            # Print the keys in the checkpoint
            print("\nCheckpoint contains the following keys:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"- {key} (dictionary with {len(checkpoint[key])} items)")
                elif isinstance(checkpoint[key], (torch.Tensor, list, tuple)):
                    print(
                        f"- {key} (shape/size: {len(checkpoint[key]) if isinstance(checkpoint[key], (list, tuple)) else checkpoint[key].shape})"
                    )
                else:
                    print(f"- {key} ({type(checkpoint[key]).__name__})")

            # Try to infer the model architecture from the state_dict keys
            print("\nAnalyzing model architecture from state_dict...")

            # Extract layer keys for analysis
            layer_keys = list(state_dict.keys())

            # Print the first few layer keys to understand naming pattern
            print("\nFirst 10 layer names in state_dict:")
            for i, key in enumerate(layer_keys[:10]):
                shape = state_dict[key].shape
                print(f"- {key} (shape: {shape})")

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

            print("\nArchitecture indicators found in layer names:")
            for indicator, count in architecture_indicators.items():
                if count > 0:
                    print(f"- '{indicator}' appears {count} times")

            # Count total parameters
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"\nTotal parameters: {total_params:,}")

            # Try to load the model with different architectures
            print("\nAttempting to match with common architectures...")

            # Try to identify if it's a segmentation model
            if any("out" in k or "classifier" in k for k in layer_keys):
                print("Model appears to be a segmentation model.")

                # Check if it might be a UNet
                if (
                    architecture_indicators["encoder"] > 0
                    and architecture_indicators["decoder"] > 0
                ):
                    print(
                        "Architecture seems to be a UNet-based model with encoder-decoder structure."
                    )
                # Check for FCN or DeepLab indicators
                elif architecture_indicators["fcn"] > 0:
                    print(
                        "Architecture seems to be FCN-based (Fully Convolutional Network)."
                    )
                elif architecture_indicators["deeplab"] > 0:
                    print("Architecture seems to be DeepLab-based.")
                elif architecture_indicators["backbone"] > 0:
                    print(
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
                    print(f"\nModel likely has {num_classes} output classes.")

            print("\nSUMMARY:")
            print("The model appears to be", end=" ")
            if architecture_indicators["unet"] > 0:
                print("a UNet architecture.", end=" ")
            elif architecture_indicators["fcn"] > 0:
                print("an FCN architecture.", end=" ")
            elif architecture_indicators["deeplab"] > 0:
                print("a DeepLab architecture.", end=" ")
            elif architecture_indicators["resnet"] > 0:
                print("ResNet-based.", end=" ")
            else:
                print("a custom architecture.", end=" ")

            # Try to load with common models
            try_common_architectures(state_dict)

        else:
            print(
                "The file contains an entire model object rather than just a state dictionary."
            )
            # If it's a complete model, we can directly examine its architecture
            print(checkpoint)

    except Exception as e:
        print(f"Error loading the model file: {str(e)}")


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

    print("\nTrying to load state_dict into common architectures:")

    for name, model_fn in models_to_try.items():
        try:
            model = model_fn()
            # Sometimes state_dict keys have 'model.' prefix
            if all(k.startswith("model.") for k in state_dict.keys()):
                cleaned_state_dict = {k[6:]: v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            print(
                f"- {name}: Successfully loaded (may have missing or unexpected keys)"
            )

            # Generate model summary
            print(f"\nSummary of {name} architecture:")
            summary = torchinfo.summary(model, input_size=(1, 3, 224, 224), verbose=0)
            print(summary)

        except Exception as e:
            print(f"- {name}: Failed to load - {str(e)}")
