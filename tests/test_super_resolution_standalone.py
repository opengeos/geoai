#!/usr/bin/env python3
"""Standalone test for super-resolution module to verify implementation."""

import sys
import os
import tempfile
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_super_resolution_import():
    """Test importing the super-resolution module."""
    try:
        # Import directly from the module file to avoid __init__.py dependencies
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "super_resolution", "geoai/super_resolution.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        SuperResolutionModel = module.SuperResolutionModel
        create_super_resolution_model = module.create_super_resolution_model

        # Make them available globally for other tests
        globals()["SuperResolutionModel"] = SuperResolutionModel
        globals()["create_super_resolution_model"] = create_super_resolution_model

        print("OK Super-resolution module imported successfully")
        return True
    except Exception as e:
        print(f"FAIL Failed to import super-resolution module: {e}")
        return False


def test_model_initialization():
    """Test model initialization."""
    try:
        # Test ESRGAN model
        model = SuperResolutionModel(model_type="esrgan", upscale_factor=4)
        print("OK ESRGAN model initialized successfully")

        # Test SRCNN model
        model = SuperResolutionModel(model_type="srcnn", upscale_factor=2)
        print("OK SRCNN model initialized successfully")

        return True
    except Exception as e:
        print(f"FAIL Model initialization failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through the model."""
    try:
        import torch

        model = SuperResolutionModel(model_type="srcnn", upscale_factor=2)

        # Create dummy input
        batch_size, channels, height, width = 1, 3, 64, 64
        dummy_input = torch.randn(batch_size, channels, height, width)

        # Move input to same device as model
        dummy_input = dummy_input.to(model.device)

        # Forward pass
        with torch.no_grad():
            output = model.model(dummy_input)

        expected_height, expected_width = height * 2, width * 2
        assert output.shape == (
            batch_size,
            channels,
            expected_height,
            expected_width,
        ), f"Expected shape {(batch_size, channels, expected_height, expected_width)}, got {output.shape}"

        print("OK Forward pass successful")
        return True
    except Exception as e:
        print(f"FAIL Forward pass failed: {e}")
        return False


def test_create_function():
    """Test the convenience creation function."""
    try:
        model = create_super_resolution_model(model_type="srcnn", upscale_factor=2)
        print("OK Convenience function works")
        return True
    except Exception as e:
        print(f"FAIL Convenience function failed: {e}")
        return False


def test_save_load_model():
    """Test saving and loading model."""
    try:
        model = SuperResolutionModel(model_type="srcnn", upscale_factor=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")

            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path), "Model file was not created"

            # Load model
            new_model = SuperResolutionModel(model_type="srcnn", upscale_factor=2)
            new_model.load_model(model_path)

            print("OK Save/load functionality works")
            return True
    except Exception as e:
        print(f"FAIL Save/load failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Super-Resolution Implementation")
    print("=" * 40)

    tests = [
        test_super_resolution_import,
        test_model_initialization,
        test_forward_pass,
        test_create_function,
        test_save_load_model,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Super-resolution implementation is working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
