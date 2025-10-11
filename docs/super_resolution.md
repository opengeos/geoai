# Super-Resolution Module

The super-resolution module provides deep learning-based image enhancement for geospatial imagery, enabling the upscaling of low-resolution satellite and aerial images while preserving geospatial metadata.

## Overview

This module implements ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) inspired architectures for super-resolution tasks specifically designed for geospatial data. It supports multiple upscaling factors, handles large image tiles efficiently, and preserves coordinate reference systems and geotransforms.

## Features

- **Multiple Architectures**: ESRGAN-inspired generator with residual blocks, SRCNN for lightweight applications
- **Configurable Upscaling**: Support for 2x, 4x, and 8x upscaling factors
- **Geospatial Metadata Preservation**: Maintains CRS, geotransform, and spatial reference information
- **Large Image Support**: Tiled processing for handling images larger than GPU memory
- **GPU Acceleration**: Automatic device detection and CUDA support
- **Training and Inference**: Complete pipeline for model training and inference
- **Evaluation Metrics**: PSNR and SSIM evaluation for model performance assessment

## Installation

The super-resolution module requires additional dependencies:

```bash
pip install torch torchvision scikit-image tqdm
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from geoai import SuperResolutionModel

# Initialize model
sr = SuperResolutionModel(model_type="esrgan", upscale_factor=4)

# Enhance image resolution
high_res_path = sr.enhance_image("low_res_satellite.tif", "high_res_output.tif")
```

### Using Pre-trained Models

```python
# Load pre-trained model
sr = SuperResolutionModel(upscale_factor=4)
sr.load_model("path/to/pretrained/model.pth")

# Process image
enhanced = sr.enhance_image("input.tif", "output.tif")
```

### Custom Model Creation

```python
from geoai import create_super_resolution_model

# Create model with specific parameters
sr = create_super_resolution_model(
    model_type="srcnn",
    upscale_factor=2,
    device="cuda"  # Use GPU
)
```

## API Reference

### SuperResolutionModel

#### Initialization

```python
SuperResolutionModel(model_type="esrgan",
                    upscale_factor=4,
                    device=None,
                    num_channels=3)
```

**Parameters:**
- `model_type` (str): Architecture type ("esrgan", "srcnn")
- `upscale_factor` (int): Upscaling factor (2, 4, 8)
- `device` (str, optional): Computing device ("cuda", "cpu", "mps")
- `num_channels` (int): Number of input channels (default: 3 for RGB)

#### Methods

##### load_model(model_path)
Load pre-trained model weights.

```python
sr.load_model("model_weights.pth")
```

##### save_model(model_path)
Save model weights to file.

```python
sr.save_model("trained_model.pth")
```

##### enhance_image(input_path, output_path=None, tile_size=512, overlap=32)
Enhance image resolution.

```python
# Save to file
sr.enhance_image("input.tif", "output.tif")

# Return as array
enhanced_array = sr.enhance_image("input.tif")
```

**Parameters:**
- `input_path` (str): Path to input image
- `output_path` (str, optional): Path to save enhanced image
- `tile_size` (int): Size of tiles for processing large images
- `overlap` (int): Overlap between tiles

**Returns:** Path to output file or numpy array

##### train(train_dir, val_dir=None, epochs=100, batch_size=16, learning_rate=1e-4, save_path=None)
Train the super-resolution model.

```python
history = sr.train(
    train_dir="path/to/training/data",
    val_dir="path/to/validation/data",
    epochs=50,
    batch_size=8,
    save_path="model_checkpoints"
)
```

**Parameters:**
- `train_dir` (str): Directory containing training images
- `val_dir` (str, optional): Directory containing validation images
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `learning_rate` (float): Learning rate
- `save_path` (str, optional): Path to save model checkpoints

**Returns:** Dictionary with training history

##### evaluate(test_dir, metrics=['psnr', 'ssim'])
Evaluate model performance.

```python
results = sr.evaluate("test_images/", metrics=['psnr', 'ssim'])
print(f"PSNR: {results['psnr']:.2f}, SSIM: {results['ssim']:.2f}")
```

## Training Custom Models

### Data Preparation

Training requires paired low-resolution and high-resolution images. The module expects GeoTIFF files in the training directory.

```python
# Directory structure
training_data/
├── image1.tif
├── image2.tif
└── ...
```

### Training Example

```python
from geoai import SuperResolutionModel

# Initialize model
sr = SuperResolutionModel(upscale_factor=4, model_type="esrgan")

# Train model
history = sr.train(
    train_dir="path/to/training/data",
    val_dir="path/to/validation/data",
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    save_path="checkpoints/esrgan_4x"
)

# Save final model
sr.save_model("esrgan_4x_final.pth")
```

### Training Script

For more advanced training scenarios, you can use the provided training utilities:

```python
import torch
from torch.utils.data import DataLoader
from geoai.super_resolution import GeospatialSRDataset

# Create datasets
train_dataset = GeospatialSRDataset("training_data/", upscale_factor=4)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for lr_images, hr_images in train_loader:
        # Training logic here
        pass
```

## Advanced Usage

### Handling Large Images

For very large images that don't fit in GPU memory:

```python
# Process in smaller tiles
sr.enhance_image(
    "large_satellite_image.tif",
    "enhanced_output.tif",
    tile_size=256,  # Smaller tiles
    overlap=64      # More overlap for seamless results
)
```

### Batch Processing

```python
import glob
from geoai import SuperResolutionModel

sr = SuperResolutionModel(upscale_factor=2)

# Process multiple images
input_files = glob.glob("satellite_images/*.tif")
for input_file in input_files:
    output_file = input_file.replace(".tif", "_sr.tif")
    sr.enhance_image(input_file, output_file)
```

### Model Fine-tuning

```python
# Load pre-trained model
sr = SuperResolutionModel(upscale_factor=4)
sr.load_model("pretrained_model.pth")

# Fine-tune on domain-specific data
history = sr.train(
    train_dir="domain_specific_data/",
    epochs=20,
    learning_rate=1e-5  # Lower learning rate for fine-tuning
)

# Save fine-tuned model
sr.save_model("fine_tuned_model.pth")
```

## Performance Considerations

### GPU Memory Management

- Use smaller batch sizes for limited GPU memory
- Enable tiled processing for large images
- Monitor GPU usage with `nvidia-smi`

### Speed Optimization

- Use CUDA-enabled PyTorch for GPU acceleration
- Process images in tiles to manage memory
- Use appropriate tile sizes based on your GPU memory

## Evaluation Metrics

The module supports standard super-resolution evaluation metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index)**: Measures structural similarity

```python
# Evaluate model
metrics = sr.evaluate("test_dataset/", metrics=['psnr', 'ssim'])
print(f"Average PSNR: {metrics['psnr']:.2f}")
print(f"Average SSIM: {metrics['ssim']:.2f}")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use tiled processing
2. **Invalid geotransform**: Ensure input images have proper geospatial metadata
3. **Low quality results**: Train longer or use larger models

### Memory Optimization

```python
# For memory-constrained environments
torch.cuda.empty_cache()  # Clear GPU cache

# Use smaller models
sr = SuperResolutionModel(model_type="srcnn", upscale_factor=2)
```

## Examples

### Satellite Image Enhancement

```python
from geoai import SuperResolutionModel

# Enhance Landsat imagery from 30m to 7.5m resolution
sr = SuperResolutionModel(upscale_factor=4)  # 4x upscaling
sr.load_model("landsat_sr_model.pth")

sr.enhance_image("landsat_low_res.tif", "landsat_high_res.tif")
```

### Aerial Photography Upscaling

```python
# Enhance aerial photos for better feature detection
sr = SuperResolutionModel(upscale_factor=2, model_type="esrgan")

# Process directory of images
import os
for filename in os.listdir("aerial_photos/"):
    if filename.endswith(".tif"):
        input_path = os.path.join("aerial_photos", filename)
        output_path = os.path.join("enhanced_aerial", filename)
        sr.enhance_image(input_path, output_path)
```

## References

- Wang, X., et al. "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks." ECCV 2018.
- Dong, C., et al. "Image Super-Resolution Using Deep Convolutional Networks." IEEE TNNLS 2016.

## Contributing

Contributions to the super-resolution module are welcome. Please ensure that:

- New features include comprehensive tests
- Documentation is updated for API changes
- Performance benchmarks are maintained
- Code follows PEP 8 standards