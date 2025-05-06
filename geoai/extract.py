"""This module provides a dataset class for object extraction from raster data"""

# Standard Library
import os
import time

# Third-Party Libraries
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy.ndimage as ndimage
import torch
from huggingface_hub import hf_hub_download
from rasterio.windows import Window
from shapely.geometry import Polygon, box
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn,
)
from tqdm import tqdm

# Local Imports
from .utils import get_raster_stats

try:
    from torchgeo.datasets import NonGeoDataset
except ImportError as e:
    raise ImportError(
        "Your torchgeo version is too old. Please upgrade to the latest version using 'pip install -U torchgeo'."
    )


class CustomDataset(NonGeoDataset):
    """
    A TorchGeo dataset for object extraction with overlapping tiles support.

    This dataset class creates overlapping image tiles for object detection,
    ensuring complete coverage of the input raster including right and bottom edges.
    It inherits from NonGeoDataset to avoid spatial indexing issues.

    Attributes:
        raster_path: Path to the input raster file.
        chip_size: Size of image chips to extract (height, width).
        overlap: Amount of overlap between adjacent tiles (0.0-1.0).
        transforms: Transforms to apply to the image.
        verbose: Whether to print detailed processing information.
        stride_x: Horizontal stride between tiles based on overlap.
        stride_y: Vertical stride between tiles based on overlap.
        row_starts: Starting Y positions for each row of tiles.
        col_starts: Starting X positions for each column of tiles.
        crs: Coordinate reference system of the raster.
        transform: Affine transform of the raster.
        height: Height of the raster in pixels.
        width: Width of the raster in pixels.
        count: Number of bands in the raster.
        bounds: Geographic bounds of the raster (west, south, east, north).
        roi: Shapely box representing the region of interest.
        rows: Number of rows of tiles.
        cols: Number of columns of tiles.
        raster_stats: Statistics of the raster.
    """

    def __init__(
        self,
        raster_path,
        chip_size=(512, 512),
        overlap=0.5,
        transforms=None,
        band_indexes=None,
        verbose=False,
    ):
        """
        Initialize the dataset with overlapping tiles.

        Args:
            raster_path: Path to the input raster file.
            chip_size: Size of image chips to extract (height, width). Default is (512, 512).
            overlap: Amount of overlap between adjacent tiles (0.0-1.0). Default is 0.5 (50%).
            transforms: Transforms to apply to the image. Default is None.
            band_indexes: List of band indexes to use. Default is None (use all bands).
            verbose: Whether to print detailed processing information. Default is False.

        Raises:
            ValueError: If overlap is too high resulting in non-positive stride.
        """
        super().__init__()

        # Initialize parameters
        self.raster_path = raster_path
        self.chip_size = chip_size
        self.overlap = overlap
        self.transforms = transforms
        self.band_indexes = band_indexes
        self.verbose = verbose
        self.warned_about_bands = False

        # Calculate stride based on overlap
        self.stride_x = int(chip_size[1] * (1 - overlap))
        self.stride_y = int(chip_size[0] * (1 - overlap))

        if self.stride_x <= 0 or self.stride_y <= 0:
            raise ValueError(
                f"Overlap {overlap} is too high, resulting in non-positive stride"
            )

        with rasterio.open(self.raster_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.height = src.height
            self.width = src.width
            self.count = src.count

            # Define the bounds of the dataset
            west, south, east, north = src.bounds
            self.bounds = (west, south, east, north)
            self.roi = box(*self.bounds)

            # Calculate starting positions for each tile
            self.row_starts = []
            self.col_starts = []

            # Normal row starts using stride
            for r in range((self.height - 1) // self.stride_y):
                self.row_starts.append(r * self.stride_y)

            # Add a special last row that ensures we reach the bottom edge
            if self.height > self.chip_size[0]:
                self.row_starts.append(max(0, self.height - self.chip_size[0]))
            else:
                # If the image is smaller than chip size, just start at 0
                if not self.row_starts:
                    self.row_starts.append(0)

            # Normal column starts using stride
            for c in range((self.width - 1) // self.stride_x):
                self.col_starts.append(c * self.stride_x)

            # Add a special last column that ensures we reach the right edge
            if self.width > self.chip_size[1]:
                self.col_starts.append(max(0, self.width - self.chip_size[1]))
            else:
                # If the image is smaller than chip size, just start at 0
                if not self.col_starts:
                    self.col_starts.append(0)

            # Update rows and cols based on actual starting positions
            self.rows = len(self.row_starts)
            self.cols = len(self.col_starts)

            print(
                f"Dataset initialized with {self.rows} rows and {self.cols} columns of chips"
            )
            print(f"Image dimensions: {self.width} x {self.height} pixels")
            print(f"Chip size: {self.chip_size[1]} x {self.chip_size[0]} pixels")
            print(
                f"Overlap: {overlap*100}% (stride_x={self.stride_x}, stride_y={self.stride_y})"
            )
            if src.crs:
                print(f"CRS: {src.crs}")

        # Get raster stats
        self.raster_stats = get_raster_stats(raster_path, divide_by=255)

    def __getitem__(self, idx):
        """
        Get an image chip from the dataset by index.

        Retrieves an image tile with the specified overlap pattern, ensuring
        proper coverage of the entire raster including edges.

        Args:
            idx: Index of the chip to retrieve.

        Returns:
            dict: Dictionary containing:
                - image: Image tensor.
                - bbox: Geographic bounding box for the window.
                - coords: Pixel coordinates as tensor [i, j].
                - window_size: Window size as tensor [width, height].
        """
        # Convert flat index to grid position
        row = idx // self.cols
        col = idx % self.cols

        # Get pre-calculated starting positions
        j = self.row_starts[row]
        i = self.col_starts[col]

        # Read window from raster
        with rasterio.open(self.raster_path) as src:
            # Make sure we don't read outside the image
            width = min(self.chip_size[1], self.width - i)
            height = min(self.chip_size[0], self.height - j)

            window = Window(i, j, width, height)
            image = src.read(window=window)

            # Handle RGBA or multispectral images - keep only first 3 bands
            if image.shape[0] > 3:
                if not self.warned_about_bands and self.verbose:
                    print(f"Image has {image.shape[0]} bands, using first 3 bands only")
                    self.warned_about_bands = True
                if self.band_indexes is not None:
                    image = image[self.band_indexes]
                else:
                    image = image[:3]
            elif image.shape[0] < 3:
                # If image has fewer than 3 bands, duplicate the last band to make 3
                if not self.warned_about_bands and self.verbose:
                    print(
                        f"Image has {image.shape[0]} bands, duplicating bands to make 3"
                    )
                    self.warned_about_bands = True
                temp = np.zeros((3, image.shape[1], image.shape[2]), dtype=image.dtype)
                for c in range(3):
                    temp[c] = image[min(c, image.shape[0] - 1)]
                image = temp

            # Handle partial windows at edges by padding
            if (
                image.shape[1] != self.chip_size[0]
                or image.shape[2] != self.chip_size[1]
            ):
                temp = np.zeros(
                    (image.shape[0], self.chip_size[0], self.chip_size[1]),
                    dtype=image.dtype,
                )
                temp[:, : image.shape[1], : image.shape[2]] = image
                image = temp

        # Convert to format expected by model (C,H,W)
        image = torch.from_numpy(image).float()

        # Normalize to [0, 1]
        if image.max() > 1:
            image = image / 255.0

        # Apply transforms if any
        if self.transforms is not None:
            image = self.transforms(image)

        # Create geographic bounding box for the window
        minx, miny = self.transform * (i, j + height)
        maxx, maxy = self.transform * (i + width, j)
        bbox = box(minx, miny, maxx, maxy)

        return {
            "image": image,
            "bbox": bbox,
            "coords": torch.tensor([i, j], dtype=torch.long),  # Consistent format
            "window_size": torch.tensor(
                [width, height], dtype=torch.long
            ),  # Consistent format
        }

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of tiles in the dataset.
        """
        return self.rows * self.cols


class ObjectDetector:
    """
    Object extraction using Mask R-CNN with TorchGeo.
    """

    def __init__(
        self, model_path=None, repo_id=None, model=None, num_classes=2, device=None
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Hugging Face repository ID for model download.
            model: Pre-initialized model object (optional).
            num_classes: Number of classes for detection (default: 2).
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default parameters for object detection - these can be overridden in process_raster
        self.chip_size = (512, 512)  # Size of image chips for processing
        self.overlap = 0.25  # Default overlap between tiles
        self.confidence_threshold = 0.5  # Default confidence threshold
        self.nms_iou_threshold = 0.5  # IoU threshold for non-maximum suppression
        self.min_object_area = 100  # Minimum area in pixels to keep an object
        self.max_object_area = None  # Maximum area in pixels to keep an object
        self.mask_threshold = 0.5  # Threshold for mask binarization
        self.simplify_tolerance = 1.0  # Tolerance for polygon simplification

        # Initialize model
        self.model = self.initialize_model(model, num_classes=num_classes)

        # Download model if needed
        if model_path is None or (not os.path.exists(model_path)):
            model_path = self.download_model_from_hf(model_path, repo_id)

        # Load model weights
        self.load_weights(model_path)

        # Set model to evaluation mode
        self.model.eval()

    def download_model_from_hf(self, model_path=None, repo_id=None):
        """
        Download the object detection model from Hugging Face.

        Args:
            model_path: Path to the model file.
            repo_id: Hugging Face repository ID.

        Returns:
            Path to the downloaded model file
        """
        try:

            print("Model path not specified, downloading from Hugging Face...")

            # Define the repository ID and model filename
            if repo_id is None:
                repo_id = "giswqs/geoai"

            if model_path is None:
                model_path = "building_footprints_usa.pth"

            # Download the model
            model_path = hf_hub_download(repo_id=repo_id, filename=model_path)
            print(f"Model downloaded to: {model_path}")

            return model_path

        except Exception as e:
            print(f"Error downloading model from Hugging Face: {e}")
            print("Please specify a local model path or ensure internet connectivity.")
            raise

    def initialize_model(self, model, num_classes=2):
        """Initialize a deep learning model for object detection.

        Args:
            model (torch.nn.Module): A pre-initialized model object.
            num_classes (int): Number of classes for detection.

        Returns:
            torch.nn.Module: A deep learning model for object detection.
        """

        if model is None:  # Initialize Mask R-CNN model with ResNet50 backbone.
            # Standard image mean and std for pre-trained models
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]

            # Create model with explicit normalization parameters
            model = maskrcnn_resnet50_fpn(
                weights=None,
                progress=False,
                num_classes=num_classes,  # Background + object
                weights_backbone=None,
                # These parameters ensure consistent normalization
                image_mean=image_mean,
                image_std=image_std,
            )

        model.to(self.device)
        return model

    def load_weights(self, model_path):
        """
        Load weights from file with error handling for different formats.

        Args:
            model_path: Path to model weights
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            # Try to load state dict
            try:
                self.model.load_state_dict(state_dict)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting to fix state_dict keys...")

                # Try to fix state_dict keys (remove module prefix if needed)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict)
                print("Model loaded successfully after key fixing")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def mask_to_polygons(self, mask, **kwargs):
        """
        Convert binary mask to polygon contours using OpenCV.

        Args:
            mask: Binary mask as numpy array
            **kwargs: Optional parameters:
                simplify_tolerance: Tolerance for polygon simplification
                mask_threshold: Threshold for mask binarization
                min_object_area: Minimum area in pixels to keep an object
                max_object_area: Maximum area in pixels to keep an object

        Returns:
            List of polygons as lists of (x, y) coordinates
        """

        # Get parameters from kwargs or use instance defaults
        simplify_tolerance = kwargs.get("simplify_tolerance", self.simplify_tolerance)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        min_object_area = kwargs.get("min_object_area", self.min_object_area)
        max_object_area = kwargs.get("max_object_area", self.max_object_area)

        # Ensure binary mask
        mask = (mask > mask_threshold).astype(np.uint8)

        # Optional: apply morphological operations to improve mask quality
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to list of [x, y] coordinates
        polygons = []
        for contour in contours:
            # Filter out too small contours
            if contour.shape[0] < 3 or cv2.contourArea(contour) < min_object_area:
                continue

            # Filter out too large contours
            if (
                max_object_area is not None
                and cv2.contourArea(contour) > max_object_area
            ):
                continue

            # Simplify contour if it has many points
            if contour.shape[0] > 50:
                epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)

            # Convert to list of [x, y] coordinates
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)

        return polygons

    def filter_overlapping_polygons(self, gdf, **kwargs):
        """
        Filter overlapping polygons using non-maximum suppression.

        Args:
            gdf: GeoDataFrame with polygons
            **kwargs: Optional parameters:
                nms_iou_threshold: IoU threshold for filtering

        Returns:
            Filtered GeoDataFrame
        """
        if len(gdf) <= 1:
            return gdf

        # Get parameters from kwargs or use instance defaults
        iou_threshold = kwargs.get("nms_iou_threshold", self.nms_iou_threshold)

        # Sort by confidence
        gdf = gdf.sort_values("confidence", ascending=False)

        # Fix any invalid geometries
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )

        keep_indices = []
        polygons = gdf.geometry.values

        for i in range(len(polygons)):
            if i in keep_indices:
                continue

            keep = True
            for j in keep_indices:
                # Skip invalid geometries
                if not polygons[i].is_valid or not polygons[j].is_valid:
                    continue

                # Calculate IoU
                try:
                    intersection = polygons[i].intersection(polygons[j]).area
                    union = polygons[i].area + polygons[j].area - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        keep = False
                        break
                except Exception:
                    # Skip on topology exceptions
                    continue

            if keep:
                keep_indices.append(i)

        return gdf.iloc[keep_indices]

    def filter_edge_objects(self, gdf, raster_path, edge_buffer=10):
        """
        Filter out object detections that fall in padding/edge areas of the image.

        Args:
            gdf: GeoDataFrame with object detections
            raster_path: Path to the original raster file
            edge_buffer: Buffer in pixels to consider as edge region

        Returns:
            GeoDataFrame with filtered objects
        """
        import rasterio
        from shapely.geometry import box

        # If no objects detected, return empty GeoDataFrame
        if gdf is None or len(gdf) == 0:
            return gdf

        print(f"Objects before filtering: {len(gdf)}")

        with rasterio.open(raster_path) as src:
            # Get raster bounds
            raster_bounds = src.bounds
            raster_width = src.width
            raster_height = src.height

            # Convert edge buffer from pixels to geographic units
            # We need the smallest dimension of a pixel in geographic units
            pixel_width = (raster_bounds[2] - raster_bounds[0]) / raster_width
            pixel_height = (raster_bounds[3] - raster_bounds[1]) / raster_height
            buffer_size = min(pixel_width, pixel_height) * edge_buffer

            # Create a slightly smaller bounding box to exclude edge regions
            inner_bounds = (
                raster_bounds[0] + buffer_size,  # min x (west)
                raster_bounds[1] + buffer_size,  # min y (south)
                raster_bounds[2] - buffer_size,  # max x (east)
                raster_bounds[3] - buffer_size,  # max y (north)
            )

            # Check that inner bounds are valid
            if inner_bounds[0] >= inner_bounds[2] or inner_bounds[1] >= inner_bounds[3]:
                print("Warning: Edge buffer too large, using original bounds")
                inner_box = box(*raster_bounds)
            else:
                inner_box = box(*inner_bounds)

            # Filter out objects that intersect with the edge of the image
            filtered_gdf = gdf[gdf.intersects(inner_box)]

            # Additional check for objects that have >50% of their area outside the valid region
            valid_objects = []
            for idx, row in filtered_gdf.iterrows():
                if row.geometry.intersection(inner_box).area >= 0.5 * row.geometry.area:
                    valid_objects.append(idx)

            filtered_gdf = filtered_gdf.loc[valid_objects]

            print(f"Objects after filtering: {len(filtered_gdf)}")

            return filtered_gdf

    def masks_to_vector(
        self,
        mask_path,
        output_path=None,
        simplify_tolerance=None,
        mask_threshold=None,
        min_object_area=None,
        max_object_area=None,
        nms_iou_threshold=None,
        regularize=True,
        angle_threshold=15,
        rectangularity_threshold=0.7,
    ):
        """
        Convert an object mask GeoTIFF to vector polygons and save as GeoJSON.

        Args:
            mask_path: Path to the object masks GeoTIFF
            output_path: Path to save the output GeoJSON or Parquet file (default: mask_path with .geojson extension)
            simplify_tolerance: Tolerance for polygon simplification (default: self.simplify_tolerance)
            mask_threshold: Threshold for mask binarization (default: self.mask_threshold)
            min_object_area: Minimum area in pixels to keep an object (default: self.min_object_area)
            max_object_area: Minimum area in pixels to keep an object (default: self.max_object_area)
            nms_iou_threshold: IoU threshold for non-maximum suppression (default: self.nms_iou_threshold)
            regularize: Whether to regularize objects to right angles (default: True)
            angle_threshold: Maximum deviation from 90 degrees for regularization (default: 15)
            rectangularity_threshold: Threshold for rectangle simplification (default: 0.7)

        Returns:
            GeoDataFrame with objects
        """
        # Use class defaults if parameters not provided
        simplify_tolerance = (
            simplify_tolerance
            if simplify_tolerance is not None
            else self.simplify_tolerance
        )
        mask_threshold = (
            mask_threshold if mask_threshold is not None else self.mask_threshold
        )
        min_object_area = (
            min_object_area if min_object_area is not None else self.min_object_area
        )
        max_object_area = (
            max_object_area if max_object_area is not None else self.max_object_area
        )
        nms_iou_threshold = (
            nms_iou_threshold
            if nms_iou_threshold is not None
            else self.nms_iou_threshold
        )

        # Set default output path if not provided
        # if output_path is None:
        #     output_path = os.path.splitext(mask_path)[0] + ".geojson"

        print(f"Converting mask to GeoJSON with parameters:")
        print(f"- Mask threshold: {mask_threshold}")
        print(f"- Min object area: {min_object_area}")
        print(f"- Max object area: {max_object_area}")
        print(f"- Simplify tolerance: {simplify_tolerance}")
        print(f"- NMS IoU threshold: {nms_iou_threshold}")
        print(f"- Regularize objects: {regularize}")
        if regularize:
            print(f"- Angle threshold: {angle_threshold}° from 90°")
            print(f"- Rectangularity threshold: {rectangularity_threshold*100}%")

        # Open the mask raster
        with rasterio.open(mask_path) as src:
            # Read the mask data
            mask_data = src.read(1)
            transform = src.transform
            crs = src.crs

            # Print mask statistics
            print(f"Mask dimensions: {mask_data.shape}")
            print(f"Mask value range: {mask_data.min()} to {mask_data.max()}")

            # Prepare for connected component analysis
            # Binarize the mask based on threshold
            binary_mask = (mask_data > (mask_threshold * 255)).astype(np.uint8)

            # Apply morphological operations for better results (optional)
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            print(
                f"Found {num_labels-1} potential objects"
            )  # Subtract 1 for background

            # Create list to store polygons and confidence values
            all_polygons = []
            all_confidences = []

            # Process each component (skip the first one which is background)
            for i in tqdm(range(1, num_labels)):
                # Extract this object
                area = stats[i, cv2.CC_STAT_AREA]

                # Skip if too small
                if area < min_object_area:
                    continue

                # Skip if too large
                if max_object_area is not None and area > max_object_area:
                    continue

                # Create a mask for this object
                object_mask = (labels == i).astype(np.uint8)

                # Find contours
                contours, _ = cv2.findContours(
                    object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Process each contour
                for contour in contours:
                    # Skip if too few points
                    if contour.shape[0] < 3:
                        continue

                    # Simplify contour if it has many points
                    if contour.shape[0] > 50 and simplify_tolerance > 0:
                        epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)

                    # Convert to list of (x, y) coordinates
                    polygon_points = contour.reshape(-1, 2)

                    # Convert pixel coordinates to geographic coordinates
                    geo_points = []
                    for x, y in polygon_points:
                        gx, gy = transform * (x, y)
                        geo_points.append((gx, gy))

                    # Create Shapely polygon
                    if len(geo_points) >= 3:
                        try:
                            shapely_poly = Polygon(geo_points)
                            if shapely_poly.is_valid and shapely_poly.area > 0:
                                all_polygons.append(shapely_poly)

                                # Calculate "confidence" as normalized size
                                # This is a proxy since we don't have model confidence scores
                                normalized_size = min(1.0, area / 1000)  # Cap at 1.0
                                all_confidences.append(normalized_size)
                        except Exception as e:
                            print(f"Error creating polygon: {e}")

            print(f"Created {len(all_polygons)} valid polygons")

            # Create GeoDataFrame
            if not all_polygons:
                print("No valid polygons found")
                return None

            gdf = gpd.GeoDataFrame(
                {
                    "geometry": all_polygons,
                    "confidence": all_confidences,
                    "class": 1,  # Object class
                },
                crs=crs,
            )

            # Apply non-maximum suppression to remove overlapping polygons
            gdf = self.filter_overlapping_polygons(
                gdf, nms_iou_threshold=nms_iou_threshold
            )

            print(f"Object count after NMS filtering: {len(gdf)}")

            # Apply regularization if requested
            if regularize and len(gdf) > 0:
                # Convert pixel area to geographic units for min_area parameter
                # Estimate pixel size in geographic units
                with rasterio.open(mask_path) as src:
                    pixel_size_x = src.transform[
                        0
                    ]  # width of a pixel in geographic units
                    pixel_size_y = abs(
                        src.transform[4]
                    )  # height of a pixel in geographic units
                    avg_pixel_area = pixel_size_x * pixel_size_y

                # Use 10 pixels as minimum area in geographic units
                min_geo_area = 10 * avg_pixel_area

                # Regularize objects
                gdf = self.regularize_objects(
                    gdf,
                    min_area=min_geo_area,
                    angle_threshold=angle_threshold,
                    rectangularity_threshold=rectangularity_threshold,
                )

            # Save to file
            if output_path:
                if output_path.endswith(".parquet"):
                    gdf.to_parquet(output_path)
                else:
                    gdf.to_file(output_path)
                print(f"Saved {len(gdf)} objects to {output_path}")

            return gdf

    @torch.no_grad()
    def process_raster(
        self,
        raster_path,
        output_path=None,
        batch_size=4,
        filter_edges=True,
        edge_buffer=20,
        band_indexes=None,
        **kwargs,
    ):
        """
        Process a raster file to extract objects with customizable parameters.

        Args:
            raster_path: Path to input raster file
            output_path: Path to output GeoJSON or Parquet file (optional)
            batch_size: Batch size for processing
            filter_edges: Whether to filter out objects at the edges of the image
            edge_buffer: Size of edge buffer in pixels to filter out objects (if filter_edges=True)
            band_indexes: List of band indexes to use (if None, use all bands)
            **kwargs: Additional parameters:
                confidence_threshold: Minimum confidence score to keep a detection (0.0-1.0)
                overlap: Overlap between adjacent tiles (0.0-1.0)
                chip_size: Size of image chips for processing (height, width)
                nms_iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0)
                mask_threshold: Threshold for mask binarization (0.0-1.0)
                min_object_area: Minimum area in pixels to keep an object
                simplify_tolerance: Tolerance for polygon simplification

        Returns:
            GeoDataFrame with objects
        """
        # Get parameters from kwargs or use instance defaults
        confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        overlap = kwargs.get("overlap", self.overlap)
        chip_size = kwargs.get("chip_size", self.chip_size)
        nms_iou_threshold = kwargs.get("nms_iou_threshold", self.nms_iou_threshold)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        min_object_area = kwargs.get("min_object_area", self.min_object_area)
        max_object_area = kwargs.get("max_object_area", self.max_object_area)
        simplify_tolerance = kwargs.get("simplify_tolerance", self.simplify_tolerance)

        # Print parameters being used
        print(f"Processing with parameters:")
        print(f"- Confidence threshold: {confidence_threshold}")
        print(f"- Tile overlap: {overlap}")
        print(f"- Chip size: {chip_size}")
        print(f"- NMS IoU threshold: {nms_iou_threshold}")
        print(f"- Mask threshold: {mask_threshold}")
        print(f"- Min object area: {min_object_area}")
        print(f"- Max object area: {max_object_area}")
        print(f"- Simplify tolerance: {simplify_tolerance}")
        print(f"- Filter edge objects: {filter_edges}")
        if filter_edges:
            print(f"- Edge buffer size: {edge_buffer} pixels")

        # Create dataset
        dataset = CustomDataset(
            raster_path=raster_path,
            chip_size=chip_size,
            overlap=overlap,
            band_indexes=band_indexes,
        )
        self.raster_stats = dataset.raster_stats

        # Custom collate function to handle Shapely objects
        def custom_collate(batch):
            """
            Custom collate function that handles Shapely geometries
            by keeping them as Python objects rather than trying to collate them.
            """
            elem = batch[0]
            if isinstance(elem, dict):
                result = {}
                for key in elem:
                    if key == "bbox":
                        # Don't collate shapely objects, keep as list
                        result[key] = [d[key] for d in batch]
                    else:
                        # For tensors and other collatable types
                        try:
                            result[key] = (
                                torch.utils.data._utils.collate.default_collate(
                                    [d[key] for d in batch]
                                )
                            )
                        except TypeError:
                            # Fall back to list for non-collatable types
                            result[key] = [d[key] for d in batch]
                return result
            else:
                # Default collate for non-dict types
                return torch.utils.data._utils.collate.default_collate(batch)

        # Create dataloader with simple indexing and custom collate
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate,
        )

        # Process batches
        all_polygons = []
        all_scores = []

        print(f"Processing raster with {len(dataloader)} batches")
        for batch in tqdm(dataloader):
            # Move images to device
            images = batch["image"].to(self.device)
            coords = batch["coords"]  # (i, j) coordinates in pixels
            bboxes = batch[
                "bbox"
            ]  # Geographic bounding boxes - now a list, not a tensor

            # Run inference
            predictions = self.model(images)

            # Process predictions
            for idx, prediction in enumerate(predictions):
                masks = prediction["masks"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                labels = prediction["labels"].cpu().numpy()

                # Skip if no predictions
                if len(scores) == 0:
                    continue

                # Filter by confidence threshold
                valid_indices = scores >= confidence_threshold
                masks = masks[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]

                # Skip if no valid predictions
                if len(scores) == 0:
                    continue

                # Get window coordinates
                # The coords might be in different formats depending on batch handling
                if isinstance(coords, list):
                    # If coords is a list of tuples
                    coord_item = coords[idx]
                    if isinstance(coord_item, tuple) and len(coord_item) == 2:
                        i, j = coord_item
                    elif isinstance(coord_item, torch.Tensor):
                        i, j = coord_item.cpu().numpy().tolist()
                    else:
                        print(f"Unexpected coords format: {type(coord_item)}")
                        continue
                elif isinstance(coords, torch.Tensor):
                    # If coords is a tensor of shape [batch_size, 2]
                    i, j = coords[idx].cpu().numpy().tolist()
                else:
                    print(f"Unexpected coords type: {type(coords)}")
                    continue

                # Get window size
                if isinstance(batch["window_size"], list):
                    window_item = batch["window_size"][idx]
                    if isinstance(window_item, tuple) and len(window_item) == 2:
                        window_width, window_height = window_item
                    elif isinstance(window_item, torch.Tensor):
                        window_width, window_height = window_item.cpu().numpy().tolist()
                    else:
                        print(f"Unexpected window_size format: {type(window_item)}")
                        continue
                elif isinstance(batch["window_size"], torch.Tensor):
                    window_width, window_height = (
                        batch["window_size"][idx].cpu().numpy().tolist()
                    )
                else:
                    print(f"Unexpected window_size type: {type(batch['window_size'])}")
                    continue

                # Process masks to polygons
                for mask_idx, mask in enumerate(masks):
                    # Get binary mask
                    binary_mask = mask[0]  # Get binary mask

                    # Convert mask to polygon with custom parameters
                    contours = self.mask_to_polygons(
                        binary_mask,
                        simplify_tolerance=simplify_tolerance,
                        mask_threshold=mask_threshold,
                        min_object_area=min_object_area,
                        max_object_area=max_object_area,
                    )

                    # Skip if no valid polygons
                    if not contours:
                        continue

                    # Transform polygons to geographic coordinates
                    with rasterio.open(raster_path) as src:
                        transform = src.transform

                        for contour in contours:
                            # Convert polygon to global coordinates
                            global_polygon = []
                            for x, y in contour:
                                # Adjust coordinates based on window position
                                gx, gy = transform * (i + x, j + y)
                                global_polygon.append((gx, gy))

                            # Create Shapely polygon
                            if len(global_polygon) >= 3:
                                try:
                                    shapely_poly = Polygon(global_polygon)
                                    if shapely_poly.is_valid and shapely_poly.area > 0:
                                        all_polygons.append(shapely_poly)
                                        all_scores.append(float(scores[mask_idx]))
                                except Exception as e:
                                    print(f"Error creating polygon: {e}")

        # Create GeoDataFrame
        if not all_polygons:
            print("No valid polygons found")
            return None

        gdf = gpd.GeoDataFrame(
            {
                "geometry": all_polygons,
                "confidence": all_scores,
                "class": 1,  # Object class
            },
            crs=dataset.crs,
        )

        # Remove overlapping polygons with custom threshold
        gdf = self.filter_overlapping_polygons(gdf, nms_iou_threshold=nms_iou_threshold)

        # Filter edge objects if requested
        if filter_edges:
            gdf = self.filter_edge_objects(gdf, raster_path, edge_buffer=edge_buffer)

        # Save to file if requested
        if output_path:
            if output_path.endswith(".parquet"):
                gdf.to_parquet(output_path)
            else:
                gdf.to_file(output_path, driver="GeoJSON")
            print(f"Saved {len(gdf)} objects to {output_path}")

        return gdf

    def save_masks_as_geotiff(
        self, raster_path, output_path=None, batch_size=4, verbose=False, **kwargs
    ):
        """
        Process a raster file to extract object masks and save as GeoTIFF.

        Args:
            raster_path: Path to input raster file
            output_path: Path to output GeoTIFF file (optional, default: input_masks.tif)
            batch_size: Batch size for processing
            verbose: Whether to print detailed processing information
            **kwargs: Additional parameters:
                confidence_threshold: Minimum confidence score to keep a detection (0.0-1.0)
                chip_size: Size of image chips for processing (height, width)
                mask_threshold: Threshold for mask binarization (0.0-1.0)

        Returns:
            Path to the saved GeoTIFF file
        """

        # Get parameters from kwargs or use instance defaults
        confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        chip_size = kwargs.get("chip_size", self.chip_size)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        overlap = kwargs.get("overlap", self.overlap)

        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.splitext(raster_path)[0] + "_masks.tif"

        # Print parameters being used
        print(f"Processing masks with parameters:")
        print(f"- Confidence threshold: {confidence_threshold}")
        print(f"- Chip size: {chip_size}")
        print(f"- Mask threshold: {mask_threshold}")

        # Create dataset
        dataset = CustomDataset(
            raster_path=raster_path,
            chip_size=chip_size,
            overlap=overlap,
            verbose=verbose,
        )

        # Store a flag to avoid repetitive messages
        self.raster_stats = dataset.raster_stats
        seen_warnings = {
            "bands": False,
            "resize": {},  # Dictionary to track resize warnings by shape
        }

        # Open original raster to get metadata
        with rasterio.open(raster_path) as src:
            # Create output binary mask raster with same dimensions as input
            output_profile = src.profile.copy()
            output_profile.update(
                dtype=rasterio.uint8,
                count=1,  # Single band for object mask
                compress="lzw",
                nodata=0,
            )

            # Create output mask raster
            with rasterio.open(output_path, "w", **output_profile) as dst:
                # Initialize mask with zeros
                mask_array = np.zeros((src.height, src.width), dtype=np.uint8)

                # Custom collate function to handle Shapely objects
                def custom_collate(batch):
                    """Custom collate function for DataLoader"""
                    elem = batch[0]
                    if isinstance(elem, dict):
                        result = {}
                        for key in elem:
                            if key == "bbox":
                                # Don't collate shapely objects, keep as list
                                result[key] = [d[key] for d in batch]
                            else:
                                # For tensors and other collatable types
                                try:
                                    result[key] = (
                                        torch.utils.data._utils.collate.default_collate(
                                            [d[key] for d in batch]
                                        )
                                    )
                                except TypeError:
                                    # Fall back to list for non-collatable types
                                    result[key] = [d[key] for d in batch]
                        return result
                    else:
                        # Default collate for non-dict types
                        return torch.utils.data._utils.collate.default_collate(batch)

                # Create dataloader
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=custom_collate,
                )

                # Process batches
                print(f"Processing raster with {len(dataloader)} batches")
                for batch in tqdm(dataloader):
                    # Move images to device
                    images = batch["image"].to(self.device)
                    coords = batch["coords"]  # (i, j) coordinates in pixels

                    # Run inference
                    with torch.no_grad():
                        predictions = self.model(images)

                    # Process predictions
                    for idx, prediction in enumerate(predictions):
                        masks = prediction["masks"].cpu().numpy()
                        scores = prediction["scores"].cpu().numpy()

                        # Skip if no predictions
                        if len(scores) == 0:
                            continue

                        # Filter by confidence threshold
                        valid_indices = scores >= confidence_threshold
                        masks = masks[valid_indices]
                        scores = scores[valid_indices]

                        # Skip if no valid predictions
                        if len(scores) == 0:
                            continue

                        # Get window coordinates
                        if isinstance(coords, list):
                            coord_item = coords[idx]
                            if isinstance(coord_item, tuple) and len(coord_item) == 2:
                                i, j = coord_item
                            elif isinstance(coord_item, torch.Tensor):
                                i, j = coord_item.cpu().numpy().tolist()
                            else:
                                print(f"Unexpected coords format: {type(coord_item)}")
                                continue
                        elif isinstance(coords, torch.Tensor):
                            i, j = coords[idx].cpu().numpy().tolist()
                        else:
                            print(f"Unexpected coords type: {type(coords)}")
                            continue

                        # Get window size
                        if isinstance(batch["window_size"], list):
                            window_item = batch["window_size"][idx]
                            if isinstance(window_item, tuple) and len(window_item) == 2:
                                window_width, window_height = window_item
                            elif isinstance(window_item, torch.Tensor):
                                window_width, window_height = (
                                    window_item.cpu().numpy().tolist()
                                )
                            else:
                                print(
                                    f"Unexpected window_size format: {type(window_item)}"
                                )
                                continue
                        elif isinstance(batch["window_size"], torch.Tensor):
                            window_width, window_height = (
                                batch["window_size"][idx].cpu().numpy().tolist()
                            )
                        else:
                            print(
                                f"Unexpected window_size type: {type(batch['window_size'])}"
                            )
                            continue

                        # Combine all masks for this window
                        combined_mask = np.zeros(
                            (window_height, window_width), dtype=np.uint8
                        )

                        for mask in masks:
                            # Get the binary mask
                            binary_mask = (mask[0] > mask_threshold).astype(
                                np.uint8
                            ) * 255

                            # Handle size mismatch - resize binary_mask if needed
                            mask_h, mask_w = binary_mask.shape
                            if mask_h != window_height or mask_w != window_width:
                                resize_key = f"{(mask_h, mask_w)}->{(window_height, window_width)}"
                                if resize_key not in seen_warnings["resize"]:
                                    if verbose:
                                        print(
                                            f"Resizing mask from {binary_mask.shape} to {(window_height, window_width)}"
                                        )
                                    else:
                                        if not seen_warnings[
                                            "resize"
                                        ]:  # If this is the first resize warning
                                            print(
                                                f"Resizing masks at image edges (set verbose=True for details)"
                                            )
                                    seen_warnings["resize"][resize_key] = True

                                # Crop or pad the binary mask to match window size
                                resized_mask = np.zeros(
                                    (window_height, window_width), dtype=np.uint8
                                )
                                copy_h = min(mask_h, window_height)
                                copy_w = min(mask_w, window_width)
                                resized_mask[:copy_h, :copy_w] = binary_mask[
                                    :copy_h, :copy_w
                                ]
                                binary_mask = resized_mask

                            # Update combined mask (taking maximum where masks overlap)
                            combined_mask = np.maximum(combined_mask, binary_mask)

                        # Write combined mask to output array
                        # Handle edge cases where window might be smaller than chip size
                        h, w = combined_mask.shape
                        valid_h = min(h, src.height - j)
                        valid_w = min(w, src.width - i)

                        if valid_h > 0 and valid_w > 0:
                            mask_array[j : j + valid_h, i : i + valid_w] = np.maximum(
                                mask_array[j : j + valid_h, i : i + valid_w],
                                combined_mask[:valid_h, :valid_w],
                            )

                # Write the final mask to the output file
                dst.write(mask_array, 1)

        print(f"Object masks saved to {output_path}")
        return output_path

    def regularize_objects(
        self,
        gdf,
        min_area=10,
        angle_threshold=15,
        orthogonality_threshold=0.3,
        rectangularity_threshold=0.7,
    ):
        """
        Regularize objects to enforce right angles and rectangular shapes.

        Args:
            gdf: GeoDataFrame with objects
            min_area: Minimum area in square units to keep an object
            angle_threshold: Maximum deviation from 90 degrees to consider an angle as orthogonal (degrees)
            orthogonality_threshold: Percentage of angles that must be orthogonal for an object to be regularized
            rectangularity_threshold: Minimum area ratio to Object's oriented bounding box for rectangular simplification

        Returns:
            GeoDataFrame with regularized objects
        """
        import math

        import cv2
        import geopandas as gpd
        import numpy as np
        from shapely.affinity import rotate, translate
        from shapely.geometry import MultiPolygon, Polygon, box
        from tqdm import tqdm

        def get_angle(p1, p2, p3):
            """Calculate angle between three points in degrees (0-180)"""
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            # Handle numerical errors that could push cosine outside [-1, 1]
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine_angle))

            return angle

        def is_orthogonal(angle, threshold=angle_threshold):
            """Check if angle is close to 90 degrees"""
            return abs(angle - 90) <= threshold

        def calculate_dominant_direction(polygon):
            """Find the dominant direction of a polygon using PCA"""
            # Extract coordinates
            coords = np.array(polygon.exterior.coords)

            # Mean center the coordinates
            mean = np.mean(coords, axis=0)
            centered_coords = coords - mean

            # Calculate covariance matrix and its eigenvalues/eigenvectors
            cov_matrix = np.cov(centered_coords.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Get the index of the largest eigenvalue
            largest_idx = np.argmax(eigenvalues)

            # Get the corresponding eigenvector (principal axis)
            principal_axis = eigenvectors[:, largest_idx]

            # Calculate the angle in degrees
            angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
            angle_deg = np.degrees(angle_rad)

            # Normalize to range 0-180
            if angle_deg < 0:
                angle_deg += 180

            return angle_deg

        def create_oriented_envelope(polygon, angle_deg):
            """Create an oriented minimum area rectangle for the polygon"""
            # Create a rotated rectangle using OpenCV method (more robust than Shapely methods)
            coords = np.array(polygon.exterior.coords)[:-1].astype(
                np.float32
            )  # Skip the last point (same as first)

            # Use OpenCV's minAreaRect
            rect = cv2.minAreaRect(coords)
            box_points = cv2.boxPoints(rect)

            # Convert to shapely polygon
            oriented_box = Polygon(box_points)

            return oriented_box

        def get_rectangularity(polygon, oriented_box):
            """Calculate the rectangularity (area ratio to its oriented bounding box)"""
            if oriented_box.area == 0:
                return 0
            return polygon.area / oriented_box.area

        def check_orthogonality(polygon):
            """Check what percentage of angles in the polygon are orthogonal"""
            coords = list(polygon.exterior.coords)
            if len(coords) <= 4:  # Triangle or point
                return 0

            # Remove last point (same as first)
            coords = coords[:-1]

            orthogonal_count = 0
            total_angles = len(coords)

            for i in range(total_angles):
                p1 = coords[i]
                p2 = coords[(i + 1) % total_angles]
                p3 = coords[(i + 2) % total_angles]

                angle = get_angle(p1, p2, p3)
                if is_orthogonal(angle):
                    orthogonal_count += 1

            return orthogonal_count / total_angles

        def simplify_to_rectangle(polygon):
            """Simplify a polygon to a rectangle using its oriented bounding box"""
            # Get dominant direction
            angle = calculate_dominant_direction(polygon)

            # Create oriented envelope
            rect = create_oriented_envelope(polygon, angle)

            return rect

        if gdf is None or len(gdf) == 0:
            print("No Objects to regularize")
            return gdf

        print(f"Regularizing {len(gdf)} objects...")
        print(f"- Angle threshold: {angle_threshold}° from 90°")
        print(f"- Min orthogonality: {orthogonality_threshold*100}% of angles")
        print(
            f"- Min rectangularity: {rectangularity_threshold*100}% of bounding box area"
        )

        # Create a copy to avoid modifying the original
        result_gdf = gdf.copy()

        # Track statistics
        total_objects = len(gdf)
        regularized_count = 0
        rectangularized_count = 0

        # Process each Object
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
            geom = row.geometry

            # Skip invalid or empty geometries
            if geom is None or geom.is_empty:
                continue

            # Handle MultiPolygons by processing the largest part
            if isinstance(geom, MultiPolygon):
                areas = [p.area for p in geom.geoms]
                if not areas:
                    continue
                geom = list(geom.geoms)[np.argmax(areas)]

            # Filter out tiny Objects
            if geom.area < min_area:
                continue

            # Check orthogonality
            orthogonality = check_orthogonality(geom)

            # Create oriented envelope
            oriented_box = create_oriented_envelope(
                geom, calculate_dominant_direction(geom)
            )

            # Check rectangularity
            rectangularity = get_rectangularity(geom, oriented_box)

            # Decide how to regularize
            if rectangularity >= rectangularity_threshold:
                # Object is already quite rectangular, simplify to a rectangle
                result_gdf.at[idx, "geometry"] = oriented_box
                result_gdf.at[idx, "regularized"] = "rectangle"
                rectangularized_count += 1
            elif orthogonality >= orthogonality_threshold:
                # Object has many orthogonal angles but isn't rectangular
                # Could implement more sophisticated regularization here
                # For now, we'll still use the oriented rectangle
                result_gdf.at[idx, "geometry"] = oriented_box
                result_gdf.at[idx, "regularized"] = "orthogonal"
                regularized_count += 1
            else:
                # Object doesn't have clear orthogonal structure
                # Keep original but flag as unmodified
                result_gdf.at[idx, "regularized"] = "original"

        # Report statistics
        print(f"Regularization completed:")
        print(f"- Total objects: {total_objects}")
        print(
            f"- Rectangular objects: {rectangularized_count} ({rectangularized_count/total_objects*100:.1f}%)"
        )
        print(
            f"- Other regularized objects: {regularized_count} ({regularized_count/total_objects*100:.1f}%)"
        )
        print(
            f"- Unmodified objects: {total_objects-rectangularized_count-regularized_count} ({(total_objects-rectangularized_count-regularized_count)/total_objects*100:.1f}%)"
        )

        return result_gdf

    def visualize_results(
        self, raster_path, gdf=None, output_path=None, figsize=(12, 12)
    ):
        """
        Visualize object detection results with proper coordinate transformation.

        This function displays objects on top of the raster image,
        ensuring proper alignment between the GeoDataFrame polygons and the image.

        Args:
            raster_path: Path to input raster
            gdf: GeoDataFrame with object polygons (optional)
            output_path: Path to save visualization (optional)
            figsize: Figure size (width, height) in inches

        Returns:
            bool: True if visualization was successful
        """
        # Check if raster file exists
        if not os.path.exists(raster_path):
            print(f"Error: Raster file '{raster_path}' not found.")
            return False

        # Process raster if GeoDataFrame not provided
        if gdf is None:
            gdf = self.process_raster(raster_path)

        if gdf is None or len(gdf) == 0:
            print("No objects to visualize")
            return False

        # Check if confidence column exists in the GeoDataFrame
        has_confidence = False
        if hasattr(gdf, "columns") and "confidence" in gdf.columns:
            # Try to access a confidence value to confirm it works
            try:
                if len(gdf) > 0:
                    # Try getitem access
                    conf_val = gdf["confidence"].iloc[0]
                    has_confidence = True
                    print(
                        f"Using confidence values (range: {gdf['confidence'].min():.2f} - {gdf['confidence'].max():.2f})"
                    )
            except Exception as e:
                print(f"Confidence column exists but couldn't access values: {e}")
                has_confidence = False
        else:
            print("No confidence column found in GeoDataFrame")
            has_confidence = False

        # Read raster for visualization
        with rasterio.open(raster_path) as src:
            # Read the entire image or a subset if it's very large
            if src.height > 2000 or src.width > 2000:
                # Calculate scale factor to reduce size
                scale = min(2000 / src.height, 2000 / src.width)
                out_shape = (
                    int(src.count),
                    int(src.height * scale),
                    int(src.width * scale),
                )

                # Read and resample
                image = src.read(
                    out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
                )

                # Create a scaled transform for the resampled image
                # Calculate scaling factors
                x_scale = src.width / out_shape[2]
                y_scale = src.height / out_shape[1]

                # Get the original transform
                orig_transform = src.transform

                # Create a scaled transform
                scaled_transform = rasterio.transform.Affine(
                    orig_transform.a * x_scale,
                    orig_transform.b,
                    orig_transform.c,
                    orig_transform.d,
                    orig_transform.e * y_scale,
                    orig_transform.f,
                )
            else:
                image = src.read()
                scaled_transform = src.transform

            # Convert to RGB for display
            if image.shape[0] > 3:
                image = image[:3]
            elif image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)

            # Normalize image for display
            image = image.transpose(1, 2, 0)  # CHW to HWC
            image = image.astype(np.float32)

            if image.max() > 10:  # Likely 0-255 range
                image = image / 255.0

            image = np.clip(image, 0, 1)

            # Get image bounds
            bounds = src.bounds
            crs = src.crs

        # Create figure with appropriate aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]  # width / height
        plt.figure(figsize=(figsize[0], figsize[0] / aspect_ratio))
        ax = plt.gca()

        # Display image
        ax.imshow(image)

        # Make sure the GeoDataFrame has the same CRS as the raster
        if gdf.crs != crs:
            print(f"Reprojecting GeoDataFrame from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)

        # Set up colors for confidence visualization
        if has_confidence:
            try:
                import matplotlib.cm as cm
                from matplotlib.colors import Normalize

                # Get min/max confidence values
                min_conf = gdf["confidence"].min()
                max_conf = gdf["confidence"].max()

                # Set up normalization and colormap
                norm = Normalize(vmin=min_conf, vmax=max_conf)
                cmap = cm.viridis

                # Create scalar mappable for colorbar
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                # Add colorbar
                cbar = plt.colorbar(
                    sm, ax=ax, orientation="vertical", shrink=0.7, pad=0.01
                )
                cbar.set_label("Confidence Score")
            except Exception as e:
                print(f"Error setting up confidence visualization: {e}")
                has_confidence = False

        # Function to convert coordinates
        def geo_to_pixel(geometry, transform):
            """Convert geometry to pixel coordinates using the provided transform."""
            if geometry.is_empty:
                return None

            if geometry.geom_type == "Polygon":
                # Get exterior coordinates
                exterior_coords = list(geometry.exterior.coords)

                # Convert to pixel coordinates
                pixel_coords = [~transform * (x, y) for x, y in exterior_coords]

                # Split into x and y lists
                pixel_x = [coord[0] for coord in pixel_coords]
                pixel_y = [coord[1] for coord in pixel_coords]

                return pixel_x, pixel_y
            else:
                print(f"Unsupported geometry type: {geometry.geom_type}")
                return None

        # Plot each object
        for idx, row in gdf.iterrows():
            try:
                # Convert polygon to pixel coordinates
                coords = geo_to_pixel(row.geometry, scaled_transform)

                if coords:
                    pixel_x, pixel_y = coords

                    if has_confidence:
                        try:
                            # Get confidence value using different methods
                            # Method 1: Try direct attribute access
                            confidence = None
                            try:
                                confidence = row.confidence
                            except:
                                pass

                            # Method 2: Try dictionary-style access
                            if confidence is None:
                                try:
                                    confidence = row["confidence"]
                                except:
                                    pass

                            # Method 3: Try accessing by index from the GeoDataFrame
                            if confidence is None:
                                try:
                                    confidence = gdf.iloc[idx]["confidence"]
                                except:
                                    pass

                            if confidence is not None:
                                color = cmap(norm(confidence))
                                # Fill polygon with semi-transparent color
                                ax.fill(pixel_x, pixel_y, color=color, alpha=0.5)
                                # Draw border
                                ax.plot(
                                    pixel_x,
                                    pixel_y,
                                    color=color,
                                    linewidth=1,
                                    alpha=0.8,
                                )
                            else:
                                # Fall back to red if confidence value couldn't be accessed
                                ax.plot(pixel_x, pixel_y, color="red", linewidth=1)
                        except Exception as e:
                            print(
                                f"Error using confidence value for polygon {idx}: {e}"
                            )
                            ax.plot(pixel_x, pixel_y, color="red", linewidth=1)
                    else:
                        # No confidence data, just plot outlines in red
                        ax.plot(pixel_x, pixel_y, color="red", linewidth=1)
            except Exception as e:
                print(f"Error plotting polygon {idx}: {e}")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"objects (Found: {len(gdf)})")

        # Save if requested
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")

        plt.close()

        # Create a simpler visualization focused just on a subset of objects
        if len(gdf) > 0:
            plt.figure(figsize=figsize)
            ax = plt.gca()

            # Choose a subset of the image to show
            with rasterio.open(raster_path) as src:
                # Get centroid of first object
                sample_geom = gdf.iloc[0].geometry
                centroid = sample_geom.centroid

                # Convert to pixel coordinates
                center_x, center_y = ~src.transform * (centroid.x, centroid.y)

                # Define a window around this object
                window_size = 500  # pixels
                window = rasterio.windows.Window(
                    max(0, int(center_x - window_size / 2)),
                    max(0, int(center_y - window_size / 2)),
                    min(window_size, src.width - int(center_x - window_size / 2)),
                    min(window_size, src.height - int(center_y - window_size / 2)),
                )

                # Read this window
                sample_image = src.read(window=window)

                # Convert to RGB for display
                if sample_image.shape[0] > 3:
                    sample_image = sample_image[:3]
                elif sample_image.shape[0] == 1:
                    sample_image = np.repeat(sample_image, 3, axis=0)

                # Normalize image for display
                sample_image = sample_image.transpose(1, 2, 0)  # CHW to HWC
                sample_image = sample_image.astype(np.float32)

                if sample_image.max() > 10:  # Likely 0-255 range
                    sample_image = sample_image / 255.0

                sample_image = np.clip(sample_image, 0, 1)

                # Display sample image
                ax.imshow(sample_image, extent=[0, window.width, window.height, 0])

                # Get the correct transform for this window
                window_transform = src.window_transform(window)

                # Calculate bounds of the window
                window_bounds = rasterio.windows.bounds(window, src.transform)
                window_box = box(*window_bounds)

                # Filter objects that intersect with this window
                visible_gdf = gdf[gdf.intersects(window_box)]

                # Set up colors for sample view if confidence data exists
                if has_confidence:
                    try:
                        # Reuse the same normalization and colormap from main view
                        sample_sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                        sample_sm.set_array([])

                        # Add colorbar to sample view
                        sample_cbar = plt.colorbar(
                            sample_sm,
                            ax=ax,
                            orientation="vertical",
                            shrink=0.7,
                            pad=0.01,
                        )
                        sample_cbar.set_label("Confidence Score")
                    except Exception as e:
                        print(f"Error setting up sample confidence visualization: {e}")

                # Plot objects in sample view
                for idx, row in visible_gdf.iterrows():
                    try:
                        # Get window-relative pixel coordinates
                        geom = row.geometry

                        # Skip empty geometries
                        if geom.is_empty:
                            continue

                        # Get exterior coordinates
                        exterior_coords = list(geom.exterior.coords)

                        # Convert to pixel coordinates relative to window origin
                        pixel_coords = []
                        for x, y in exterior_coords:
                            px, py = ~src.transform * (x, y)  # Convert to image pixels
                            # Make coordinates relative to window
                            px = px - window.col_off
                            py = py - window.row_off
                            pixel_coords.append((px, py))

                        # Extract x and y coordinates
                        pixel_x = [coord[0] for coord in pixel_coords]
                        pixel_y = [coord[1] for coord in pixel_coords]

                        # Use confidence colors if available
                        if has_confidence:
                            try:
                                # Try different methods to access confidence
                                confidence = None
                                try:
                                    confidence = row.confidence
                                except:
                                    pass

                                if confidence is None:
                                    try:
                                        confidence = row["confidence"]
                                    except:
                                        pass

                                if confidence is None:
                                    try:
                                        confidence = visible_gdf.iloc[idx]["confidence"]
                                    except:
                                        pass

                                if confidence is not None:
                                    color = cmap(norm(confidence))
                                    # Fill polygon with semi-transparent color
                                    ax.fill(pixel_x, pixel_y, color=color, alpha=0.5)
                                    # Draw border
                                    ax.plot(
                                        pixel_x,
                                        pixel_y,
                                        color=color,
                                        linewidth=1.5,
                                        alpha=0.8,
                                    )
                                else:
                                    ax.plot(
                                        pixel_x, pixel_y, color="red", linewidth=1.5
                                    )
                            except Exception as e:
                                print(
                                    f"Error using confidence in sample view for polygon {idx}: {e}"
                                )
                                ax.plot(pixel_x, pixel_y, color="red", linewidth=1.5)
                        else:
                            ax.plot(pixel_x, pixel_y, color="red", linewidth=1.5)
                    except Exception as e:
                        print(f"Error plotting polygon in sample view: {e}")

                # Set title
                ax.set_title(f"Sample Area - objects (Showing: {len(visible_gdf)})")

                # Remove axes
                ax.set_xticks([])
                ax.set_yticks([])

                # Save if requested
                if output_path:
                    sample_output = (
                        os.path.splitext(output_path)[0]
                        + "_sample"
                        + os.path.splitext(output_path)[1]
                    )
                    plt.tight_layout()
                    plt.savefig(sample_output, dpi=300, bbox_inches="tight")
                    print(f"Sample visualization saved to {sample_output}")

    def generate_masks(
        self,
        raster_path,
        output_path=None,
        confidence_threshold=None,
        mask_threshold=None,
        min_object_area=10,
        max_object_area=float("inf"),
        overlap=0.25,
        batch_size=4,
        band_indexes=None,
        verbose=False,
        **kwargs,
    ):
        """
        Save masks with confidence values as a multi-band GeoTIFF.

        Objects with area smaller than min_object_area or larger than max_object_area
        will be filtered out.

        Args:
            raster_path: Path to input raster
            output_path: Path for output GeoTIFF
            confidence_threshold: Minimum confidence score (0.0-1.0)
            mask_threshold: Threshold for mask binarization (0.0-1.0)
            min_object_area: Minimum area (in pixels) for an object to be included
            max_object_area: Maximum area (in pixels) for an object to be included
            overlap: Overlap between tiles (0.0-1.0)
            batch_size: Batch size for processing
            band_indexes: List of band indexes to use (default: all bands)
            verbose: Whether to print detailed processing information

        Returns:
            Path to the saved GeoTIFF
        """
        # Use provided thresholds or fall back to instance defaults
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if mask_threshold is None:
            mask_threshold = self.mask_threshold

        chip_size = kwargs.get("chip_size", self.chip_size)

        # Default output path
        if output_path is None:
            output_path = os.path.splitext(raster_path)[0] + "_masks_conf.tif"

        # Process the raster to get individual masks with confidence
        with rasterio.open(raster_path) as src:
            # Create dataset with the specified overlap
            dataset = CustomDataset(
                raster_path=raster_path,
                chip_size=chip_size,
                overlap=overlap,
                band_indexes=band_indexes,
                verbose=verbose,
            )

            # Create output profile
            output_profile = src.profile.copy()
            output_profile.update(
                dtype=rasterio.uint8,
                count=2,  # Two bands: mask and confidence
                compress="lzw",
                nodata=0,
            )

            # Initialize mask and confidence arrays
            mask_array = np.zeros((src.height, src.width), dtype=np.uint8)
            conf_array = np.zeros((src.height, src.width), dtype=np.uint8)

            # Define custom collate function to handle Shapely objects
            def custom_collate(batch):
                """
                Custom collate function that handles Shapely geometries
                by keeping them as Python objects rather than trying to collate them.
                """
                elem = batch[0]
                if isinstance(elem, dict):
                    result = {}
                    for key in elem:
                        if key == "bbox":
                            # Don't collate shapely objects, keep as list
                            result[key] = [d[key] for d in batch]
                        else:
                            # For tensors and other collatable types
                            try:
                                result[key] = (
                                    torch.utils.data._utils.collate.default_collate(
                                        [d[key] for d in batch]
                                    )
                                )
                            except TypeError:
                                # Fall back to list for non-collatable types
                                result[key] = [d[key] for d in batch]
                    return result
                else:
                    # Default collate for non-dict types
                    return torch.utils.data._utils.collate.default_collate(batch)

            # Create dataloader with custom collate function
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate,
            )

            # Process batches
            print(f"Processing raster with {len(dataloader)} batches")
            for batch in tqdm(dataloader):
                # Move images to device
                images = batch["image"].to(self.device)
                coords = batch["coords"]  # Tensor of shape [batch_size, 2]

                # Run inference
                with torch.no_grad():
                    predictions = self.model(images)

                # Process predictions
                for idx, prediction in enumerate(predictions):
                    masks = prediction["masks"].cpu().numpy()
                    scores = prediction["scores"].cpu().numpy()

                    # Filter by confidence threshold
                    valid_indices = scores >= confidence_threshold
                    masks = masks[valid_indices]
                    scores = scores[valid_indices]

                    # Skip if no valid predictions
                    if len(masks) == 0:
                        continue

                    # Get window coordinates
                    i, j = coords[idx].cpu().numpy()

                    # Process each mask
                    for mask_idx, mask in enumerate(masks):
                        # Convert to binary mask
                        binary_mask = (mask[0] > mask_threshold).astype(np.uint8) * 255

                        # Check object area - calculate number of pixels in the mask
                        object_area = np.sum(binary_mask > 0)

                        # Skip objects that don't meet area criteria
                        if (
                            object_area < min_object_area
                            or object_area > max_object_area
                        ):
                            if verbose:
                                print(
                                    f"Filtering out object with area {object_area} pixels"
                                )
                            continue

                        conf_value = int(scores[mask_idx] * 255)  # Scale to 0-255

                        # Update the mask and confidence arrays
                        h, w = binary_mask.shape
                        valid_h = min(h, src.height - j)
                        valid_w = min(w, src.width - i)

                        if valid_h > 0 and valid_w > 0:
                            # Use maximum for overlapping regions in the mask
                            mask_array[j : j + valid_h, i : i + valid_w] = np.maximum(
                                mask_array[j : j + valid_h, i : i + valid_w],
                                binary_mask[:valid_h, :valid_w],
                            )

                            # For confidence, only update where mask is positive
                            # and confidence is higher than existing
                            mask_region = binary_mask[:valid_h, :valid_w] > 0
                            if np.any(mask_region):
                                # Only update where mask is positive and new confidence is higher
                                current_conf = conf_array[
                                    j : j + valid_h, i : i + valid_w
                                ]

                                # Where to update confidence (mask positive & higher confidence)
                                update_mask = np.logical_and(
                                    mask_region,
                                    np.logical_or(
                                        current_conf == 0, current_conf < conf_value
                                    ),
                                )

                                if np.any(update_mask):
                                    conf_array[j : j + valid_h, i : i + valid_w][
                                        update_mask
                                    ] = conf_value

            # Write to GeoTIFF
            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(mask_array, 1)
                dst.write(conf_array, 2)

            print(f"Masks with confidence values saved to {output_path}")
            return output_path

    def vectorize_masks(
        self,
        masks_path,
        output_path=None,
        confidence_threshold=0.5,
        min_object_area=100,
        max_object_area=None,
        n_workers=None,
        **kwargs,
    ):
        """
        Convert masks with confidence to vector polygons.

        Args:
            masks_path: Path to masks GeoTIFF with confidence band.
            output_path: Path for output GeoJSON.
            confidence_threshold: Minimum confidence score (0.0-1.0). Default: 0.5
            min_object_area: Minimum area in pixels to keep an object. Default: 100
            max_object_area: Maximum area in pixels to keep an object. Default: None
            n_workers: int, default=None
                The number of worker threads to use.
                "None" means single-threaded processing.
                "-1"   means using all available CPU processors.
                Positive integer means using that specific number of threads.
            **kwargs: Additional parameters

        Returns:
            GeoDataFrame with car detections and confidence values
        """

        def _process_single_component(
            component_mask,
            conf_data,
            transform,
            confidence_threshold,
            min_object_area,
            max_object_area,
        ):
            # Get confidence value
            conf_region = conf_data[component_mask > 0]
            if len(conf_region) > 0:
                confidence = np.mean(conf_region) / 255.0
            else:
                confidence = 0.0

            # Skip if confidence is below threshold
            if confidence < confidence_threshold:
                return None

            # Find contours
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            results = []

            for contour in contours:
                # Filter by size
                area = cv2.contourArea(contour)
                if area < min_object_area:
                    continue

                if max_object_area is not None and area > max_object_area:
                    continue

                # Get minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box_points = cv2.boxPoints(rect)

                # Convert to geographic coordinates
                geo_points = []
                for x, y in box_points:
                    gx, gy = transform * (x, y)
                    geo_points.append((gx, gy))

                # Create polygon
                poly = Polygon(geo_points)
                results.append((poly, confidence, area))

            return results

        import concurrent.futures
        from functools import partial

        def process_component(args):
            """
            Helper function to process a single component
            """
            (
                label,
                labeled_mask,
                conf_data,
                transform,
                confidence_threshold,
                min_object_area,
                max_object_area,
            ) = args

            # Create mask for this component
            component_mask = (labeled_mask == label).astype(np.uint8)

            return _process_single_component(
                component_mask,
                conf_data,
                transform,
                confidence_threshold,
                min_object_area,
                max_object_area,
            )

        start_time = time.time()
        print(f"Processing masks from: {masks_path}")

        if n_workers == -1:
            n_workers = os.cpu_count()

        with rasterio.open(masks_path) as src:
            # Read mask and confidence bands
            mask_data = src.read(1)
            conf_data = src.read(2)
            transform = src.transform
            crs = src.crs

            # Convert to binary mask
            binary_mask = mask_data > 0

            # Find connected components
            labeled_mask, num_features = ndimage.label(binary_mask)
            print(f"Found {num_features} connected components")

            # Process each component
            polygons = []
            confidences = []
            pixels = []

            if n_workers is None or n_workers == 1:
                print(
                    "Using single-threaded processing, you can speed up processing by setting n_workers > 1"
                )
                # Add progress bar
                for label in tqdm(
                    range(1, num_features + 1), desc="Processing components"
                ):
                    # Create mask for this component
                    component_mask = (labeled_mask == label).astype(np.uint8)

                    result = _process_single_component(
                        component_mask,
                        conf_data,
                        transform,
                        confidence_threshold,
                        min_object_area,
                        max_object_area,
                    )

                    if result:
                        for poly, confidence, area in result:
                            # Add to lists
                            polygons.append(poly)
                            confidences.append(confidence)
                            pixels.append(area)

            else:
                # Process components in parallel
                print(f"Using {n_workers} workers for parallel processing")

                process_args = [
                    (
                        label,
                        labeled_mask,
                        conf_data,
                        transform,
                        confidence_threshold,
                        min_object_area,
                        max_object_area,
                    )
                    for label in range(1, num_features + 1)
                ]

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers
                ) as executor:
                    results = list(
                        tqdm(
                            executor.map(process_component, process_args),
                            total=num_features,
                            desc="Processing components",
                        )
                    )

                    for result in results:
                        if result:
                            for poly, confidence, area in result:
                                # Add to lists
                                polygons.append(poly)
                                confidences.append(confidence)
                                pixels.append(area)

            # Create GeoDataFrame
            if polygons:
                gdf = gpd.GeoDataFrame(
                    {
                        "geometry": polygons,
                        "confidence": confidences,
                        "class": [1] * len(polygons),
                        "pixels": pixels,
                    },
                    crs=crs,
                )

                # Save to file if requested
                if output_path:
                    gdf.to_file(output_path, driver="GeoJSON")
                    print(f"Saved {len(gdf)} objects with confidence to {output_path}")

                end_time = time.time()
                print(f"Total processing time: {end_time - start_time:.2f} seconds")
                return gdf
            else:
                end_time = time.time()
                print(f"Total processing time: {end_time - start_time:.2f} seconds")
                print("No valid polygons found")
                return None


class BuildingFootprintExtractor(ObjectDetector):
    """
    Building footprint extraction using a pre-trained Mask R-CNN model.

    This class extends the
    `ObjectDetector` class with additional methods for building footprint extraction."
    """

    def __init__(
        self,
        model_path="building_footprints_usa.pth",
        repo_id=None,
        model=None,
        device=None,
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        super().__init__(
            model_path=model_path, repo_id=repo_id, model=model, device=device
        )

    def regularize_buildings(
        self,
        gdf,
        min_area=10,
        angle_threshold=15,
        orthogonality_threshold=0.3,
        rectangularity_threshold=0.7,
    ):
        """
        Regularize building footprints to enforce right angles and rectangular shapes.

        Args:
            gdf: GeoDataFrame with building footprints
            min_area: Minimum area in square units to keep a building
            angle_threshold: Maximum deviation from 90 degrees to consider an angle as orthogonal (degrees)
            orthogonality_threshold: Percentage of angles that must be orthogonal for a building to be regularized
            rectangularity_threshold: Minimum area ratio to building's oriented bounding box for rectangular simplification

        Returns:
            GeoDataFrame with regularized building footprints
        """
        return self.regularize_objects(
            gdf,
            min_area=min_area,
            angle_threshold=angle_threshold,
            orthogonality_threshold=orthogonality_threshold,
            rectangularity_threshold=rectangularity_threshold,
        )


class CarDetector(ObjectDetector):
    """
    Car detection using a pre-trained Mask R-CNN model.

    This class extends the `ObjectDetector` class with additional methods for car detection.
    """

    def __init__(
        self, model_path="car_detection_usa.pth", repo_id=None, model=None, device=None
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        super().__init__(
            model_path=model_path, repo_id=repo_id, model=model, device=device
        )


class ShipDetector(ObjectDetector):
    """
    Ship detection using a pre-trained Mask R-CNN model.

    This class extends the
    `ObjectDetector` class with additional methods for ship detection."
    """

    def __init__(
        self, model_path="ship_detection.pth", repo_id=None, model=None, device=None
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        super().__init__(
            model_path=model_path, repo_id=repo_id, model=model, device=device
        )


class SolarPanelDetector(ObjectDetector):
    """
    Solar panel detection using a pre-trained Mask R-CNN model.

    This class extends the
    `ObjectDetector` class with additional methods for solar panel detection."
    """

    def __init__(
        self,
        model_path="solar_panel_detection.pth",
        repo_id=None,
        model=None,
        device=None,
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        super().__init__(
            model_path=model_path, repo_id=repo_id, model=model, device=device
        )


class ParkingSplotDetector(ObjectDetector):
    """
    Car detection using a pre-trained Mask R-CNN model.

    This class extends the `ObjectDetector` class with additional methods for car detection.
    """

    def __init__(
        self,
        model_path="parking_spot_detection.pth",
        repo_id=None,
        model=None,
        num_classes=3,
        device=None,
    ):
        """
        Initialize the object extractor.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            num_classes: Number of classes for the model. Default: 3
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
        """
        super().__init__(
            model_path=model_path,
            repo_id=repo_id,
            model=model,
            num_classes=num_classes,
            device=device,
        )


class AgricultureFieldDelineator(ObjectDetector):
    """
    Agricultural field boundary delineation using a pre-trained Mask R-CNN model.

    This class extends the ObjectDetector class to specifically handle Sentinel-2
    imagery with 12 spectral bands for agricultural field boundary detection.

    Attributes:
        band_selection: List of band indices to use for prediction (default: RGB)
        sentinel_band_stats: Per-band statistics for Sentinel-2 data
        use_ndvi: Whether to calculate and include NDVI as an additional channel
    """

    def __init__(
        self,
        model_path="field_boundary_detector.pth",
        repo_id=None,
        model=None,
        device=None,
        band_selection=None,
        use_ndvi=False,
    ):
        """
        Initialize the field boundary delineator.

        Args:
            model_path: Path to the .pth model file.
            repo_id: Repo ID for loading models from the Hub.
            model: Custom model to use for inference.
            device: Device to use for inference ('cuda:0', 'cpu', etc.).
            band_selection: List of Sentinel-2 band indices to use (None = adapt based on model)
            use_ndvi: Whether to calculate and include NDVI as an additional channel
        """
        # Save parameters before calling parent constructor
        self.custom_band_selection = band_selection
        self.use_ndvi = use_ndvi

        # Set device (copied from parent init to ensure it's set before initialize_model)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model differently for multi-spectral input
        model = self.initialize_sentinel2_model(model)

        # Call parent but with our custom model
        super().__init__(
            model_path=model_path, repo_id=repo_id, model=model, device=device
        )

        # Default Sentinel-2 band statistics (can be overridden with actual stats)
        # Band order: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
        self.sentinel_band_stats = {
            "means": [
                0.0975,
                0.0476,
                0.0598,
                0.0697,
                0.1077,
                0.1859,
                0.2378,
                0.2061,
                0.2598,
                0.4120,
                0.1956,
                0.1410,
            ],
            "stds": [
                0.0551,
                0.0290,
                0.0298,
                0.0479,
                0.0506,
                0.0505,
                0.0747,
                0.0642,
                0.0782,
                0.1187,
                0.0651,
                0.0679,
            ],
        }

        # Set default band selection (RGB - typically B4, B3, B2 for Sentinel-2)
        self.band_selection = (
            self.custom_band_selection
            if self.custom_band_selection is not None
            else [3, 2, 1]
        )  # R, G, B bands

        # Customize parameters for field delineation
        self.confidence_threshold = 0.5  # Default confidence threshold
        self.overlap = 0.5  # Higher overlap for field boundary detection
        self.min_object_area = 1000  # Minimum area in pixels for field detection
        self.simplify_tolerance = 2.0  # Higher tolerance for field boundaries

    def initialize_sentinel2_model(self, model=None):
        """
        Initialize a Mask R-CNN model with a modified first layer to accept Sentinel-2 data.

        Args:
            model: Pre-initialized model (optional)

        Returns:
            Modified model with appropriate input channels
        """
        import torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        if model is not None:
            return model

        # Determine number of input channels based on band selection and NDVI
        num_input_channels = (
            len(self.custom_band_selection)
            if self.custom_band_selection is not None
            else 3
        )
        if self.use_ndvi:
            num_input_channels += 1

        print(f"Initializing Mask R-CNN model with {num_input_channels} input channels")

        # Create a ResNet50 backbone with modified input channels
        backbone = resnet_fpn_backbone("resnet50", weights=None)

        # Replace the first conv layer to accept multi-spectral input
        original_conv = backbone.body.conv1
        backbone.body.conv1 = torch.nn.Conv2d(
            num_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Create Mask R-CNN with our modified backbone
        model = maskrcnn_resnet50_fpn(
            backbone=backbone,
            num_classes=2,  # Background + field
            image_mean=[0.485] * num_input_channels,  # Extend mean to all channels
            image_std=[0.229] * num_input_channels,  # Extend std to all channels
        )

        model.to(self.device)
        return model

    def preprocess_sentinel_bands(self, image_data, band_selection=None, use_ndvi=None):
        """
        Preprocess Sentinel-2 band data for model input.

        Args:
            image_data: Raw Sentinel-2 image data as numpy array [bands, height, width]
            band_selection: List of band indices to use (overrides instance default if provided)
            use_ndvi: Whether to include NDVI (overrides instance default if provided)

        Returns:
            Processed tensor ready for model input
        """
        # Use instance defaults if not specified
        band_selection = (
            band_selection if band_selection is not None else self.band_selection
        )
        use_ndvi = use_ndvi if use_ndvi is not None else self.use_ndvi

        # Select bands
        selected_bands = image_data[band_selection]

        # Calculate NDVI if requested (using B8 and B4 which are indices 7 and 3)
        if (
            use_ndvi
            and 7 in range(image_data.shape[0])
            and 3 in range(image_data.shape[0])
        ):
            nir = image_data[7].astype(np.float32)  # B8 (NIR)
            red = image_data[3].astype(np.float32)  # B4 (Red)

            # Avoid division by zero
            denominator = nir + red
            ndvi = np.zeros_like(nir)
            valid_mask = denominator > 0
            ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[
                valid_mask
            ]

            # Rescale NDVI from [-1, 1] to [0, 1]
            ndvi = (ndvi + 1) / 2

            # Add NDVI as an additional channel
            selected_bands = np.vstack([selected_bands, ndvi[np.newaxis, :, :]])

        # Convert to tensor
        image_tensor = torch.from_numpy(selected_bands).float()

        # Normalize using band statistics
        for i, band_idx in enumerate(band_selection):
            # Make sure band_idx is within range of our statistics
            if band_idx < len(self.sentinel_band_stats["means"]):
                mean = self.sentinel_band_stats["means"][band_idx]
                std = self.sentinel_band_stats["stds"][band_idx]
                image_tensor[i] = (image_tensor[i] - mean) / std

        # If NDVI was added, normalize it too (last channel)
        if use_ndvi:
            # NDVI is already roughly in [0,1] range, just standardize it slightly
            image_tensor[-1] = (image_tensor[-1] - 0.5) / 0.5

        return image_tensor

    def update_band_stats(self, raster_path, band_selection=None, sample_size=1000):
        """
        Update band statistics from the input Sentinel-2 raster.

        Args:
            raster_path: Path to the Sentinel-2 raster file
            band_selection: Specific bands to update (None = update all available)
            sample_size: Number of random pixels to sample for statistics calculation

        Returns:
            Updated band statistics dictionary
        """
        with rasterio.open(raster_path) as src:
            # Check if this is likely a Sentinel-2 product
            band_count = src.count
            if band_count < 3:
                print(
                    f"Warning: Raster has only {band_count} bands, may not be Sentinel-2 data"
                )

            # Get dimensions
            height, width = src.height, src.width

            # Determine which bands to analyze
            if band_selection is None:
                band_selection = list(range(1, band_count + 1))  # 1-indexed

            # Initialize arrays for band statistics
            means = []
            stds = []

            # Sample random pixels
            np.random.seed(42)  # For reproducibility
            sample_rows = np.random.randint(0, height, sample_size)
            sample_cols = np.random.randint(0, width, sample_size)

            # Calculate statistics for each band
            for band in band_selection:
                # Read band data
                band_data = src.read(band)

                # Sample values
                sample_values = band_data[sample_rows, sample_cols]

                # Remove invalid values (e.g., nodata)
                valid_samples = sample_values[np.isfinite(sample_values)]

                # Calculate statistics
                mean = float(np.mean(valid_samples))
                std = float(np.std(valid_samples))

                # Store results
                means.append(mean)
                stds.append(std)

                print(f"Band {band}: mean={mean:.4f}, std={std:.4f}")

            # Update instance variables
            self.sentinel_band_stats = {"means": means, "stds": stds}

            return self.sentinel_band_stats

    def process_sentinel_raster(
        self,
        raster_path,
        output_path=None,
        batch_size=4,
        band_selection=None,
        use_ndvi=None,
        filter_edges=True,
        edge_buffer=20,
        **kwargs,
    ):
        """
        Process a Sentinel-2 raster to extract field boundaries.

        Args:
            raster_path: Path to Sentinel-2 raster file
            output_path: Path to output GeoJSON or Parquet file (optional)
            batch_size: Batch size for processing
            band_selection: List of bands to use (None = use instance default)
            use_ndvi: Whether to include NDVI (None = use instance default)
            filter_edges: Whether to filter out objects at the edges of the image
            edge_buffer: Size of edge buffer in pixels to filter out objects
            **kwargs: Additional parameters for processing

        Returns:
            GeoDataFrame with field boundaries
        """
        # Use instance defaults if not specified
        band_selection = (
            band_selection if band_selection is not None else self.band_selection
        )
        use_ndvi = use_ndvi if use_ndvi is not None else self.use_ndvi

        # Get parameters from kwargs or use instance defaults
        confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        overlap = kwargs.get("overlap", self.overlap)
        chip_size = kwargs.get("chip_size", self.chip_size)
        nms_iou_threshold = kwargs.get("nms_iou_threshold", self.nms_iou_threshold)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        min_object_area = kwargs.get("min_object_area", self.min_object_area)
        simplify_tolerance = kwargs.get("simplify_tolerance", self.simplify_tolerance)

        # Update band statistics if not already done
        if kwargs.get("update_stats", True):
            self.update_band_stats(raster_path, band_selection)

        print(f"Processing with parameters:")
        print(f"- Using bands: {band_selection}")
        print(f"- Include NDVI: {use_ndvi}")
        print(f"- Confidence threshold: {confidence_threshold}")
        print(f"- Tile overlap: {overlap}")
        print(f"- Chip size: {chip_size}")
        print(f"- Filter edge objects: {filter_edges}")

        # Create a custom Sentinel-2 dataset class
        class Sentinel2Dataset(torch.utils.data.Dataset):
            def __init__(
                self,
                raster_path,
                chip_size,
                stride_x,
                stride_y,
                band_selection,
                use_ndvi,
                field_delineator,
            ):
                self.raster_path = raster_path
                self.chip_size = chip_size
                self.stride_x = stride_x
                self.stride_y = stride_y
                self.band_selection = band_selection
                self.use_ndvi = use_ndvi
                self.field_delineator = field_delineator

                with rasterio.open(self.raster_path) as src:
                    self.height = src.height
                    self.width = src.width
                    self.count = src.count
                    self.crs = src.crs
                    self.transform = src.transform

                    # Calculate row_starts and col_starts
                    self.row_starts = []
                    self.col_starts = []

                    # Normal row starts using stride
                    for r in range((self.height - 1) // self.stride_y):
                        self.row_starts.append(r * self.stride_y)

                    # Add a special last row that ensures we reach the bottom edge
                    if self.height > self.chip_size[0]:
                        self.row_starts.append(max(0, self.height - self.chip_size[0]))
                    else:
                        # If the image is smaller than chip size, just start at 0
                        if not self.row_starts:
                            self.row_starts.append(0)

                    # Normal column starts using stride
                    for c in range((self.width - 1) // self.stride_x):
                        self.col_starts.append(c * self.stride_x)

                    # Add a special last column that ensures we reach the right edge
                    if self.width > self.chip_size[1]:
                        self.col_starts.append(max(0, self.width - self.chip_size[1]))
                    else:
                        # If the image is smaller than chip size, just start at 0
                        if not self.col_starts:
                            self.col_starts.append(0)

                # Calculate number of tiles
                self.rows = len(self.row_starts)
                self.cols = len(self.col_starts)

                print(
                    f"Dataset initialized with {self.rows} rows and {self.cols} columns of chips"
                )
                print(f"Image dimensions: {self.width} x {self.height} pixels")
                print(f"Chip size: {self.chip_size[1]} x {self.chip_size[0]} pixels")

            def __len__(self):
                return self.rows * self.cols

            def __getitem__(self, idx):
                # Convert flat index to grid position
                row = idx // self.cols
                col = idx % self.cols

                # Get pre-calculated starting positions
                j = self.row_starts[row]
                i = self.col_starts[col]

                # Read window from raster
                with rasterio.open(self.raster_path) as src:
                    # Make sure we don't read outside the image
                    width = min(self.chip_size[1], self.width - i)
                    height = min(self.chip_size[0], self.height - j)

                    window = Window(i, j, width, height)

                    # Read all bands
                    image = src.read(window=window)

                    # Handle partial windows at edges by padding
                    if (
                        image.shape[1] != self.chip_size[0]
                        or image.shape[2] != self.chip_size[1]
                    ):
                        temp = np.zeros(
                            (image.shape[0], self.chip_size[0], self.chip_size[1]),
                            dtype=image.dtype,
                        )
                        temp[:, : image.shape[1], : image.shape[2]] = image
                        image = temp

                # Preprocess bands for the model
                image_tensor = self.field_delineator.preprocess_sentinel_bands(
                    image, self.band_selection, self.use_ndvi
                )

                # Get geographic bounds for the window
                with rasterio.open(self.raster_path) as src:
                    window_transform = src.window_transform(window)
                    minx, miny = window_transform * (0, height)
                    maxx, maxy = window_transform * (width, 0)
                    bbox = [minx, miny, maxx, maxy]

                return {
                    "image": image_tensor,
                    "bbox": bbox,
                    "coords": torch.tensor([i, j], dtype=torch.long),
                    "window_size": torch.tensor([width, height], dtype=torch.long),
                }

        # Calculate stride based on overlap
        stride_x = int(chip_size[1] * (1 - overlap))
        stride_y = int(chip_size[0] * (1 - overlap))

        # Create dataset
        dataset = Sentinel2Dataset(
            raster_path=raster_path,
            chip_size=chip_size,
            stride_x=stride_x,
            stride_y=stride_y,
            band_selection=band_selection,
            use_ndvi=use_ndvi,
            field_delineator=self,
        )

        # Define custom collate function
        def custom_collate(batch):
            elem = batch[0]
            if isinstance(elem, dict):
                result = {}
                for key in elem:
                    if key == "bbox":
                        # Don't collate bbox objects, keep as list
                        result[key] = [d[key] for d in batch]
                    else:
                        # For tensors and other collatable types
                        try:
                            result[key] = (
                                torch.utils.data._utils.collate.default_collate(
                                    [d[key] for d in batch]
                                )
                            )
                        except TypeError:
                            # Fall back to list for non-collatable types
                            result[key] = [d[key] for d in batch]
                return result
            else:
                # Default collate for non-dict types
                return torch.utils.data._utils.collate.default_collate(batch)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate,
        )

        # Process batches (call the parent class's process_raster method)
        # We'll adapt the process_raster method to work with our Sentinel2Dataset
        results = super().process_raster(
            raster_path=raster_path,
            output_path=output_path,
            batch_size=batch_size,
            filter_edges=filter_edges,
            edge_buffer=edge_buffer,
            confidence_threshold=confidence_threshold,
            overlap=overlap,
            chip_size=chip_size,
            nms_iou_threshold=nms_iou_threshold,
            mask_threshold=mask_threshold,
            min_object_area=min_object_area,
            simplify_tolerance=simplify_tolerance,
        )

        return results
