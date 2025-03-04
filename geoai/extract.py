import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import geopandas as gpd
from tqdm import tqdm

import cv2
from torchgeo.datasets import NonGeoDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from huggingface_hub import hf_hub_download


class BuildingFootprintDataset(NonGeoDataset):
    """
    A TorchGeo dataset for building footprint extraction.
    Using NonGeoDataset to avoid spatial indexing issues.
    """

    def __init__(self, raster_path, chip_size=(512, 512), transforms=None):
        """
        Initialize the dataset.

        Args:
            raster_path: Path to the input raster file
            chip_size: Size of image chips to extract (height, width)
            transforms: Transforms to apply to the image
        """
        super().__init__()

        # Initialize parameters
        self.raster_path = raster_path
        self.chip_size = chip_size
        self.transforms = transforms

        # Open raster and get metadata
        with rasterio.open(self.raster_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.height = src.height
            self.width = src.width
            self.count = src.count

            # Define the bounds of the dataset
            west, south, east, north = src.bounds
            self.bounds = (west, south, east, north)

            # Define the ROI for the dataset
            self.roi = box(*self.bounds)

            # Calculate number of chips in each dimension
            self.rows = self.height // self.chip_size[0]
            self.cols = self.width // self.chip_size[1]

            print(
                f"Dataset initialized with {self.rows} rows and {self.cols} columns of chips"
            )
            if src.crs:
                print(f"CRS: {src.crs}")

    def __getitem__(self, idx):
        """
        Get an image chip from the dataset by index.

        Args:
            idx: Index of the chip

        Returns:
            Dict containing image tensor
        """
        # Convert flat index to grid position
        row = idx // self.cols
        col = idx % self.cols

        # Calculate pixel coordinates
        i = col * self.chip_size[1]
        j = row * self.chip_size[0]

        # Read window from raster
        with rasterio.open(self.raster_path) as src:
            # Make sure we don't read outside the image
            width = min(self.chip_size[1], self.width - i)
            height = min(self.chip_size[0], self.height - j)

            window = Window(i, j, width, height)
            image = src.read(window=window)

            # Handle RGBA or multispectral images - keep only first 3 bands
            if image.shape[0] > 3:
                print(f"Image has {image.shape[0]} bands, using first 3 bands only")
                image = image[:3]
            elif image.shape[0] < 3:
                # If image has fewer than 3 bands, duplicate the last band to make 3
                print(f"Image has {image.shape[0]} bands, duplicating bands to make 3")
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
        """Return the number of samples in the dataset."""
        return self.rows * self.cols


class BuildingFootprintExtractor:
    """
    Building footprint extraction using Mask R-CNN with TorchGeo.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize the building footprint extractor.

        Args:
            model_path: Path to the .pth model file
            device: Device to use for inference ('cuda:0', 'cpu', etc.)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default parameters for building detection - these can be overridden in process_raster
        self.chip_size = (512, 512)  # Size of image chips for processing
        self.overlap = 0.25  # Default overlap between tiles
        self.confidence_threshold = 0.5  # Default confidence threshold
        self.nms_iou_threshold = 0.5  # IoU threshold for non-maximum suppression
        self.small_building_area = 100  # Minimum area in pixels to keep a building
        self.mask_threshold = 0.5  # Threshold for mask binarization
        self.simplify_tolerance = 1.0  # Tolerance for polygon simplification

        # Initialize model
        self.model = self._initialize_model()

        # Download model if needed
        if model_path is None:
            model_path = self._download_model_from_hf()

        # Load model weights
        self._load_weights(model_path)

        # Set model to evaluation mode
        self.model.eval()

    def _download_model_from_hf(self):
        """
        Download the USA building footprints model from Hugging Face.

        Returns:
            Path to the downloaded model file
        """
        try:

            print("Model path not specified, downloading from Hugging Face...")

            # Define the repository ID and model filename
            repo_id = "giswqs/geoai"  # Update with your actual username/repo
            filename = "building_footprints_usa.pth"

            # Ensure cache directory exists
            # cache_dir = os.path.join(
            #     os.path.expanduser("~"), ".cache", "building_footprints"
            # )
            # os.makedirs(cache_dir, exist_ok=True)

            # Download the model
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"Model downloaded to: {model_path}")

            return model_path

        except Exception as e:
            print(f"Error downloading model from Hugging Face: {e}")
            print("Please specify a local model path or ensure internet connectivity.")
            raise

    def _initialize_model(self):
        """Initialize Mask R-CNN model with ResNet50 backbone."""
        # Standard image mean and std for pre-trained models
        # Note: This would normally come from your config file
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        # Create model with explicit normalization parameters
        model = maskrcnn_resnet50_fpn(
            weights=None,
            progress=False,
            num_classes=2,  # Background + building
            weights_backbone=None,
            # These parameters ensure consistent normalization
            image_mean=image_mean,
            image_std=image_std,
        )

        model.to(self.device)
        return model

    def _load_weights(self, model_path):
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

    def _mask_to_polygons(self, mask, **kwargs):
        """
        Convert binary mask to polygon contours using OpenCV.

        Args:
            mask: Binary mask as numpy array
            **kwargs: Optional parameters:
                simplify_tolerance: Tolerance for polygon simplification
                mask_threshold: Threshold for mask binarization
                small_building_area: Minimum area in pixels to keep a building

        Returns:
            List of polygons as lists of (x, y) coordinates
        """

        # Get parameters from kwargs or use instance defaults
        simplify_tolerance = kwargs.get("simplify_tolerance", self.simplify_tolerance)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        small_building_area = kwargs.get(
            "small_building_area", self.small_building_area
        )

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
            if contour.shape[0] < 3 or cv2.contourArea(contour) < small_building_area:
                continue

            # Simplify contour if it has many points
            if contour.shape[0] > 50:
                epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)

            # Convert to list of [x, y] coordinates
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)

        return polygons

    def _filter_overlapping_polygons(self, gdf, **kwargs):
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

    @torch.no_grad()
    def process_raster(self, raster_path, output_path=None, batch_size=4, **kwargs):
        """
        Process a raster file to extract building footprints with customizable parameters.

        Args:
            raster_path: Path to input raster file
            output_path: Path to output GeoJSON file (optional)
            batch_size: Batch size for processing
            **kwargs: Additional parameters:
                confidence_threshold: Minimum confidence score to keep a detection (0.0-1.0)
                overlap: Overlap between adjacent tiles (0.0-1.0)
                chip_size: Size of image chips for processing (height, width)
                nms_iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0)
                mask_threshold: Threshold for mask binarization (0.0-1.0)
                small_building_area: Minimum area in pixels to keep a building
                simplify_tolerance: Tolerance for polygon simplification

        Returns:
            GeoDataFrame with building footprints
        """
        # Get parameters from kwargs or use instance defaults
        confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        overlap = kwargs.get("overlap", self.overlap)
        chip_size = kwargs.get("chip_size", self.chip_size)
        nms_iou_threshold = kwargs.get("nms_iou_threshold", self.nms_iou_threshold)
        mask_threshold = kwargs.get("mask_threshold", self.mask_threshold)
        small_building_area = kwargs.get(
            "small_building_area", self.small_building_area
        )
        simplify_tolerance = kwargs.get("simplify_tolerance", self.simplify_tolerance)

        # Print parameters being used
        print(f"Processing with parameters:")
        print(f"- Confidence threshold: {confidence_threshold}")
        print(f"- Tile overlap: {overlap}")
        print(f"- Chip size: {chip_size}")
        print(f"- NMS IoU threshold: {nms_iou_threshold}")
        print(f"- Mask threshold: {mask_threshold}")
        print(f"- Min building area: {small_building_area}")
        print(f"- Simplify tolerance: {simplify_tolerance}")

        # Create dataset
        dataset = BuildingFootprintDataset(raster_path=raster_path, chip_size=chip_size)

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
                    contours = self._mask_to_polygons(
                        binary_mask,
                        simplify_tolerance=simplify_tolerance,
                        mask_threshold=mask_threshold,
                        small_building_area=small_building_area,
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
                "class": 1,  # Building class
            },
            crs=dataset.crs,
        )

        # Remove overlapping polygons with custom threshold
        gdf = self._filter_overlapping_polygons(
            gdf, nms_iou_threshold=nms_iou_threshold
        )

        # Save to file if requested
        if output_path:
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"Saved {len(gdf)} building footprints to {output_path}")

        return gdf

    def visualize_results(
        self, raster_path, gdf=None, output_path=None, figsize=(12, 12)
    ):
        """
        Visualize building detection results.

        Args:
            raster_path: Path to input raster
            gdf: GeoDataFrame with building polygons (optional)
            output_path: Path to save visualization (optional)
            figsize: Figure size (width, height) in inches
        """
        # Check if raster file exists
        if not os.path.exists(raster_path):
            print(f"Error: Raster file '{raster_path}' not found.")
            return

        # Process raster if GeoDataFrame not provided
        if gdf is None:
            gdf = self.process_raster(raster_path)

        if gdf is None or len(gdf) == 0:
            print("No buildings to visualize")
            return

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
            else:
                image = src.read()

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

        # Create figure with appropriate aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]  # width / height
        plt.figure(figsize=(figsize[0], figsize[0] / aspect_ratio))

        # Create axis with the right projection if CRS is available
        ax = plt.gca()

        # Display image
        ax.imshow(image)

        # Convert GeoDataFrame to pixel coordinates for plotting
        with rasterio.open(raster_path) as src:

            def geo_to_pixel(x, y):
                return ~src.transform * (x, y)

            # Plot each building footprint
            for _, row in gdf.iterrows():
                # Convert polygon to pixel coordinates
                geom = row.geometry
                if geom.is_empty:
                    continue

                try:
                    # Get polygon exterior coordinates
                    x, y = geom.exterior.xy

                    # Convert to pixel coordinates
                    pixel_coords = [geo_to_pixel(x[i], y[i]) for i in range(len(x))]
                    pixel_x = [coord[0] for coord in pixel_coords]
                    pixel_y = [coord[1] for coord in pixel_coords]

                    # Plot polygon
                    ax.plot(pixel_x, pixel_y, color="red", linewidth=1)
                except Exception as e:
                    print(f"Error plotting polygon: {e}")

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Building Footprints (Found: {len(gdf)})")

        # Add colorbar for confidence if available
        if "confidence" in gdf.columns:
            # Create a colorbar legend
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap("viridis"),
                norm=plt.Normalize(gdf.confidence.min(), gdf.confidence.max()),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
            cbar.set_label("Confidence")

        # Save if requested
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")

        plt.close()

        # Create a simpler visualization focused just on a subset of buildings
        # This helps when the raster is very large
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Choose a subset of the image to show
        with rasterio.open(raster_path) as src:
            # Get a sample window based on the first few buildings
            if len(gdf) > 0:
                # Get centroid of first building
                sample_geom = gdf.iloc[0].geometry
                centroid = sample_geom.centroid

                # Convert to pixel coordinates
                center_x, center_y = ~src.transform * (centroid.x, centroid.y)

                # Define a window around this building
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

                # Get transform for this window
                window_transform = src.window_transform(window)

                # Display sample image
                ax.imshow(sample_image)

                # Filter buildings that intersect with this window
                window_bounds = rasterio.windows.bounds(window, src.transform)
                window_box = box(*window_bounds)
                visible_gdf = gdf[gdf.intersects(window_box)]

                # Plot building footprints in this view
                for _, row in visible_gdf.iterrows():
                    try:
                        # Get polygon exterior coordinates
                        geom = row.geometry
                        if geom.is_empty:
                            continue

                        x, y = geom.exterior.xy

                        # Convert to pixel coordinates relative to window
                        pixel_coords = [
                            ~window_transform * (x[i], y[i]) for i in range(len(x))
                        ]
                        pixel_x = [coord[0] for coord in pixel_coords]
                        pixel_y = [coord[1] for coord in pixel_coords]

                        # Plot polygon
                        ax.plot(pixel_x, pixel_y, color="red", linewidth=1.5)
                    except Exception as e:
                        print(f"Error plotting polygon in sample view: {e}")

                # Set title
                ax.set_title(
                    f"Sample Area - Building Footprints (Showing: {len(visible_gdf)})"
                )

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

        return True
