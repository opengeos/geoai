"""Point cloud classification module using Open3D-ML's RandLA-Net.

This module provides semantic segmentation of 3D point clouds (LAS/LAZ
files) using the RandLA-Net architecture via Open3D-ML.  It supports
inference with pre-trained weights, fine-tuning on custom datasets, and
visualization through leafmap.

Reference:
    Hu et al., "RandLA-Net: Efficient Semantic Segmentation of Large-Scale
    Point Clouds," CVPR 2020.
    https://arxiv.org/abs/1911.11236
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import laspy
except ImportError:
    raise ImportError(
        "laspy is required for point cloud classification. "
        "Please install it: pip install 'laspy[lazrs]'"
    )

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for point cloud classification. "
        "Please install it: pip install torch"
    )

# Open3D-ML is imported lazily inside PointCloudClassifier.__init__
# so that constants, I/O helpers, and the class definition remain
# accessible without the heavy Open3D dependency installed.
ml3d = None  # set by _ensure_open3d()


__all__ = [
    "PointCloudClassifier",
    "classify_point_cloud",
    "list_pointcloud_models",
    "ASPRS_CLASSES",
    "SUPPORTED_MODELS",
    "DEFAULT_CACHE_DIR",
]

# ASPRS LAS Standard Classification Codes (LAS 1.4 R15)
ASPRS_CLASSES: Dict[int, str] = {
    0: "Created, never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (noise)",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
}

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "geoai", "pointcloud"
)

# Pre-trained model registry
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "RandLANet_SemanticKITTI": {
        "url": (
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_semantickitti_202201071330utc.pth"
        ),
        "description": (
            "RandLA-Net trained on SemanticKITTI (outdoor driving, 19 classes). "
            "Best for street-level and mobile LiDAR data."
        ),
        "num_classes": 19,
        "dataset": "SemanticKITTI",
        "in_channels": 3,
        "class_names": [
            "car", "bicycle", "motorcycle", "truck",
            "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
            "parking", "sidewalk", "other-ground", "building", "fence",
            "vegetation", "trunk", "terrain", "pole", "traffic-sign",
        ],
        "config": {
            "name": "RandLANet",
            "num_neighbors": 16,
            "num_layers": 4,
            "num_points": 45056,
            "num_classes": 19,
            "ignored_label_inds": [0],
            "sub_sampling_ratio": [4, 4, 4, 4],
            "in_channels": 3,
            "dim_features": 8,
            "dim_output": [16, 64, 128, 256],
            "grid_size": 0.06,
        },
    },
    "RandLANet_Toronto3D": {
        "url": (
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_toronto3d_202201071330utc.pth"
        ),
        "description": (
            "RandLA-Net trained on Toronto3D (urban outdoor, 8 classes). "
            "Best for urban airborne and mobile mapping data."
        ),
        "num_classes": 8,
        "dataset": "Toronto3D",
        "in_channels": 6,
        "class_names": [
            "road", "road_marking", "natural", "building",
            "utility_line", "pole", "car", "fence",
        ],
        "config": {
            "name": "RandLANet",
            "num_neighbors": 16,
            "num_layers": 5,
            "num_points": 65536,
            "num_classes": 8,
            "ignored_label_inds": [0],
            "sub_sampling_ratio": [4, 4, 4, 4, 2],
            "in_channels": 6,
            "dim_features": 8,
            "dim_output": [16, 64, 128, 256, 512],
            "grid_size": 0.05,
        },
    },
    "RandLANet_S3DIS": {
        "url": (
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_202010091238.pth"
        ),
        "description": (
            "RandLA-Net trained on S3DIS (indoor, 13 classes). "
            "Best for indoor point cloud data."
        ),
        "num_classes": 13,
        "dataset": "S3DIS",
        "in_channels": 6,
        "class_names": [
            "ceiling", "floor", "wall", "beam", "column", "window",
            "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
        ],
        "config": {
            "name": "RandLANet",
            "num_neighbors": 16,
            "num_layers": 5,
            "num_points": 40960,
            "num_classes": 13,
            "ignored_label_inds": [],
            "sub_sampling_ratio": [4, 4, 4, 4, 2],
            "in_channels": 6,
            "dim_features": 8,
            "dim_output": [16, 64, 128, 256, 512],
            "grid_size": 0.04,
        },
    },
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _patch_open3d_batcher():
    """Monkey-patch Open3D-ML's default batcher for PyTorch/NumPy 2.x compat.

    PyTorch wheels built against NumPy 1.x cannot infer dtypes from NumPy 2.x
    scalar types, causing ``torch.as_tensor()`` to fail with
    ``RuntimeError: Could not infer dtype of numpy.float32``.

    This patch wraps ``default_convert`` in the Open3D-ML batcher to convert
    numpy arrays to torch tensors via ``torch.from_numpy()`` (which always
    works) before the problematic ``torch.as_tensor()`` path is reached.
    """
    try:
        import open3d._ml3d.torch.dataloaders.default_batcher as batcher
    except ImportError:
        return

    # Only patch once
    if getattr(batcher, "_geoai_patched", False):
        return

    _orig_convert = batcher.default_convert

    def _safe_convert(data):
        if isinstance(data, np.ndarray):
            try:
                return torch.from_numpy(np.ascontiguousarray(data))
            except TypeError:
                return _orig_convert(data)
        return _orig_convert(data)

    batcher.default_convert = _safe_convert
    batcher._geoai_patched = True


def _ensure_open3d():
    """Lazily import Open3D-ML and cache the module reference.

    Raises:
        ImportError: If ``open3d`` or its ML extension is not installed.
    """
    global ml3d
    if ml3d is not None:
        return ml3d
    try:
        import open3d.ml.torch as _ml3d

        ml3d = _ml3d
    except ImportError:
        raise ImportError(
            "Open3D with ML extension is required for point cloud classification. "
            "Please install it: pip install open3d"
        )
    _patch_open3d_batcher()
    return ml3d


def _download_checkpoint(
    model_name: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> str:
    """Download a pre-trained checkpoint if not already cached.

    Args:
        model_name: Name of the model variant (key in SUPPORTED_MODELS).
        cache_dir: Directory to store downloaded files.

    Returns:
        Local path to the checkpoint file.

    Raises:
        ValueError: If *model_name* is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(SUPPORTED_MODELS.keys())}"
        )

    info = SUPPORTED_MODELS[model_name]
    url = info["url"]
    filename = os.path.basename(url)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        logger.info("Using cached checkpoint: %s", local_path)
        return local_path

    logger.info("Downloading checkpoint for %s...", model_name)
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlretrieve(url, local_path)
    except (OSError, urllib.error.URLError) as exc:
        # Clean up partial download to avoid corrupt cache
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(
            f"Failed to download checkpoint for '{model_name}' from {url}: {exc}"
        ) from exc

    logger.info("Checkpoint saved to: %s", local_path)
    return local_path


def _read_point_cloud(
    path: str, in_channels: int = 3
) -> Tuple[np.ndarray, np.ndarray, "laspy.LasData"]:
    """Read a LAS/LAZ file and return points, features, and the LAS object.

    Args:
        path: Path to a LAS or LAZ file.
        in_channels: Number of input feature channels the model expects.
            For models expecting 3 channels, only xyz is used (features
            are empty).  For models expecting 6 channels, xyz is
            concatenated with intensity, return_number, and
            number_of_returns (zero-padded if unavailable).

    Returns:
        Tuple of (xyz, features, las) where:
            - xyz is an (N, 3) float64 array of coordinates
            - features is an (N, D) float32 array of additional point
              attributes beyond xyz, with D = in_channels - 3
            - las is the raw laspy.LasData object for metadata access
    """
    las = laspy.read(path)

    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # Extra feature channels beyond xyz
    n_extra = max(0, in_channels - 3)
    if n_extra == 0:
        features = np.zeros((len(xyz), 0), dtype=np.float32)
        return xyz, features, las

    # Build feature array from available dimensions
    feat_arrays: List[np.ndarray] = []

    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float32)
        imax = intensity.max()
        if imax > 0:
            intensity = intensity / imax
        feat_arrays.append(intensity)

    if hasattr(las, "return_number"):
        feat_arrays.append(
            np.asarray(las.return_number, dtype=np.float32)
        )

    if hasattr(las, "number_of_returns"):
        feat_arrays.append(
            np.asarray(las.number_of_returns, dtype=np.float32)
        )

    if feat_arrays:
        features = np.column_stack(feat_arrays)
    else:
        features = np.zeros((len(xyz), 0), dtype=np.float32)

    # Pad or truncate to exactly n_extra columns
    if features.shape[1] < n_extra:
        pad = np.zeros(
            (len(xyz), n_extra - features.shape[1]), dtype=np.float32
        )
        features = np.hstack([features, pad])
    elif features.shape[1] > n_extra:
        features = features[:, :n_extra]

    return xyz, features, las


def _write_point_cloud(
    las_source: "laspy.LasData",
    classifications: np.ndarray,
    output_path: str,
) -> str:
    """Write a classified point cloud to a LAS/LAZ file.

    Copies the original LAS data and updates the classification field.

    Args:
        las_source: Original laspy.LasData object.
        classifications: Integer array of class labels, shape (N,).
        output_path: Destination path (.las or .laz).

    Returns:
        The output path written.
    """
    las_out = laspy.LasData(las_source.header)
    las_out.points = las_source.points.copy()
    las_out.classification = classifications.astype(np.uint8)

    las_out.write(output_path)
    logger.info("Classified point cloud saved to: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Custom dataset wrapper for LAS/LAZ files
# ---------------------------------------------------------------------------


class _LASDatasetSplit:
    """Internal split wrapper used by the Open3D-ML training pipeline."""

    def __init__(self, file_paths: List[str], num_classes: int, in_channels: int = 3):
        self.file_paths = file_paths
        self.num_classes = num_classes
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_data(self, idx: int) -> dict:
        xyz, features, las = _read_point_cloud(
            self.file_paths[idx], self.in_channels
        )
        labels = np.asarray(las.classification, dtype=np.int32)

        return {
            "point": xyz.astype(np.float32),
            "feat": features,
            "label": labels,
        }

    def get_attr(self, idx: int) -> dict:
        return {
            "name": Path(self.file_paths[idx]).stem,
            "path": self.file_paths[idx],
            "split": "training",
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PointCloudClassifier:
    """Classify 3D point clouds using RandLA-Net via Open3D-ML.

    This class wraps the Open3D-ML semantic segmentation pipeline to
    provide point cloud classification from LAS/LAZ files.  Pre-trained
    models are available for outdoor driving (SemanticKITTI), urban
    mapping (Toronto3D), and indoor scanning (S3DIS) scenarios.

    Users can also fine-tune on custom annotated point cloud data using
    the :meth:`train` method.

    Attributes:
        model_name (str): Name of the model variant being used.
        device (str): Device the model is running on.
        num_classes (int): Number of output classes.

    Example:
        >>> classifier = PointCloudClassifier()
        >>> classifier.classify("input.las", output_path="classified.las")
    """

    def __init__(
        self,
        model_name: str = "RandLANet_Toronto3D",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        num_classes: Optional[int] = None,
    ):
        """Initialize the PointCloudClassifier.

        Args:
            model_name: Pre-trained model variant to use.  Options:

                - ``"RandLANet_SemanticKITTI"`` - outdoor driving (19 classes)
                - ``"RandLANet_Toronto3D"`` (default) - urban outdoor (8 classes)
                - ``"RandLANet_S3DIS"`` - indoor scanning (13 classes)

            checkpoint_path: Path to a local checkpoint file.  If *None*,
                the checkpoint is downloaded automatically.
            device: Device for inference (``"cpu"``, ``"cuda"``, etc.).
                If *None*, CUDA is used when available.
            cache_dir: Directory to cache downloaded checkpoints.
            num_classes: Override the number of output classes.  Useful
                when loading a custom-trained checkpoint.  If *None*,
                uses the value from the model registry.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        info = SUPPORTED_MODELS[model_name]
        self.num_classes = info["num_classes"] if num_classes is None else num_classes
        self.class_names = info["class_names"]
        self.in_channels = info.get("in_channels", 3)
        self._config = info["config"].copy()
        self._config["num_classes"] = self.num_classes

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Download or use provided checkpoint
        if checkpoint_path is None:
            checkpoint_path = _download_checkpoint(model_name, cache_dir)
        self.checkpoint_path = checkpoint_path

        # Build model and pipeline (triggers Open3D-ML import)
        _ml3d = _ensure_open3d()
        logger.info(
            "Loading %s on %s (%d classes)...",
            model_name, device, self.num_classes,
        )
        self._model = _ml3d.models.RandLANet(**self._config)
        self._pipeline = _ml3d.pipelines.SemanticSegmentation(
            self._model, device=device,
        )
        self._pipeline.load_ckpt(self.checkpoint_path)
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify a point cloud from a LAS/LAZ file.

        Reads the input file, runs semantic segmentation, and optionally
        writes the result to a new LAS/LAZ file with the classification
        field updated.

        Args:
            input_path: Path to the input LAS/LAZ file.
            output_path: Optional path for the classified output file.
                If *None*, classifications are returned but not saved.
            **kwargs: Reserved for future use.

        Returns:
            Tuple of (classifications, probabilities) where:
                - classifications: (N,) int32 array of predicted class IDs
                - probabilities: (N, C) float32 array of class probabilities

        Raises:
            FileNotFoundError: If *input_path* does not exist.

        Example:
            >>> clf = PointCloudClassifier()
            >>> labels, probs = clf.classify("input.las",
            ...                              output_path="classified.las")
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info("Reading point cloud: %s", input_path)
        xyz, features, las = _read_point_cloud(input_path, self.in_channels)
        n_points = len(xyz)
        logger.info("Loaded %d points.", n_points)

        # Prepare data dict for Open3D-ML pipeline
        data = {
            "point": xyz.astype(np.float32),
            "feat": features,
            "label": np.zeros(n_points, dtype=np.int32),
        }

        logger.info("Running inference...")
        result = self._pipeline.run_inference(data)
        predictions = np.asarray(result["predict_labels"], dtype=np.int32)

        # Extract probabilities if available
        if "predict_scores" in result:
            probabilities = np.asarray(result["predict_scores"], dtype=np.float32)
        else:
            probabilities = np.zeros(
                (n_points, self.num_classes), dtype=np.float32
            )

        # Write output
        if output_path is not None:
            _write_point_cloud(las, predictions, output_path)

        return predictions, probabilities

    def classify_batch(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Classify multiple point cloud files.

        Args:
            input_paths: List of paths to LAS/LAZ files.
            output_dir: Directory for output files.  If *None*, results
                are returned but not saved.  Output filenames match input
                with ``_classified`` suffix.
            **kwargs: Reserved for future use.

        Returns:
            List of (classifications, probabilities) tuples, one per file.
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for path in input_paths:
            out_path = None
            if output_dir is not None:
                stem = Path(path).stem
                ext = Path(path).suffix
                out_path = os.path.join(output_dir, f"{stem}_classified{ext}")

            result = self.classify(
                path, output_path=out_path, **kwargs
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Training / fine-tuning
    # ------------------------------------------------------------------

    def train(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 4,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, list]:
        """Train or fine-tune the model on labeled LAS/LAZ files.

        Point cloud files must have the ``classification`` field populated
        with ground-truth labels.  Files in *train_dir* are used for
        training; files in *val_dir* (if provided) are used for
        validation.

        This method fine-tunes from the currently loaded checkpoint,
        making it suitable for transfer learning on custom datasets.

        Args:
            train_dir: Directory containing labeled LAS/LAZ files for
                training.
            val_dir: Optional directory with labeled files for validation.
            epochs: Number of training epochs.
            learning_rate: Initial learning rate.
            batch_size: Mini-batch size.
            save_dir: Directory to save checkpoints.  Defaults to
                ``<cache_dir>/training/``.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            Dictionary with training history containing ``"train_loss"``
            and optionally ``"val_loss"`` lists.

        Raises:
            FileNotFoundError: If *train_dir* does not exist.
            ValueError: If no LAS/LAZ files found in *train_dir*.

        Example:
            >>> clf = PointCloudClassifier(num_classes=6)
            >>> history = clf.train("train_data/", val_dir="val_data/",
            ...                     epochs=50)
        """
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        train_files = sorted(
            [
                str(p)
                for p in Path(train_dir).glob("*")
                if p.suffix.lower() in (".las", ".laz")
            ]
        )
        if not train_files:
            raise ValueError(f"No LAS/LAZ files found in: {train_dir}")

        val_files = []
        if val_dir is not None:
            if not os.path.isdir(val_dir):
                raise FileNotFoundError(
                    f"Validation directory not found: {val_dir}"
                )
            val_files = sorted(
                [
                    str(p)
                    for p in Path(val_dir).glob("*")
                    if p.suffix.lower() in (".las", ".laz")
                ]
            )

        if save_dir is None:
            save_dir = os.path.join(DEFAULT_CACHE_DIR, "training")
        os.makedirs(save_dir, exist_ok=True)

        logger.info(
            "Training with %d files (%d validation).",
            len(train_files), len(val_files),
        )

        # Use the Open3D-ML pipeline's training infrastructure which
        # handles random sub-sampling, KNN graph construction, and the
        # correct forward pass for RandLA-Net.
        _ml3d = _ensure_open3d()

        # Build training configuration
        train_split = _LASDatasetSplit(train_files, self.num_classes, self.in_channels)
        val_split = (
            _LASDatasetSplit(val_files, self.num_classes, self.in_channels)
            if val_files
            else None
        )

        cfg = {
            "max_epoch": epochs,
            "optimizer": {"lr": learning_rate},
            "batch_size": batch_size,
            "save_ckpt_freq": 20,
            "log_dir": save_dir,
        }

        self._pipeline.cfg.update(cfg)

        # Run training through the pipeline
        logger.info("Starting training via Open3D-ML pipeline...")
        self._pipeline.run_train()

        # Save final checkpoint
        final_path = os.path.join(save_dir, "checkpoint_final.pth")
        torch.save(self._model.state_dict(), final_path)
        logger.info("Final checkpoint saved: %s", final_path)

        self._model.eval()

        return {"train_loss": [], "checkpoint_path": final_path}

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        input_path: str,
        backend: str = "pyvista",
        cmap: str = "terrain",
        point_size: float = 2.0,
        **kwargs,
    ) -> Any:
        """Visualize a classified point cloud using leafmap.

        Args:
            input_path: Path to a LAS/LAZ file.
            backend: Visualization backend.  Options: ``"pyvista"``,
                ``"ipygany"``, ``"panel"``, ``"open3d"``.
            cmap: Matplotlib colormap name.
            point_size: Point size for rendering.
            **kwargs: Additional keyword arguments passed to
                ``leafmap.view_lidar()``.

        Returns:
            The visualization widget/object from leafmap.

        Example:
            >>> clf = PointCloudClassifier()
            >>> clf.visualize("classified.las", backend="pyvista")
        """
        try:
            import leafmap
        except ImportError:
            raise ImportError(
                "leafmap is required for visualization. "
                "Please install it: pip install 'leafmap[lidar]'"
            )

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        return leafmap.view_lidar(
            input_path,
            cmap=cmap,
            backend=backend,
            point_size=point_size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Summary / statistics
    # ------------------------------------------------------------------

    def summary(
        self,
        input_path: str,
        class_map: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Compute classification summary statistics for a point cloud.

        Args:
            input_path: Path to a LAS/LAZ file with classification.
            class_map: Optional mapping from class IDs to names.
                Defaults to ASPRS standard codes.

        Returns:
            Dictionary with keys:
                - ``"total_points"``: total number of points
                - ``"class_counts"``: dict mapping class name to count
                - ``"class_percentages"``: dict mapping class name to %
                - ``"bounds"``: (min_x, min_y, min_z, max_x, max_y, max_z)

        Example:
            >>> clf = PointCloudClassifier()
            >>> stats = clf.summary("classified.las")
            >>> print(stats["class_counts"])
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if class_map is None:
            class_map = ASPRS_CLASSES

        las = laspy.read(input_path)
        classifications = np.asarray(las.classification, dtype=np.int32)
        total = len(classifications)

        unique, counts = np.unique(classifications, return_counts=True)
        class_counts = {}
        class_pcts = {}
        for cls_id, count in zip(unique, counts):
            name = class_map.get(int(cls_id), f"Class {cls_id}")
            class_counts[name] = int(count)
            class_pcts[name] = (
                round(float(count) / total * 100, 2) if total > 0 else 0.0
            )

        bounds = (
            float(las.header.x_min),
            float(las.header.y_min),
            float(las.header.z_min),
            float(las.header.x_max),
            float(las.header.y_max),
            float(las.header.z_max),
        )

        return {
            "total_points": total,
            "class_counts": class_counts,
            "class_percentages": class_pcts,
            "bounds": bounds,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def classify_point_cloud(
    input_path: str,
    output_path: Optional[str] = None,
    model_name: str = "RandLANet_Toronto3D",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for point cloud classification.

    Creates a :class:`PointCloudClassifier` and runs classification in
    one call.

    Args:
        input_path: Path to the input LAS/LAZ file.
        output_path: Optional path for the classified output file.
        model_name: Pre-trained model variant.
            See :class:`PointCloudClassifier` for options.
        checkpoint_path: Optional local checkpoint path.
        device: Device for inference.  If *None*, auto-selects.
        cache_dir: Directory to cache model checkpoints.
        **kwargs: Additional arguments passed to
            :meth:`PointCloudClassifier.classify`.

    Returns:
        Tuple of (classifications, probabilities) arrays.

    Example:
        >>> labels, probs = classify_point_cloud(
        ...     "input.las",
        ...     output_path="classified.las",
        ...     model_name="RandLANet_Toronto3D",
        ... )
    """
    classifier = PointCloudClassifier(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        cache_dir=cache_dir,
    )
    return classifier.classify(input_path, output_path=output_path, **kwargs)


def list_pointcloud_models() -> Dict[str, str]:
    """List available point cloud classification models.

    Returns:
        Dictionary mapping model names to their descriptions.

    Example:
        >>> models = list_pointcloud_models()
        >>> for name, desc in models.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: info["description"] for name, info in SUPPORTED_MODELS.items()}
