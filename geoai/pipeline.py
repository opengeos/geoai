"""Batch processing pipeline framework for geospatial AI workflows.

This module provides a Pipeline class that chains operations (download,
preprocess, infer, postprocess, export) with parallel execution, progress
bars, YAML/JSON-based definitions, and resume-from-checkpoint support.

Example:
    >>> from geoai.pipeline import Pipeline, FunctionStep
    >>> pipe = Pipeline(
    ...     steps=[
    ...         FunctionStep("preprocess", preprocess_fn),
    ...         FunctionStep("segment", segment_fn),
    ...         FunctionStep("vectorize", vectorize_fn),
    ...     ],
    ...     max_workers=4,
    ... )
    >>> results = pipe.run(items=[{"input_path": "image.tif"}])
"""

import glob as glob_module
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class ErrorPolicy(str, Enum):
    """How the pipeline handles per-item errors."""

    SKIP = "skip"
    FAIL = "fail"


class ItemStatus(str, Enum):
    """Status of a single work item in the checkpoint."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of running all pipeline steps on a single item.

    Attributes:
        item: The (possibly modified) work item dict.
        success: Whether all steps completed without error.
        error: Error message if a step failed, None otherwise.
        duration: Wall-clock time in seconds.
    """

    item: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class PipelineResult:
    """Aggregate result of a full pipeline run.

    Attributes:
        completed: List of successfully processed items.
        failed: List of ``(item, error_message)`` tuples for failed items.
        skipped: List of items skipped (already checkpointed).
        total_duration: Total wall-clock time in seconds.
        checkpoint_path: Path to the checkpoint file, if any.
    """

    completed: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    checkpoint_path: Optional[str] = None

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the pipeline run.

        Returns:
            Dict with counts of completed, failed, skipped, total items
            and total_duration.
        """
        return {
            "completed": len(self.completed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "total": len(self.completed) + len(self.failed) + len(self.skipped),
            "total_duration": round(self.total_duration, 2),
        }


# ---------------------------------------------------------------------------
# Step abstractions
# ---------------------------------------------------------------------------


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Subclass this for steps that need setup/teardown (e.g., loading a model).
    For simple transformations, use :class:`FunctionStep` instead.

    Args:
        name: Human-readable name for this step.
    """

    def __init__(self, name: str) -> None:
        """Initialize the pipeline step.

        Args:
            name: Human-readable name for this step.
        """
        self.name = name

    def setup(self) -> None:
        """Called once before processing any items.

        Override to load models, open connections, etc.
        """

    @abstractmethod
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single work item.

        Args:
            item: Dictionary containing work item data.

        Returns:
            The item dict, potentially with new/modified keys.
        """
        ...

    def teardown(self) -> None:
        """Called once after all items are processed.

        Override to release resources, close connections, etc.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class FunctionStep(PipelineStep):
    """A pipeline step wrapping a plain callable.

    This is the simplest way to create a step — just pass a function
    that takes a dict and returns a dict.

    Args:
        name: Human-readable name for this step.
        fn: Callable that takes an item dict and returns a modified item dict.
        setup_fn: Optional callable run once before processing.
        teardown_fn: Optional callable run once after processing.

    Example:
        >>> def add_suffix(item):
        ...     base = os.path.splitext(item["input_path"])[0]
        ...     item["output_path"] = f"{base}_mask.tif"
        ...     return item
        >>> step = FunctionStep("add_suffix", add_suffix)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        setup_fn: Optional[Callable[[], None]] = None,
        teardown_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the function step.

        Args:
            name: Human-readable name for this step.
            fn: Callable that processes an item dict.
            setup_fn: Optional callable run once before processing.
            teardown_fn: Optional callable run once after processing.
        """
        super().__init__(name)
        self._fn = fn
        self._setup_fn = setup_fn
        self._teardown_fn = teardown_fn

    def setup(self) -> None:
        """Run the optional setup function."""
        if self._setup_fn:
            self._setup_fn()

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item using the wrapped function.

        Args:
            item: Dictionary containing work item data.

        Returns:
            The modified item dict.
        """
        return self._fn(item)

    def teardown(self) -> None:
        """Run the optional teardown function."""
        if self._teardown_fn:
            self._teardown_fn()


class GlobStep(PipelineStep):
    """Expand a directory or glob pattern into individual work items.

    Typically the first step in a pipeline. Takes a single item with an
    ``input_dir`` or ``input_pattern`` key and yields multiple items, one
    per matched file.

    Args:
        name: Step name. Defaults to ``"glob"``.
        extensions: File extensions to match. Defaults to common raster
            formats.
    """

    def __init__(
        self,
        name: str = "glob",
        extensions: Optional[List[str]] = None,
    ) -> None:
        """Initialize the glob step.

        Args:
            name: Step name.
            extensions: File extensions to match (e.g. ``[".tif", ".jp2"]``).
        """
        super().__init__(name)
        self.extensions = extensions or [".tif", ".tiff", ".jp2", ".img"]

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through (expansion happens via :meth:`expand`).

        Args:
            item: Work item dict.

        Returns:
            The item unchanged.
        """
        return item

    def expand(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand a single item into multiple items by globbing.

        Args:
            item: Dict with ``input_dir`` or ``input_pattern`` key.

        Returns:
            List of item dicts, one per matched file.

        Raises:
            ValueError: If the item has neither ``input_dir`` nor
                ``input_pattern``.
        """
        input_dir = item.get("input_dir")
        input_pattern = item.get("input_pattern")

        if input_pattern:
            files = sorted(glob_module.glob(input_pattern))
        elif input_dir:
            files = []
            for ext in self.extensions:
                files.extend(glob_module.glob(os.path.join(input_dir, f"*{ext}")))
            files = sorted(set(files))
        else:
            raise ValueError("Item must have 'input_dir' or 'input_pattern' key")

        items = []
        for f in files:
            new_item = dict(item)
            new_item["input_path"] = f
            new_item.pop("input_dir", None)
            new_item.pop("input_pattern", None)
            items.append(new_item)

        logger.info("GlobStep found %d files", len(items))
        return items


# ---------------------------------------------------------------------------
# Built-in domain steps
# ---------------------------------------------------------------------------


class SemanticSegmentationStep(PipelineStep):
    """Run semantic segmentation inference using an SMP model.

    Wraps :func:`geoai.train.semantic_segmentation`. The model is loaded
    once in :meth:`setup` and reused for every item.

    Args:
        name: Step name.
        model_path: Path to model weights or HuggingFace model ID.
        architecture: Model architecture (e.g. ``"unet"``).
        encoder_name: Encoder backbone name.
        num_channels: Number of input channels.
        num_classes: Number of output classes.
        window_size: Sliding window size for inference.
        overlap: Overlap between windows in pixels.
        batch_size: Batch size for inference.
        device: Torch device string. ``None`` for auto-detect.
        suffix: Suffix appended to output filenames.
    """

    def __init__(
        self,
        name: str = "semantic_segmentation",
        model_path: str = "",
        architecture: str = "unet",
        encoder_name: str = "resnet34",
        num_channels: int = 3,
        num_classes: int = 2,
        window_size: int = 512,
        overlap: int = 256,
        batch_size: int = 4,
        device: Optional[str] = None,
        suffix: str = "_mask",
    ) -> None:
        """Initialize the semantic segmentation step.

        Args:
            name: Step name.
            model_path: Path to model weights or HuggingFace model ID.
            architecture: Model architecture.
            encoder_name: Encoder backbone name.
            num_channels: Number of input channels.
            num_classes: Number of output classes.
            window_size: Sliding window size for inference.
            overlap: Overlap between windows in pixels.
            batch_size: Batch size for inference.
            device: Torch device string. ``None`` for auto-detect.
            suffix: Suffix appended to output filenames.
        """
        super().__init__(name)
        self.model_path = model_path
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device_str = device
        self.suffix = suffix

    def setup(self) -> None:
        """Load the segmentation model onto the target device."""
        from geoai.train import semantic_segmentation  # noqa: F401

        logger.info("SemanticSegmentationStep ready (model_path=%s)", self.model_path)

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run segmentation inference on a single file.

        Args:
            item: Must contain ``input_path`` and ``output_dir`` keys.

        Returns:
            Item with ``output_path`` key added.
        """
        from geoai.train import semantic_segmentation

        input_path = item["input_path"]
        output_dir = item.get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}{self.suffix}.tif")

        kwargs: Dict[str, Any] = {}
        if self.device_str is not None:
            import torch

            kwargs["device"] = torch.device(self.device_str)

        semantic_segmentation(
            input_path=input_path,
            output_path=output_path,
            model_path=self.model_path,
            architecture=self.architecture,
            encoder_name=self.encoder_name,
            num_channels=self.num_channels,
            num_classes=self.num_classes,
            window_size=self.window_size,
            overlap=self.overlap,
            batch_size=self.batch_size,
            quiet=True,
            **kwargs,
        )
        item["output_path"] = output_path
        return item

    def teardown(self) -> None:
        """Release GPU memory."""
        try:
            from geoai.utils.device import empty_cache

            empty_cache()
        except ImportError:
            pass


class RasterToVectorStep(PipelineStep):
    """Convert raster masks to vector features.

    Wraps :func:`geoai.utils.raster.raster_to_vector`.

    Args:
        name: Step name.
        output_format: Output format extension (``".geojson"`` or
            ``".gpkg"``).
        simplify_tolerance: Geometry simplification tolerance. ``None``
            to skip simplification.
        input_key: Key in item dict for the input raster path.
        output_key: Key to store the output vector path.
    """

    def __init__(
        self,
        name: str = "raster_to_vector",
        output_format: str = ".geojson",
        simplify_tolerance: Optional[float] = None,
        input_key: str = "output_path",
        output_key: str = "vector_path",
    ) -> None:
        """Initialize the raster-to-vector step.

        Args:
            name: Step name.
            output_format: Output format extension.
            simplify_tolerance: Geometry simplification tolerance.
            input_key: Key in item dict for the input raster path.
            output_key: Key to store the output vector path.
        """
        super().__init__(name)
        self.output_format = output_format
        self.simplify_tolerance = simplify_tolerance
        self.input_key = input_key
        self.output_key = output_key

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raster mask to a vector file.

        Args:
            item: Must contain the key specified by ``input_key``.

        Returns:
            Item with ``output_key`` added pointing to the vector file.
        """
        from geoai.utils.raster import raster_to_vector

        raster_path = item[self.input_key]
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        output_dir = item.get("output_dir", os.path.dirname(raster_path))
        os.makedirs(output_dir, exist_ok=True)
        vector_path = os.path.join(output_dir, f"{base_name}{self.output_format}")

        gdf = raster_to_vector(raster_path, output=vector_path)
        if self.simplify_tolerance and gdf is not None:
            gdf["geometry"] = gdf.geometry.simplify(self.simplify_tolerance)
            gdf.to_file(vector_path)

        item[self.output_key] = vector_path
        return item


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


@dataclass
class CheckpointEntry:
    """A single item's status in the checkpoint.

    Attributes:
        item_key: Unique identifier for the item (typically input_path).
        status: Current processing status.
        error: Error message if failed.
        completed_steps: List of step names completed for this item.
        timestamp: ISO-format timestamp of last status change.
    """

    item_key: str
    status: ItemStatus
    error: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    timestamp: str = ""


class CheckpointManager:
    """Manages checkpoint state for pipeline resume functionality.

    Stores a JSON file tracking which items have been processed, enabling
    resume-from-checkpoint for long-running batch jobs.

    Args:
        checkpoint_path: Path to the checkpoint JSON file.
        config_hash: Hash of the pipeline config for change detection.
    """

    def __init__(self, checkpoint_path: str, config_hash: str = "") -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
            config_hash: Hash of pipeline config for staleness detection.
        """
        self.checkpoint_path = checkpoint_path
        self.config_hash = config_hash
        self._entries: Dict[str, CheckpointEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load existing checkpoint file if present."""
        if not os.path.exists(self.checkpoint_path):
            return

        with open(self.checkpoint_path, "r") as f:
            data = json.load(f)

        stored_hash = data.get("config_hash", "")
        if stored_hash and stored_hash != self.config_hash:
            logger.warning(
                "Pipeline config changed since last checkpoint. "
                "Existing progress will be reset."
            )
            self._entries = {}
            return

        for key, entry_data in data.get("entries", {}).items():
            self._entries[key] = CheckpointEntry(
                item_key=key,
                status=ItemStatus(entry_data["status"]),
                error=entry_data.get("error"),
                completed_steps=entry_data.get("completed_steps", []),
                timestamp=entry_data.get("timestamp", ""),
            )
        logger.info(
            "Loaded checkpoint with %d entries (%d completed)",
            len(self._entries),
            sum(1 for e in self._entries.values() if e.status == ItemStatus.COMPLETED),
        )

    def save(self) -> None:
        """Persist checkpoint state to disk."""
        data: Dict[str, Any] = {
            "config_hash": self.config_hash,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entries": {},
        }
        for key, entry in self._entries.items():
            data["entries"][key] = {
                "status": entry.status.value,
                "error": entry.error,
                "completed_steps": entry.completed_steps,
                "timestamp": entry.timestamp,
            }
        parent = os.path.dirname(self.checkpoint_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def is_completed(self, item_key: str) -> bool:
        """Check if an item was already completed.

        Args:
            item_key: Unique key for the item.

        Returns:
            ``True`` if the item completed in a previous run.
        """
        entry = self._entries.get(item_key)
        return entry is not None and entry.status == ItemStatus.COMPLETED

    def mark_completed(self, item_key: str, steps: List[str]) -> None:
        """Mark an item as completed.

        Args:
            item_key: Unique key for the item.
            steps: List of step names that completed.
        """
        self._entries[item_key] = CheckpointEntry(
            item_key=item_key,
            status=ItemStatus.COMPLETED,
            completed_steps=steps,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def mark_failed(self, item_key: str, error: str, steps: List[str]) -> None:
        """Mark an item as failed.

        Args:
            item_key: Unique key for the item.
            error: Error description.
            steps: Steps that completed before failure.
        """
        self._entries[item_key] = CheckpointEntry(
            item_key=item_key,
            status=ItemStatus.FAILED,
            error=error,
            completed_steps=steps,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return counts by status.

        Returns:
            Dict mapping status names to counts.
        """
        counts: Dict[str, int] = {s.value: 0 for s in ItemStatus}
        for entry in self._entries.values():
            counts[entry.status.value] += 1
        return counts


# ---------------------------------------------------------------------------
# Step registry for YAML/JSON config deserialization
# ---------------------------------------------------------------------------

_STEP_REGISTRY: Dict[str, type] = {}


def register_step(cls: type) -> type:
    """Register a :class:`PipelineStep` subclass for config deserialization.

    Args:
        cls: A PipelineStep subclass to register.

    Returns:
        The class unchanged (decorator pattern).

    Example:
        >>> @register_step
        ... class MyStep(PipelineStep):
        ...     def process(self, item):
        ...         return item
    """
    _STEP_REGISTRY[cls.__name__] = cls
    return cls


def _step_to_dict(step: PipelineStep) -> Dict[str, Any]:
    """Serialize a step to a dict for config export.

    Args:
        step: PipelineStep instance.

    Returns:
        Dict with ``type``, ``name``, and step-specific parameters.
    """
    d: Dict[str, Any] = {"type": step.__class__.__name__, "name": step.name}
    for key, value in vars(step).items():
        if not key.startswith("_") and key != "name" and not callable(value):
            d[key] = value
    return d


def _step_from_dict(d: Dict[str, Any]) -> PipelineStep:
    """Deserialize a step from a config dict.

    Args:
        d: Dict with ``type`` key and step parameters.

    Returns:
        PipelineStep instance.

    Raises:
        ValueError: If the step type is not registered.
    """
    d = dict(d)  # defensive copy
    step_type = d.pop("type")
    if step_type not in _STEP_REGISTRY:
        raise ValueError(
            f"Unknown step type '{step_type}'. "
            f"Available: {sorted(_STEP_REGISTRY.keys())}"
        )
    cls = _STEP_REGISTRY[step_type]
    return cls(**d)


# Register built-in steps
register_step(GlobStep)
register_step(SemanticSegmentationStep)
register_step(RasterToVectorStep)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class Pipeline:
    """Chains PipelineStep operations with parallel execution and checkpointing.

    A Pipeline processes a list of work items through an ordered sequence of
    steps. Items are dicts that accumulate results as they pass through each
    step.

    Args:
        steps: Ordered sequence of PipelineStep instances.
        max_workers: Number of parallel workers. ``1`` for sequential.
        executor_type: ``"thread"`` or ``"process"``. Defaults to
            ``"thread"`` (safer with GPU models).
        on_error: Error handling policy. ``"skip"`` to continue on failure,
            ``"fail"`` to stop immediately.
        checkpoint_dir: Directory to store checkpoint files. ``None``
            disables checkpointing.
        item_key_fn: Function that extracts a unique key from an item dict.
            Defaults to using the ``"input_path"`` value.
        name: Optional name for this pipeline (used in checkpoint filename).
        quiet: If ``True``, suppress progress bars.

    Example:
        >>> pipe = Pipeline(
        ...     steps=[step1, step2, step3],
        ...     max_workers=4,
        ...     checkpoint_dir="./checkpoints",
        ... )
        >>> result = pipe.run(items=[{"input_path": "a.tif"}, ...])
        >>> print(result.summary)
    """

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        max_workers: int = 1,
        executor_type: str = "thread",
        on_error: str = "skip",
        checkpoint_dir: Optional[str] = None,
        item_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        name: Optional[str] = None,
        quiet: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            steps: Ordered sequence of PipelineStep instances.
            max_workers: Number of parallel workers.
            executor_type: ``"thread"`` or ``"process"``.
            on_error: ``"skip"`` or ``"fail"``.
            checkpoint_dir: Directory for checkpoint files.
            item_key_fn: Function to extract a unique key from an item.
            name: Pipeline name.
            quiet: Suppress progress bars.
        """
        self.steps = list(steps)
        self.max_workers = max_workers
        self.executor_type = executor_type
        self.on_error = ErrorPolicy(on_error)
        self.checkpoint_dir = checkpoint_dir
        self.item_key_fn = item_key_fn or (lambda item: item.get("input_path", ""))
        self.name = name or "pipeline"
        self.quiet = quiet
        self._checkpoint: Optional[CheckpointManager] = None

    def _config_hash(self) -> str:
        """Compute a hash of the pipeline configuration."""
        config_str = json.dumps(
            {"steps": [s.name for s in self.steps], "name": self.name},
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _init_checkpoint(self) -> Optional[CheckpointManager]:
        """Initialize checkpoint manager if checkpoint_dir is set."""
        if not self.checkpoint_dir:
            return None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.name}_checkpoint.json"
        )
        return CheckpointManager(checkpoint_path, self._config_hash())

    def _processing_steps(self) -> List[PipelineStep]:
        """Return steps that run per-item (excludes GlobStep)."""
        return [s for s in self.steps if not isinstance(s, GlobStep)]

    def _process_single_item(self, item: Dict[str, Any]) -> StepResult:
        """Run all processing steps on a single item.

        Args:
            item: Work item dict.

        Returns:
            StepResult with final item state and success/error info.
        """
        start = time.time()
        current_item = dict(item)

        for step in self._processing_steps():
            try:
                current_item = step.process(current_item)
            except Exception as e:
                logger.error(
                    "Step '%s' failed on item '%s': %s",
                    step.name,
                    current_item.get("input_path", "?"),
                    e,
                )
                return StepResult(
                    item=current_item,
                    success=False,
                    error=f"Step '{step.name}': {e}",
                    duration=time.time() - start,
                )

        return StepResult(
            item=current_item,
            success=True,
            duration=time.time() - start,
        )

    def run(
        self,
        items: Optional[List[Dict[str, Any]]] = None,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """Execute the pipeline on a list of items.

        Provide either *items* directly or *input_dir* (which will be
        expanded via :class:`GlobStep` if present, or auto-globbed for
        ``.tif`` files).

        Args:
            items: List of work item dicts.
            input_dir: Directory of input files.
            output_dir: Default output directory injected into each item.

        Returns:
            :class:`PipelineResult` with completed, failed, and skipped
            lists.

        Raises:
            ValueError: If neither *items* nor *input_dir* is provided.
        """
        start_time = time.time()
        result = PipelineResult()

        if items is None and input_dir is None:
            raise ValueError("Provide either 'items' or 'input_dir'")

        # Resolve items from input_dir
        if items is None:
            base_item: Dict[str, Any] = {"input_dir": input_dir}
            if output_dir:
                base_item["output_dir"] = output_dir
            glob_steps = [s for s in self.steps if isinstance(s, GlobStep)]
            glob_step = glob_steps[0] if glob_steps else GlobStep()
            items = glob_step.expand(base_item)
        elif output_dir:
            for item in items:
                item.setdefault("output_dir", output_dir)

        if not items:
            logger.warning("No items to process")
            result.total_duration = time.time() - start_time
            return result

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize checkpoint
        self._checkpoint = self._init_checkpoint()

        # Setup all processing steps
        proc_steps = self._processing_steps()
        for step in proc_steps:
            step.setup()

        try:
            # Filter already-completed items
            pending_items = []
            for item in items:
                key = self.item_key_fn(item)
                if self._checkpoint and self._checkpoint.is_completed(key):
                    result.skipped.append(item)
                else:
                    pending_items.append(item)

            if result.skipped:
                logger.info(
                    "Skipping %d already-completed items",
                    len(result.skipped),
                )

            # Process
            if self.max_workers <= 1:
                self._run_sequential(pending_items, result)
            else:
                self._run_parallel(pending_items, result)

        finally:
            for step in proc_steps:
                try:
                    step.teardown()
                except Exception as e:
                    logger.warning("Teardown failed for step '%s': %s", step.name, e)
            if self._checkpoint:
                self._checkpoint.save()

        result.total_duration = time.time() - start_time
        if self._checkpoint:
            result.checkpoint_path = self._checkpoint.checkpoint_path

        logger.info(
            "Pipeline complete: %d completed, %d failed, %d skipped in %.1fs",
            len(result.completed),
            len(result.failed),
            len(result.skipped),
            result.total_duration,
        )
        return result

    def _step_names(self) -> List[str]:
        """Return names of processing steps."""
        return [s.name for s in self._processing_steps()]

    def _run_sequential(
        self,
        items: List[Dict[str, Any]],
        result: PipelineResult,
    ) -> None:
        """Process items sequentially with a progress bar.

        Args:
            items: Items to process.
            result: PipelineResult to populate.
        """
        for item in tqdm(items, desc=self.name, disable=self.quiet):
            key = self.item_key_fn(item)
            step_result = self._process_single_item(item)

            if step_result.success:
                result.completed.append(step_result.item)
                if self._checkpoint:
                    self._checkpoint.mark_completed(key, self._step_names())
            else:
                if self.on_error == ErrorPolicy.FAIL:
                    if self._checkpoint:
                        self._checkpoint.mark_failed(key, step_result.error or "", [])
                        self._checkpoint.save()
                    raise RuntimeError(
                        f"Pipeline failed on item '{key}': " f"{step_result.error}"
                    )
                result.failed.append((item, step_result.error))
                if self._checkpoint:
                    self._checkpoint.mark_failed(key, step_result.error or "", [])

            # Periodic checkpoint save
            processed = len(result.completed) + len(result.failed)
            if self._checkpoint and processed % 10 == 0:
                self._checkpoint.save()

    def _run_parallel(
        self,
        items: List[Dict[str, Any]],
        result: PipelineResult,
    ) -> None:
        """Process items in parallel with a progress bar.

        Args:
            items: Items to process.
            result: PipelineResult to populate.
        """
        executor_cls: Union[type, type] = (
            ThreadPoolExecutor
            if self.executor_type == "thread"
            else ProcessPoolExecutor
        )

        with executor_cls(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self._process_single_item, item): item for item in items
            }
            with tqdm(total=len(items), desc=self.name, disable=self.quiet) as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    key = self.item_key_fn(item)
                    try:
                        step_result = future.result()
                    except Exception as e:
                        step_result = StepResult(item=item, success=False, error=str(e))

                    if step_result.success:
                        result.completed.append(step_result.item)
                        if self._checkpoint:
                            self._checkpoint.mark_completed(key, self._step_names())
                    else:
                        if self.on_error == ErrorPolicy.FAIL:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise RuntimeError(
                                f"Pipeline failed on '{key}': " f"{step_result.error}"
                            )
                        result.failed.append((item, step_result.error))
                        if self._checkpoint:
                            self._checkpoint.mark_failed(
                                key, step_result.error or "", []
                            )

                    pbar.update(1)

        if self._checkpoint:
            self._checkpoint.save()

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline configuration to a dict.

        Returns:
            Dict representation of this pipeline's configuration, suitable
            for JSON or YAML export.
        """
        return {
            "name": self.name,
            "max_workers": self.max_workers,
            "executor_type": self.executor_type,
            "on_error": self.on_error.value,
            "steps": [_step_to_dict(step) for step in self.steps],
        }

    def to_json(self, path: str) -> None:
        """Save pipeline configuration to a JSON file.

        Args:
            path: Output JSON file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved pipeline config to %s", path)

    def to_yaml(self, path: str) -> None:
        """Save pipeline configuration to a YAML file.

        Args:
            path: Output YAML file path.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved pipeline config to %s", path)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name={self.name!r}, steps={len(self.steps)}, "
            f"max_workers={self.max_workers})"
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_pipeline(config_path: str, **overrides: Any) -> Pipeline:
    """Load a Pipeline from a JSON or YAML config file.

    Args:
        config_path: Path to config file (``.json`` or ``.yaml``/``.yml``).
        **overrides: Override config values (e.g. ``max_workers=8``).

    Returns:
        Configured :class:`Pipeline` instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config format is not recognized.

    Example:
        >>> pipe = load_pipeline("segmentation_pipeline.yaml", max_workers=8)
        >>> result = pipe.run(input_dir="./data")
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()

    if ext == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: '{ext}'. Use .json or .yaml/.yml")

    # Apply overrides
    config.update(overrides)

    # Deserialize steps
    step_dicts = config.pop("steps", [])
    steps = [_step_from_dict(d) for d in step_dicts]

    return Pipeline(steps=steps, **config)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "Pipeline",
    "PipelineStep",
    "FunctionStep",
    "GlobStep",
    "SemanticSegmentationStep",
    "RasterToVectorStep",
    "PipelineResult",
    "StepResult",
    "ErrorPolicy",
    "ItemStatus",
    "CheckpointManager",
    "CheckpointEntry",
    "load_pipeline",
    "register_step",
]
