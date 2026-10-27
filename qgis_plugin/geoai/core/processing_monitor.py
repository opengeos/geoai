"""Task lifecycle and optional GPU metrics for the GeoAI QGIS plugin."""

from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """States that can be observed for a processing task."""

    IDLE = "IDLE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class GPUMetrics:
    """Best-effort CUDA metrics in bytes."""

    available: bool
    device_name: Optional[str] = None
    allocated_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None
    peak_allocated_bytes: Optional[int] = None
    unavailable_reason: Optional[str] = None


@dataclass
class ProcessingTask:
    """A task snapshot maintained independently from the Qt UI."""

    task_id: str
    name: str
    status: TaskStatus = TaskStatus.IDLE
    progress: Optional[int] = None
    progress_message: str = ""
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    started_timestamp: Optional[datetime] = None
    ended_timestamp: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    error_summary: Optional[str] = None
    error_details: Optional[str] = None
    gpu: Optional[GPUMetrics] = None
    _last_progress: Optional[int] = field(default=None, repr=False)

    def update_elapsed(self, now: Optional[float] = None) -> None:
        """Update elapsed time using a monotonic clock."""
        if self.started_at is None:
            self.elapsed_seconds = 0.0
            return
        end = now if self.ended_at is None else self.ended_at
        self.elapsed_seconds = max(0.0, end - self.started_at)


def format_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def classify_error(error: BaseException | str) -> str:
    """Return a category only when the exception type/message is reliable."""
    if isinstance(error, (FileNotFoundError, IsADirectoryError)):
        return "Missing or invalid input file"
    if isinstance(error, ImportError):
        return "Missing Python dependency"
    if isinstance(error, PermissionError):
        return "Output or file access failure"
    message = str(error).lower()
    if any(token in message for token in ("cuda", "cudnn", "out of memory", "gpu")):
        return "GPU or CUDA error"
    if any(token in message for token in ("write", "save", "disk full", "output")):
        return "Output writing failure"
    return "Processing failed"


def get_gpu_metrics() -> GPUMetrics:
    """Read optional CUDA metrics without making PyTorch a plugin dependency."""
    try:
        import torch
    except (ImportError, OSError):
        return GPUMetrics(False, unavailable_reason="PyTorch unavailable")

    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return GPUMetrics(False, unavailable_reason="CUDA unavailable")
        device = cuda.current_device()
        return GPUMetrics(
            available=True,
            device_name=cuda.get_device_name(device),
            allocated_bytes=int(cuda.memory_allocated(device)),
            reserved_bytes=int(cuda.memory_reserved(device)),
            peak_allocated_bytes=int(cuda.max_memory_allocated(device)),
        )
    except Exception as exc:
        logger.info("GPU monitoring unavailable: %s", exc)
        return GPUMetrics(False, unavailable_reason="GPU metrics unavailable")


class ProcessingMonitor:
    """Task registry used by worker signal handlers and the Qt UI timer.

    The class has no Qt dependency, which keeps lifecycle behavior unit-testable.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._tasks: Dict[str, ProcessingTask] = {}
        self._listeners: List[Callable[[], None]] = []

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            listener()

    def submit(self, task_id: str, name: str) -> ProcessingTask:
        task = ProcessingTask(task_id=task_id, name=name, status=TaskStatus.QUEUED)
        self._tasks[task_id] = task
        logger.info("AI task submitted: %s (%s)", name, task_id)
        self._notify()
        return task

    def start(self, task_id: str) -> None:
        task = self._tasks[task_id]
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return
        task.status = TaskStatus.RUNNING
        task.started_at = self._clock()
        task.started_timestamp = datetime.now()
        self._reset_gpu_peak_metrics(task_id)
        task.gpu = get_gpu_metrics()
        logger.info("AI task started: %s (%s)", task.name, task_id)
        self._notify()

    def _reset_gpu_peak_metrics(self, task_id: str) -> None:
        if any(
            item.status == TaskStatus.RUNNING and item.task_id != task_id
            for task_id, item in self._tasks.items()
        ):
            return
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (ImportError, OSError, RuntimeError, AttributeError):
            logger.info("GPU peak memory reset unavailable")

    def progress(
        self, task_id: str, message: str = "", percent: Optional[float] = None
    ) -> None:
        task = self._tasks[task_id]
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return
        if percent is not None:
            try:
                percent_value = float(percent)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid progress for task %s", task_id)
                return
            if not math.isfinite(percent_value):
                logger.warning("Ignoring non-finite progress for task %s", task_id)
                return
            percent = max(0, min(100, round(percent_value)))
            task.progress = percent
            if task._last_progress == percent and not message:
                return
            task._last_progress = percent
        task.progress_message = message
        if percent is not None and percent in (0, 100):
            logger.info("AI task progress: %s (%d%%)", task.name, percent)
        self._notify()

    def complete(self, task_id: str) -> None:
        self._finish(task_id, TaskStatus.COMPLETED)

    def cancel(self, task_id: str) -> None:
        self._finish(task_id, TaskStatus.CANCELLED)

    def fail(self, task_id: str, error: BaseException | str) -> None:
        task = self._tasks[task_id]
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return
        if isinstance(error, BaseException):
            task.error_summary = f"{classify_error(error)}: {error}"
            task.error_details = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            logger.error("AI task failed: %s (%s): %s", task.name, task_id, error)
        else:
            task.error_summary = (
                f"{classify_error(error)}: {str(error).splitlines()[0]}"
            )
            task.error_details = str(error)
            logger.error("AI task failed: %s (%s): %s", task.name, task_id, error)
        self._finish(task_id, TaskStatus.FAILED)

    def _finish(self, task_id: str, status: TaskStatus) -> None:
        task = self._tasks[task_id]
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return
        task.status = status
        task.ended_at = self._clock()
        task.ended_timestamp = datetime.now()
        task.update_elapsed()
        task.gpu = get_gpu_metrics()
        logger.info("AI task %s: %s (%s)", status.value.lower(), task.name, task_id)
        self._notify()

    def snapshot(self) -> List[ProcessingTask]:
        now = self._clock()
        for task in self._tasks.values():
            task.update_elapsed(now)
        return list(self._tasks.values())

    def get(self, task_id: str) -> Optional[ProcessingTask]:
        return self._tasks.get(task_id)

    def clear_finished(self) -> None:
        self._tasks = {
            key: task
            for key, task in self._tasks.items()
            if task.status in (TaskStatus.IDLE, TaskStatus.QUEUED, TaskStatus.RUNNING)
        }
        self._notify()


_monitor = ProcessingMonitor()


def get_processing_monitor() -> ProcessingMonitor:
    """Return the plugin-wide monitor registry."""
    return _monitor
