"""Pure-Python tests for the QGIS plugin processing monitor."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "qgis_plugin"
    / "geoai"
    / "core"
    / "processing_monitor.py"
)
SPEC = importlib.util.spec_from_file_location("qgis_processing_monitor", MODULE_PATH)
monitor_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = monitor_module
SPEC.loader.exec_module(monitor_module)


def test_task_lifecycle_uses_monotonic_elapsed_time():
    now = iter((10.0, 12.5, 15.0))
    monitor = monitor_module.ProcessingMonitor(clock=lambda: next(now))

    monitor.submit("task-1", "Inference")
    monitor.start("task-1")
    assert monitor.get("task-1").status == monitor_module.TaskStatus.RUNNING
    assert monitor.snapshot()[0].elapsed_seconds == 2.5

    monitor.complete("task-1")
    task = monitor.get("task-1")
    assert task.status == monitor_module.TaskStatus.COMPLETED
    assert task.elapsed_seconds == 5.0
    assert task.ended_at == 15.0


def test_new_task_has_independent_lifecycle_data():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("first", "First")
    monitor.start("first")
    monitor.fail("first", "failed")

    monitor.submit("second", "Second")
    task = monitor.get("second")

    assert task.status == monitor_module.TaskStatus.QUEUED
    assert task.started_at is None
    assert task.ended_at is None
    assert task.error_summary is None


def test_failure_preserves_error_details_and_terminal_state():
    monitor = monitor_module.ProcessingMonitor(clock=iter((1.0, 2.0)).__next__)
    monitor.submit("task-1", "Inference")
    monitor.start("task-1")

    try:
        raise FileNotFoundError("input.tif")
    except FileNotFoundError as error:
        monitor.fail("task-1", error)

    task = monitor.get("task-1")
    assert task.status == monitor_module.TaskStatus.FAILED
    assert task.error_summary.startswith("Missing or invalid input file:")
    assert "FileNotFoundError" in task.error_details
    monitor.complete("task-1")
    assert task.status == monitor_module.TaskStatus.FAILED


@pytest.mark.parametrize(
    ("error", "summary"),
    [
        ("single line", "Processing failed: single line"),
        ("first line\ntechnical detail", "Processing failed: first line"),
        ("", "Processing failed"),
    ],
)
def test_failure_summary_uses_first_line(error, summary):
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")

    monitor.fail("task-1", error)

    task = monitor.get("task-1")
    assert task.error_summary == summary
    assert task.error_details == error


def test_exception_summary_uses_first_line_and_preserves_category():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")

    monitor.fail(
        "task-1",
        FileNotFoundError("missing.tif\nadditional diagnostic detail"),
    )

    task = monitor.get("task-1")
    assert task.error_summary == "Missing or invalid input file: missing.tif"
    assert "additional diagnostic detail" in task.error_details


def test_repeated_terminal_callbacks_are_ignored():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")
    monitor.start("task-1")
    monitor.complete("task-1")
    ended_at = monitor.get("task-1").ended_at

    monitor.complete("task-1")
    monitor.fail("task-1", RuntimeError("stale error"))

    task = monitor.get("task-1")
    assert task.status == monitor_module.TaskStatus.COMPLETED
    assert task.ended_at == ended_at
    assert task.error_summary is None


def test_stale_start_callback_does_not_reopen_terminal_task():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")
    monitor.start("task-1")
    monitor.complete("task-1")

    monitor.start("task-1")

    assert monitor.get("task-1").status == monitor_module.TaskStatus.COMPLETED


def test_progress_is_clamped_and_unknown_progress_is_indeterminate():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Export")
    monitor.progress("task-1", "Preparing")
    assert monitor.get("task-1").progress is None

    monitor.progress("task-1", "Working", 125)
    assert monitor.get("task-1").progress == 100
    monitor.progress("task-1", "Restarting", -10)
    assert monitor.get("task-1").progress == 0


@pytest.mark.parametrize("invalid_progress", [float("nan"), float("inf"), "invalid"])
def test_invalid_progress_is_ignored(invalid_progress):
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Export")
    monitor.progress("task-1", "Working", 25)

    monitor.progress("task-1", "Bad update", invalid_progress)

    task = monitor.get("task-1")
    assert task.progress == 25
    assert task.progress_message == "Working"


def test_progress_after_completion_does_not_change_terminal_task():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Export")
    monitor.start("task-1")
    monitor.complete("task-1")

    monitor.progress("task-1", "Stale update", 90)

    task = monitor.get("task-1")
    assert task.status == monitor_module.TaskStatus.COMPLETED
    assert task.progress is None
    assert task.progress_message == ""


def test_error_categories_and_technical_logging(caplog):
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")
    monitor.start("task-1")

    with caplog.at_level("ERROR", logger=monitor_module.logger.name):
        monitor.fail("task-1", RuntimeError("CUDA out of memory"))

    task = monitor.get("task-1")
    assert task.error_summary == "GPU or CUDA error: CUDA out of memory"
    assert "CUDA out of memory" in task.error_details
    assert "AI task failed" in caplog.text


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ImportError("model dependency"), "Missing Python dependency"),
        (RuntimeError("input file does not exist"), "Missing or invalid input file"),
        (RuntimeError("No module named 'torch'"), "Missing Python dependency"),
        (PermissionError("cannot save output"), "Output or file access failure"),
        (RuntimeError("failed to write output"), "Output writing failure"),
        (RuntimeError("unexpected failure"), "Processing failed"),
    ],
)
def test_error_categories_are_best_effort(error, category):
    assert monitor_module.classify_error(error) == category


def test_listener_notifications_can_be_registered_and_removed():
    monitor = monitor_module.ProcessingMonitor()
    notifications = []

    def listener():
        notifications.append(True)

    monitor.add_listener(listener)
    monitor.submit("task-1", "Inference")
    monitor.remove_listener(listener)
    monitor.start("task-1")

    assert notifications == [True]


def test_permission_error_keeps_file_access_category():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Output")
    monitor.start("task-1")

    monitor.fail("task-1", PermissionError("cannot save output"))

    assert monitor.get("task-1").error_summary == (
        "Output or file access failure: cannot save output"
    )


def test_whitespace_only_error_still_finishes_task():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")
    monitor.start("task-1")

    monitor.fail("task-1", "\n")

    assert monitor.get("task-1").status == monitor_module.TaskStatus.FAILED
    assert monitor.get("task-1").error_summary == "Processing failed"


def test_gpu_metrics_are_optional_when_cuda_is_unavailable():
    class CpuCuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = CpuCuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is False
    assert metrics.unavailable_reason == "CUDA unavailable"


def test_gpu_metrics_report_supported_cuda_values():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "Test GPU"

        @staticmethod
        def memory_allocated(_device):
            return 10

        @staticmethod
        def memory_reserved(_device):
            return 20

        @staticmethod
        def max_memory_allocated(_device):
            return 30

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is True
    assert metrics.device_name == "Test GPU"
    assert metrics.allocated_bytes == 10
    assert metrics.reserved_bytes == 20
    assert metrics.peak_allocated_bytes == 30


def test_gpu_metrics_skip_uninitialized_cuda():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def current_device():
            raise AssertionError("CUDA calls should be skipped")

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is False
    assert metrics.unavailable_reason == "CUDA unavailable"


def test_peak_memory_reset_skips_uninitialized_cuda():
    reset_calls = []

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def reset_peak_memory_stats():
            reset_calls.append(True)

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        monitor = monitor_module.ProcessingMonitor()
        monitor.submit("task-1", "Inference")
        monitor.start("task-1")

    assert reset_calls == []


def test_peak_memory_reset_failure_does_not_block_task_start():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            raise ValueError("reset failed")

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        monitor = monitor_module.ProcessingMonitor()
        monitor.submit("task-1", "Inference")
        monitor.start("task-1")

    assert monitor.get("task-1").status == monitor_module.TaskStatus.RUNNING


def test_gpu_metrics_support_cuda_double_without_initialization_api():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "Test GPU"

        memory_allocated = staticmethod(lambda _device: 1)
        memory_reserved = staticmethod(lambda _device: 2)
        max_memory_allocated = staticmethod(lambda _device: 3)

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is True


def test_gpu_metrics_do_not_access_torch_when_absent():
    with patch.dict("sys.modules", {"torch": None}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is False
    assert metrics.unavailable_reason == "PyTorch unavailable"


def test_peak_memory_reset_is_preserved_for_concurrent_tasks():
    reset_calls = []

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            reset_calls.append(True)

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "Test GPU"

        memory_allocated = staticmethod(lambda _device: 1)
        memory_reserved = staticmethod(lambda _device: 2)
        max_memory_allocated = staticmethod(lambda _device: 3)

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        monitor = monitor_module.ProcessingMonitor()
        monitor.submit("first", "First")
        monitor.start("first")
        assert len(reset_calls) == 1
        monitor.submit("second", "Second")
        monitor.start("second")

    assert len(reset_calls) == 1


def test_gpu_metrics_do_not_clear_cuda_cache():
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "Test GPU"

        @staticmethod
        def memory_allocated(_device):
            return 1

        @staticmethod
        def memory_reserved(_device):
            return 2

        @staticmethod
        def max_memory_allocated(_device):
            return 3

        @staticmethod
        def empty_cache():
            raise AssertionError("cache clearing is not part of monitoring")

    class Torch:
        cuda = Cuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.reserved_bytes == 2


def test_gpu_metric_errors_do_not_fail_processing():
    class BrokenCuda:
        @staticmethod
        def is_available():
            raise RuntimeError("driver unavailable")

    class Torch:
        cuda = BrokenCuda()

    with patch.dict("sys.modules", {"torch": Torch()}):
        metrics = monitor_module.get_gpu_metrics()

    assert metrics.available is False
    assert metrics.unavailable_reason == "GPU metrics unavailable"


def test_cancellation_is_terminal():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("task-1", "Inference")
    monitor.cancel("task-1")

    assert monitor.get("task-1").status == monitor_module.TaskStatus.CANCELLED
    monitor.complete("task-1")
    assert monitor.get("task-1").status == monitor_module.TaskStatus.CANCELLED


def test_clear_finished_keeps_active_tasks():
    monitor = monitor_module.ProcessingMonitor()
    monitor.submit("active", "Active")
    monitor.submit("done", "Done")
    monitor.complete("done")
    monitor.clear_finished()

    assert monitor.get("active") is not None
    assert monitor.get("done") is None
