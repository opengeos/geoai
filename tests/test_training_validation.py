"""Tests for training path validation (_validate_training_paths and _check_readable)."""

import os
from unittest.mock import patch

import pytest

from geoai.train import _check_readable, _validate_training_paths

# ---------------------------------------------------------------------------
# _validate_training_paths — directory format
# ---------------------------------------------------------------------------


class TestValidateDirectoryFormat:
    """Tests for the default 'directory' input format."""

    def test_missing_images_dir(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            _validate_training_paths(
                str(tmp_path / "nonexistent"),
                str(labels),
                str(output),
                input_format="directory",
            )

    def test_images_dir_is_file(self, tmp_path):
        fake_file = tmp_path / "images"
        fake_file.write_text("not a dir")
        labels = tmp_path / "labels"
        labels.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="not a directory"):
            _validate_training_paths(
                str(fake_file),
                str(labels),
                str(output),
                input_format="directory",
            )

    def test_missing_labels_dir(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="Labels directory not found"):
            _validate_training_paths(
                str(images),
                str(tmp_path / "nonexistent"),
                str(output),
                input_format="directory",
            )

    def test_labels_dir_is_file(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        fake_file = tmp_path / "labels"
        fake_file.write_text("not a dir")
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="not a directory"):
            _validate_training_paths(
                str(images),
                str(fake_file),
                str(output),
                input_format="directory",
            )

    def test_valid_directory_format(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        labels = tmp_path / "labels"
        labels.mkdir()
        output = tmp_path / "output"

        # Should not raise
        _validate_training_paths(
            str(images), str(labels), str(output), input_format="directory"
        )
        assert os.path.isdir(str(output))


# ---------------------------------------------------------------------------
# _validate_training_paths — COCO format
# ---------------------------------------------------------------------------


class TestValidateCOCOFormat:
    """Tests for 'coco' and 'coco_detection' input formats."""

    def test_coco_missing_json(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="annotations file not found"):
            _validate_training_paths(
                str(images),
                str(tmp_path / "missing.json"),
                str(output),
                input_format="coco",
            )

    def test_coco_labels_is_dir_not_file(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="not a file"):
            _validate_training_paths(
                str(images),
                str(labels_dir),
                str(output),
                input_format="coco",
            )

    def test_coco_detection_missing_json(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="annotations file not found"):
            _validate_training_paths(
                str(images),
                str(tmp_path / "missing.json"),
                str(output),
                input_format="coco_detection",
            )

    def test_valid_coco_format(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        json_file = tmp_path / "instances.json"
        json_file.write_text("{}")
        output = tmp_path / "output"

        # Should not raise
        _validate_training_paths(
            str(images), str(json_file), str(output), input_format="coco"
        )


# ---------------------------------------------------------------------------
# _validate_training_paths — YOLO format
# ---------------------------------------------------------------------------


class TestValidateYOLOFormat:
    """Tests for 'yolo' input format."""

    def test_yolo_missing_images_subdir(self, tmp_path):
        root = tmp_path / "yolo_root"
        root.mkdir()
        (root / "labels").mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="YOLO images subdirectory"):
            _validate_training_paths(
                str(root),
                str(root / "labels"),
                str(output),
                input_format="yolo",
            )

    def test_yolo_missing_labels_subdir(self, tmp_path):
        root = tmp_path / "yolo_root"
        root.mkdir()
        (root / "images").mkdir()
        output = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="YOLO labels subdirectory"):
            _validate_training_paths(
                str(root),
                str(root / "labels"),
                str(output),
                input_format="yolo",
            )

    def test_valid_yolo_format(self, tmp_path):
        root = tmp_path / "yolo_root"
        root.mkdir()
        (root / "images").mkdir()
        (root / "labels").mkdir()
        output = tmp_path / "output"

        # Should not raise
        _validate_training_paths(
            str(root), str(root / "labels"), str(output), input_format="yolo"
        )


# ---------------------------------------------------------------------------
# _validate_training_paths — output_dir
# ---------------------------------------------------------------------------


class TestValidateOutputDir:
    """Tests for output directory creation."""

    def test_output_dir_created(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        labels = tmp_path / "labels"
        labels.mkdir()
        output = tmp_path / "new_output" / "nested"

        _validate_training_paths(
            str(images), str(labels), str(output), input_format="directory"
        )
        assert os.path.isdir(str(output))

    def test_output_dir_is_file(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        labels = tmp_path / "labels"
        labels.mkdir()
        output = tmp_path / "output_file"
        output.write_text("I am a file, not a directory")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            _validate_training_paths(
                str(images), str(labels), str(output), input_format="directory"
            )


# ---------------------------------------------------------------------------
# _check_readable
# ---------------------------------------------------------------------------


class TestCheckReadable:
    """Tests for _check_readable with retry logic."""

    def test_readable_directory(self, tmp_path):
        # Should not raise
        _check_readable(str(tmp_path))

    def test_readable_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        # Should not raise
        _check_readable(str(f))

    @patch("geoai.train.platform")
    def test_permission_denied_no_retry_on_linux(self, mock_platform, tmp_path):
        mock_platform.system.return_value = "Linux"
        target = tmp_path / "locked"
        target.mkdir()

        with patch("geoai.train.os.listdir", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError, match="permission denied"):
                _check_readable(str(target))

    @patch("geoai.train.platform")
    @patch("geoai.train.time.sleep")
    def test_retry_on_windows_then_succeed(self, mock_sleep, mock_platform, tmp_path):
        mock_platform.system.return_value = "Windows"
        target = tmp_path / "locked"
        target.mkdir()

        call_count = 0
        original_listdir = os.listdir

        def flaky_listdir(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("transient lock")
            return original_listdir(path)

        with patch("geoai.train.os.listdir", side_effect=flaky_listdir):
            # Should succeed on second attempt
            _check_readable(str(target), max_retries=3)

        assert call_count == 2
        mock_sleep.assert_called_once()

    @patch("geoai.train.platform")
    @patch("geoai.train.time.sleep")
    def test_retry_on_windows_all_fail(self, mock_sleep, mock_platform, tmp_path):
        mock_platform.system.return_value = "Windows"
        target = tmp_path / "locked"
        target.mkdir()

        with patch(
            "geoai.train.os.listdir", side_effect=PermissionError("always locked")
        ):
            with pytest.raises(PermissionError, match="permission denied"):
                _check_readable(str(target), max_retries=3)

        assert mock_sleep.call_count == 2  # retries between attempts, not after last

    @patch("geoai.train.platform")
    @patch("geoai.train.time.sleep")
    def test_windows_error_message_includes_hint(
        self, mock_sleep, mock_platform, tmp_path
    ):
        mock_platform.system.return_value = "Windows"
        target = tmp_path / "locked"
        target.mkdir()

        with patch("geoai.train.os.listdir", side_effect=PermissionError("locked")):
            with pytest.raises(PermissionError, match="Windows Defender"):
                _check_readable(str(target), max_retries=1)
