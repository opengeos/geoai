#!/usr/bin/env python

"""Tests for `geoai.progress` and the ``quiet``/``verbose`` flags that use it."""

import ast
import contextlib
import os
import pathlib
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from geoai import progress


class TestProgressSwitch(unittest.TestCase):
    """Tests for the package-wide progress bar switch."""

    def setUp(self):
        """Remember the current switch state."""
        self._original = progress._DISABLED

    def tearDown(self):
        """Restore the original switch state."""
        progress._DISABLED = self._original

    def test_enabled_by_default(self):
        """Progress bars are shown unless explicitly disabled."""
        progress._DISABLED = None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEOAI_DISABLE_PROGRESS", None)
            self.assertFalse(progress.progress_bars_disabled())

    def test_disable_and_enable(self):
        """disable_progress_bars/enable_progress_bars flip the switch."""
        progress.disable_progress_bars()
        self.assertTrue(progress.progress_bars_disabled())
        progress.enable_progress_bars()
        self.assertFalse(progress.progress_bars_disabled())

    def test_set_progress_bars(self):
        """set_progress_bars takes an ``enabled`` flag."""
        progress.set_progress_bars(False)
        self.assertTrue(progress.progress_bars_disabled())
        progress.set_progress_bars(True)
        self.assertFalse(progress.progress_bars_disabled())

    def test_environment_variable(self):
        """GEOAI_DISABLE_PROGRESS disables progress bars when unset in code."""
        progress._DISABLED = None
        with patch.dict(os.environ, {"GEOAI_DISABLE_PROGRESS": "1"}):
            self.assertTrue(progress.progress_bars_disabled())
        with patch.dict(os.environ, {"GEOAI_DISABLE_PROGRESS": "0"}):
            self.assertFalse(progress.progress_bars_disabled())

    def test_explicit_call_overrides_environment_variable(self):
        """An explicit call wins over the environment variable."""
        with patch.dict(os.environ, {"GEOAI_DISABLE_PROGRESS": "1"}):
            progress.enable_progress_bars()
            self.assertFalse(progress.progress_bars_disabled())

    def test_tqdm_respects_global_switch(self):
        """The tqdm wrapper is disabled when progress bars are turned off."""
        progress.disable_progress_bars()
        with progress.tqdm(range(3)) as pbar:
            self.assertTrue(pbar.disable)

        progress.enable_progress_bars()
        with progress.tqdm(range(3), disable=True) as pbar:
            self.assertTrue(pbar.disable)
        with progress.tqdm(range(3), disable=False) as pbar:
            self.assertFalse(pbar.disable)

    def test_tqdm_still_iterates_when_disabled(self):
        """Disabling progress bars must not change iteration results."""
        progress.disable_progress_bars()
        self.assertEqual(list(progress.tqdm(range(4))), [0, 1, 2, 3])

    def test_exposed_from_top_level_package(self):
        """The switch is reachable as ``geoai.disable_progress_bars`` etc."""
        import geoai

        self.assertIs(geoai.disable_progress_bars, progress.disable_progress_bars)
        self.assertIs(geoai.enable_progress_bars, progress.enable_progress_bars)
        self.assertIs(geoai.set_progress_bars, progress.set_progress_bars)
        self.assertIs(geoai.progress_bars_disabled, progress.progress_bars_disabled)


class TestNoDirectTqdmImports(unittest.TestCase):
    """Guard the invariant that all progress bars go through geoai.progress."""

    def test_modules_import_tqdm_from_geoai_progress(self):
        """No geoai module imports tqdm directly."""
        package_dir = pathlib.Path(progress.__file__).parent
        offenders = []
        for path in sorted(package_dir.rglob("*.py")):
            if path.name == "progress.py":
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "tqdm"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "tqdm":
                            offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            f"Import tqdm from geoai.progress instead: {offenders}",
        )


class TestQuietFlags(unittest.TestCase):
    """Tests that ``quiet``/``verbose`` flags actually suppress progress bars."""

    def setUp(self):
        """Remember the current switch state."""
        self._original = progress._DISABLED
        progress._DISABLED = False

    def tearDown(self):
        """Restore the original switch state."""
        progress._DISABLED = self._original

    @contextlib.contextmanager
    def _record_bars(self):
        """Record the resolved ``disable`` flag of every progress bar created.

        The flag has to be captured at construction time because ``tqdm.close()``
        sets ``disable`` to True on the way out.

        Yields:
            list: Booleans, one per progress bar created inside the context.
        """
        flags = []
        original_init = progress.tqdm.__init__

        def record_init(pbar, *args, **kwargs):
            original_init(pbar, *args, **kwargs)
            flags.append(pbar.disable)

        with patch.object(progress.tqdm, "__init__", record_init):
            yield flags

    @staticmethod
    def _create_data(tmp_dir):
        """Create a small raster and matching vector file for tiling."""
        raster_path = os.path.join(tmp_dir, "raster.tif")
        vector_path = os.path.join(tmp_dir, "vector.geojson")

        width = height = 300
        data = np.random.randint(0, 256, (3, height, width), dtype=np.uint8)
        transform = from_bounds(-122.5, 37.7, -122.3, 37.9, width, height)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        gdf = gpd.GeoDataFrame(
            {"class": [1]},
            geometry=[box(-122.45, 37.75, -122.4, 37.8)],
            crs="EPSG:4326",
        )
        gdf.to_file(vector_path, driver="GeoJSON")
        return raster_path, vector_path

    def test_export_geotiff_tiles_quiet(self):
        """export_geotiff_tiles(quiet=True) writes no progress output."""
        import tempfile

        from geoai.utils import export_geotiff_tiles

        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path, vector_path = self._create_data(tmp_dir)
            with self._record_bars() as created:
                export_geotiff_tiles(
                    in_raster=raster_path,
                    out_folder=os.path.join(tmp_dir, "output"),
                    in_class_data=vector_path,
                    tile_size=128,
                    stride=64,
                    quiet=True,
                )

            self.assertTrue(created, "expected a progress bar to be created")
            self.assertTrue(all(created), "expected every progress bar to be disabled")

    def test_export_geotiff_tiles_progress_shown_by_default(self):
        """The progress bar is still enabled when quiet=False."""
        import tempfile

        from geoai.utils import export_geotiff_tiles

        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path, vector_path = self._create_data(tmp_dir)
            with self._record_bars() as created:
                export_geotiff_tiles(
                    in_raster=raster_path,
                    out_folder=os.path.join(tmp_dir, "output"),
                    in_class_data=vector_path,
                    tile_size=128,
                    stride=64,
                    quiet=False,
                )

            self.assertTrue(
                any(not flag for flag in created),
                "expected at least one visible progress bar",
            )

    def test_export_geotiff_tiles_respects_global_switch(self):
        """Globally disabling progress bars silences export_geotiff_tiles."""
        import tempfile

        from geoai.utils import export_geotiff_tiles

        progress.disable_progress_bars()
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path, vector_path = self._create_data(tmp_dir)
            with self._record_bars() as created:
                export_geotiff_tiles(
                    in_raster=raster_path,
                    out_folder=os.path.join(tmp_dir, "output"),
                    in_class_data=vector_path,
                    tile_size=128,
                    stride=64,
                    quiet=False,
                )

            self.assertTrue(created, "expected a progress bar to be created")
            self.assertTrue(all(created), "expected every progress bar to be disabled")

    @staticmethod
    def _stub_detector(chip_size=(128, 128)):
        """Build an ObjectDetector that runs without downloading model weights.

        Args:
            chip_size (tuple): Chip size used by the inference dataset.

        Returns:
            ObjectDetector: An instance whose model returns no detections.
        """
        import torch

        from geoai.extract import ObjectDetector

        def stub_model(images):
            """Return an empty prediction for every image in the batch."""
            return [
                {
                    "masks": torch.zeros((0, 1, *chip_size)),
                    "scores": torch.zeros(0),
                }
                for _ in images
            ]

        detector = ObjectDetector.__new__(ObjectDetector)
        detector.device = torch.device("cpu")
        detector.model = stub_model
        detector.chip_size = chip_size
        detector.confidence_threshold = 0.5
        detector.mask_threshold = 0.5
        return detector

    def test_generate_masks_quiet_when_not_verbose(self):
        """generate_masks(verbose=False) creates no visible progress bar."""
        import tempfile

        detector = self._stub_detector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path, _ = self._create_data(tmp_dir)
            with self._record_bars() as created:
                with self.assertNoLogs("geoai.extract", level="INFO"):
                    detector.generate_masks(
                        raster_path=raster_path,
                        output_path=os.path.join(tmp_dir, "masks.tif"),
                        verbose=False,
                    )

            self.assertTrue(created, "expected a progress bar to be created")
            self.assertTrue(all(created), "expected every progress bar to be disabled")

    def test_generate_masks_progress_shown_when_verbose(self):
        """generate_masks(verbose=True) still shows the progress bar."""
        import tempfile

        detector = self._stub_detector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path, _ = self._create_data(tmp_dir)
            with self._record_bars() as created:
                detector.generate_masks(
                    raster_path=raster_path,
                    output_path=os.path.join(tmp_dir, "masks.tif"),
                    verbose=True,
                )

            self.assertTrue(
                any(not flag for flag in created),
                "expected at least one visible progress bar",
            )


if __name__ == "__main__":
    unittest.main()
