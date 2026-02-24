#!/usr/bin/env python

"""Tests for `geoai.tools` subpackage."""

import inspect
import os
import unittest


class TestToolsImport(unittest.TestCase):
    """Tests for tools subpackage import behavior."""

    def test_tools_package_imports(self):
        """Test that the tools package can be imported."""
        import geoai.tools

        self.assertTrue(hasattr(geoai.tools, "__all__"))

    def test_all_is_list(self):
        """Test that __all__ is a list."""
        import geoai.tools

        self.assertIsInstance(geoai.tools.__all__, list)


class TestCloudMaskModule(unittest.TestCase):
    """Tests for the cloudmask module."""

    def test_module_imports(self):
        """Test that cloudmask module can be imported."""
        from geoai.tools import cloudmask

        self.assertTrue(hasattr(cloudmask, "CLEAR"))
        self.assertTrue(hasattr(cloudmask, "THICK_CLOUD"))
        self.assertTrue(hasattr(cloudmask, "THIN_CLOUD"))
        self.assertTrue(hasattr(cloudmask, "CLOUD_SHADOW"))

    def test_cloud_mask_constants(self):
        """Test cloud mask constant values."""
        from geoai.tools.cloudmask import (
            CLEAR,
            CLOUD_SHADOW,
            THICK_CLOUD,
            THIN_CLOUD,
        )

        self.assertEqual(CLEAR, 0)
        self.assertEqual(THICK_CLOUD, 1)
        self.assertEqual(THIN_CLOUD, 2)
        self.assertEqual(CLOUD_SHADOW, 3)

    def test_check_omnicloudmask_available_exists(self):
        """Test that check_omnicloudmask_available function exists."""
        from geoai.tools.cloudmask import check_omnicloudmask_available

        self.assertTrue(callable(check_omnicloudmask_available))

    def test_predict_cloud_mask_exists(self):
        """Test that predict_cloud_mask function exists and is callable."""
        from geoai.tools.cloudmask import predict_cloud_mask

        self.assertTrue(callable(predict_cloud_mask))

    def test_predict_cloud_mask_signature(self):
        """Test predict_cloud_mask has expected parameters."""
        from geoai.tools.cloudmask import predict_cloud_mask

        sig = inspect.signature(predict_cloud_mask)
        self.assertIn("image", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("inference_device", sig.parameters)


class TestMultiCleanModule(unittest.TestCase):
    """Tests for the multiclean module."""

    def test_module_imports(self):
        """Test that multiclean module can be imported."""
        from geoai.tools import multiclean

        self.assertTrue(hasattr(multiclean, "check_multiclean_available"))

    def test_check_multiclean_available_exists(self):
        """Test that check_multiclean_available function exists."""
        from geoai.tools.multiclean import check_multiclean_available

        self.assertTrue(callable(check_multiclean_available))

    def test_clean_segmentation_mask_exists(self):
        """Test that clean_segmentation_mask function exists."""
        from geoai.tools.multiclean import clean_segmentation_mask

        self.assertTrue(callable(clean_segmentation_mask))

    def test_clean_segmentation_mask_signature(self):
        """Test clean_segmentation_mask has expected parameters."""
        from geoai.tools.multiclean import clean_segmentation_mask

        sig = inspect.signature(clean_segmentation_mask)
        self.assertIn("mask", sig.parameters)
        self.assertIn("smooth_edge_size", sig.parameters)
        self.assertIn("min_island_size", sig.parameters)


class TestTimeseriesModule(unittest.TestCase):
    """Tests for the timeseries module."""

    def test_module_imports(self):
        """Test that timeseries module can be imported."""
        from geoai.tools import timeseries

        self.assertTrue(hasattr(timeseries, "COMPOSITE_METHODS"))
        self.assertTrue(hasattr(timeseries, "SPECTRAL_INDICES"))
        self.assertTrue(hasattr(timeseries, "CHANGE_METHODS"))
        self.assertTrue(hasattr(timeseries, "TEMPORAL_STATISTICS"))

    def test_constants_values(self):
        """Test timeseries constant values."""
        from geoai.tools.timeseries import (
            CHANGE_METHODS,
            COMPOSITE_METHODS,
            SPECTRAL_INDICES,
            TEMPORAL_STATISTICS,
        )

        self.assertIn("median", COMPOSITE_METHODS)
        self.assertIn("mean", COMPOSITE_METHODS)
        self.assertIn("medoid", COMPOSITE_METHODS)
        self.assertIn("NDVI", SPECTRAL_INDICES)
        self.assertIn("NDWI", SPECTRAL_INDICES)
        self.assertIn("EVI", SPECTRAL_INDICES)
        self.assertIn("difference", CHANGE_METHODS)
        self.assertIn("ratio", CHANGE_METHODS)
        self.assertIn("mean", TEMPORAL_STATISTICS)
        self.assertIn("std", TEMPORAL_STATISTICS)
        self.assertIn("count", TEMPORAL_STATISTICS)

    def test_date_patterns(self):
        """Test date pattern constants are valid regex."""
        import re

        from geoai.tools.timeseries import (
            GENERIC_DATE_PATTERN,
            LANDSAT_DATE_PATTERN,
            SENTINEL2_DATE_PATTERN,
        )

        for pattern in [
            SENTINEL2_DATE_PATTERN,
            LANDSAT_DATE_PATTERN,
            GENERIC_DATE_PATTERN,
        ]:
            compiled = re.compile(pattern)
            self.assertGreaterEqual(compiled.groups, 1)

    def test_check_rasterio_available_exists(self):
        """Test that check_rasterio_available function exists."""
        from geoai.tools.timeseries import check_rasterio_available

        self.assertTrue(callable(check_rasterio_available))

    def test_validate_temporal_stack_exists(self):
        """Test that validate_temporal_stack function exists."""
        from geoai.tools.timeseries import validate_temporal_stack

        self.assertTrue(callable(validate_temporal_stack))

    def test_validate_temporal_stack_signature(self):
        """Test validate_temporal_stack has expected parameters."""
        from geoai.tools.timeseries import validate_temporal_stack

        sig = inspect.signature(validate_temporal_stack)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("tolerance", sig.parameters)

    def test_create_temporal_composite_exists(self):
        """Test that create_temporal_composite function exists."""
        from geoai.tools.timeseries import create_temporal_composite

        self.assertTrue(callable(create_temporal_composite))

    def test_create_temporal_composite_signature(self):
        """Test create_temporal_composite has expected parameters."""
        from geoai.tools.timeseries import create_temporal_composite

        sig = inspect.signature(create_temporal_composite)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("method", sig.parameters)
        self.assertIn("cloud_masks", sig.parameters)

    def test_create_cloud_free_composite_exists(self):
        """Test that create_cloud_free_composite function exists."""
        from geoai.tools.timeseries import create_cloud_free_composite

        self.assertTrue(callable(create_cloud_free_composite))

    def test_create_cloud_free_composite_signature(self):
        """Test create_cloud_free_composite has expected parameters."""
        from geoai.tools.timeseries import create_cloud_free_composite

        sig = inspect.signature(create_cloud_free_composite)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("red_band", sig.parameters)
        self.assertIn("green_band", sig.parameters)
        self.assertIn("nir_band", sig.parameters)
        self.assertIn("method", sig.parameters)
        self.assertIn("include_thin_clouds", sig.parameters)

    def test_calculate_spectral_index_timeseries_exists(self):
        """Test that calculate_spectral_index_timeseries function exists."""
        from geoai.tools.timeseries import calculate_spectral_index_timeseries

        self.assertTrue(callable(calculate_spectral_index_timeseries))

    def test_calculate_spectral_index_timeseries_signature(self):
        """Test calculate_spectral_index_timeseries has expected parameters."""
        from geoai.tools.timeseries import calculate_spectral_index_timeseries

        sig = inspect.signature(calculate_spectral_index_timeseries)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("index_type", sig.parameters)
        self.assertIn("scale_factor", sig.parameters)

    def test_detect_change_exists(self):
        """Test that detect_change function exists."""
        from geoai.tools.timeseries import detect_change

        self.assertTrue(callable(detect_change))

    def test_detect_change_signature(self):
        """Test detect_change has expected parameters."""
        from geoai.tools.timeseries import detect_change

        sig = inspect.signature(detect_change)
        self.assertIn("image1_path", sig.parameters)
        self.assertIn("image2_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("method", sig.parameters)
        self.assertIn("threshold", sig.parameters)

    def test_calculate_temporal_statistics_exists(self):
        """Test that calculate_temporal_statistics function exists."""
        from geoai.tools.timeseries import calculate_temporal_statistics

        self.assertTrue(callable(calculate_temporal_statistics))

    def test_calculate_temporal_statistics_signature(self):
        """Test calculate_temporal_statistics has expected parameters."""
        from geoai.tools.timeseries import calculate_temporal_statistics

        sig = inspect.signature(calculate_temporal_statistics)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("statistics", sig.parameters)
        self.assertIn("bands", sig.parameters)

    def test_extract_dates_from_filenames_exists(self):
        """Test that extract_dates_from_filenames function exists."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        self.assertTrue(callable(extract_dates_from_filenames))

    def test_sort_by_date_exists(self):
        """Test that sort_by_date function exists."""
        from geoai.tools.timeseries import sort_by_date

        self.assertTrue(callable(sort_by_date))

    def test_extract_dates_from_sentinel2_filenames(self):
        """Test date extraction from Sentinel-2 filenames."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        files = [
            "S2A_MSIL2A_20230115T101301_N0509.tif",
            "S2A_MSIL2A_20230315T101301_N0509.tif",
        ]
        dates = extract_dates_from_filenames(files)
        self.assertEqual(len(dates), 2)
        self.assertIsNotNone(dates[0])
        self.assertIsNotNone(dates[1])
        self.assertEqual(dates[0].year, 2023)
        self.assertEqual(dates[0].month, 1)
        self.assertEqual(dates[0].day, 15)
        self.assertEqual(dates[1].month, 3)

    def test_extract_dates_from_landsat_filenames(self):
        """Test date extraction from Landsat filenames."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        files = [
            "LC08_L2SP_015033_20230601_20230607_02_T1.tif",
            "LC08_L2SP_015033_20231015_20231020_02_T1.tif",
        ]
        dates = extract_dates_from_filenames(files)
        self.assertEqual(len(dates), 2)
        self.assertIsNotNone(dates[0])
        self.assertIsNotNone(dates[1])
        self.assertEqual(dates[0].year, 2023)
        self.assertEqual(dates[0].month, 6)
        self.assertEqual(dates[0].day, 1)

    def test_extract_dates_generic_pattern(self):
        """Test date extraction from generic date patterns."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        files = [
            "scene_2023-01-15.tif",
            "scene_2023_06_01.tif",
            "scene_20230915.tif",
        ]
        dates = extract_dates_from_filenames(files)
        self.assertEqual(len(dates), 3)
        for d in dates:
            self.assertIsNotNone(d)
        self.assertEqual(dates[0].month, 1)
        self.assertEqual(dates[1].month, 6)
        self.assertEqual(dates[2].month, 9)

    def test_extract_dates_no_date_returns_none(self):
        """Test that files without dates return None."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        files = ["no_date_here.tif", "also_nothing.tif"]
        dates = extract_dates_from_filenames(files)
        self.assertEqual(len(dates), 2)
        self.assertIsNone(dates[0])
        self.assertIsNone(dates[1])

    def test_extract_dates_custom_pattern(self):
        """Test date extraction with a custom pattern."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        files = ["data_D20230115.tif", "data_D20230601.tif"]
        dates = extract_dates_from_filenames(
            files, date_pattern=r"D(\d{8})", date_format="%Y%m%d"
        )
        self.assertEqual(len(dates), 2)
        self.assertIsNotNone(dates[0])
        self.assertEqual(dates[0].month, 1)
        self.assertEqual(dates[1].month, 6)

    def test_extract_dates_invalid_pattern_raises(self):
        """Test that a pattern without capture group raises ValueError."""
        from geoai.tools.timeseries import extract_dates_from_filenames

        with self.assertRaises(ValueError):
            extract_dates_from_filenames(["file.tif"], date_pattern=r"\d{8}")

    def test_sort_by_date_ordering(self):
        """Test that sort_by_date produces chronological order."""
        from datetime import datetime

        from geoai.tools.timeseries import sort_by_date

        files = ["c.tif", "a.tif", "b.tif"]
        dates = [
            datetime(2023, 6, 1),
            datetime(2023, 1, 1),
            datetime(2023, 3, 15),
        ]
        sorted_files, sorted_dates = sort_by_date(files, dates=dates)
        self.assertEqual(sorted_files, ["a.tif", "b.tif", "c.tif"])
        self.assertEqual(sorted_dates[0], datetime(2023, 1, 1))
        self.assertEqual(sorted_dates[1], datetime(2023, 3, 15))
        self.assertEqual(sorted_dates[2], datetime(2023, 6, 1))

    def test_sort_by_date_length_mismatch_raises(self):
        """Test that mismatched dates length raises ValueError."""
        from datetime import datetime

        from geoai.tools.timeseries import sort_by_date

        with self.assertRaises(ValueError):
            sort_by_date(["a.tif", "b.tif"], dates=[datetime(2023, 1, 1)])

    def test_sort_by_date_no_dates_raises(self):
        """Test that no extractable dates raises ValueError."""
        from geoai.tools.timeseries import sort_by_date

        with self.assertRaises(ValueError):
            sort_by_date(["no_date.tif", "also_no_date.tif"])

    def test_sort_by_date_from_filenames(self):
        """Test sort_by_date with automatic date extraction."""
        from geoai.tools.timeseries import sort_by_date

        files = [
            "scene_20230601.tif",
            "scene_20230101.tif",
            "scene_20230315.tif",
        ]
        sorted_files, sorted_dates = sort_by_date(files)
        self.assertEqual(
            [os.path.basename(f) for f in sorted_files],
            ["scene_20230101.tif", "scene_20230315.tif", "scene_20230601.tif"],
        )


if __name__ == "__main__":
    unittest.main()
