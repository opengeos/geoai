import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import sys
import numpy as np
import rasterio

import geoai
from geoai.osd import classify_optically_shallow_deep


class TestOSDClassification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = (
            Path(self.temp_dir.name)
            / "S2A_MSIL1C_20230515T012345_N0509_R068_T54NVD_20230515T012345.SAFE"
        )
        self.input_dir.mkdir(parents=True)

        # Create minimal Sentinel-2 L1C SAFE structure
        (self.input_dir / "GRANULE").mkdir()
        (self.input_dir / "MTD_MSIL1C.xml").write_text("<xml></xml>")

        img_data_dir = self.input_dir / "GRANULE" / "L1C_T54NVD" / "IMG_DATA"
        img_data_dir.mkdir(parents=True)

        # Create dummy jp2 band files
        self.bands = ["B02", "B03", "B04", "B05", "B08", "B11"]
        for band in self.bands:
            band_path = img_data_dir / f"T54NVD_20230515T012345_{band}.jp2"
            with rasterio.open(
                band_path,
                "w",
                driver="JP2OpenJPEG",
                width=10,
                height=10,
                count=1,
                dtype=rasterio.uint16,
                crs="EPSG:32654",
                transform=rasterio.transform.from_origin(0, 10, 1, 1),
            ) as dst:
                dst.write(np.ones((10, 10), dtype=np.uint16), 1)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _mock_side_effect(self, file_L1C, folder_out, file_L2R=None, to_log=False):
        out_file = Path(folder_out) / "dummy_OSW_ODW.tif"
        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype=rasterio.uint8,
            crs="EPSG:32654",
            transform=rasterio.transform.from_origin(0, 10, 1, 1),
            nodata=255,
        ) as dst:
            prob = np.full((10, 10), 255, dtype=np.uint8)
            prob[2:5, 2:5] = 80
            prob[5:8, 5:8] = 20
            dst.write(prob, 1)

    def _get_sys_patch(
        self, mock_run, keras_mock=None, tf_mock=None, tifffile_mock=None
    ):
        return {
            "opticallyshallowdeep": MagicMock(),
            "opticallyshallowdeep.run": MagicMock(run=mock_run),
            "tensorflow": tf_mock or MagicMock(),
            "keras": keras_mock or MagicMock(),
            "tifffile": tifffile_mock or MagicMock(),
        }

    def test_classify_success(self):
        mock_run = MagicMock(side_effect=self._mock_side_effect)
        out_prob = Path(self.temp_dir.name) / "prob.tif"

        sys_patch = self._get_sys_patch(mock_run)
        with patch.dict(sys.modules, sys_patch):
            res = classify_optically_shallow_deep(
                image=self.input_dir,
                output=out_prob,
            )

        # Verify probability output exists, return value matches out_prob, and values are correct
        self.assertEqual(res, str(out_prob))
        self.assertTrue(out_prob.exists())
        with rasterio.open(out_prob) as src:
            data = src.read([1])[0]
            self.assertEqual(data[3, 3], 80)  # OSW water
            self.assertEqual(data[6, 6], 20)  # ODW water
            self.assertEqual(data[0, 0], 255)  # land (nodata)

    def test_missing_dependency(self):
        # Temporarily hide opticallyshallowdeep and all its submodules from sys.modules
        sys_modules_patch = {
            k: None
            for k in list(sys.modules.keys())
            if k == "opticallyshallowdeep" or k.startswith("opticallyshallowdeep.")
        }
        sys_modules_patch["opticallyshallowdeep"] = None

        with patch.dict(sys.modules, sys_modules_patch):
            with self.assertRaises(ImportError):
                classify_optically_shallow_deep(
                    image=self.input_dir,
                    output=Path(self.temp_dir.name) / "out.tif",
                )

    def test_classify_l2r_success(self):
        # Create dummy L2R file (.nc)
        dummy_l2r = (
            Path(self.temp_dir.name) / "S2A_MSI_2023_05_15_01_23_45_T54NVD_L2R.nc"
        )
        dummy_l2r.write_text("dummy netcdf content")

        mock_run = MagicMock(side_effect=self._mock_side_effect)
        out_prob = Path(self.temp_dir.name) / "prob_l2r.tif"

        sys_patch = self._get_sys_patch(mock_run)
        with patch.dict(sys.modules, sys_patch):
            res = classify_optically_shallow_deep(
                image=self.input_dir,
                output=out_prob,
                acolite_l2r=dummy_l2r,
            )
        self.assertEqual(res, str(out_prob))
        self.assertEqual(mock_run.call_args.kwargs.get("file_L2R"), str(dummy_l2r))
        self.assertTrue(out_prob.exists())

    def test_keras_patch_restoration(self):
        mock_keras = MagicMock()
        original_get = MagicMock(name="original_get")
        mock_keras.get = original_get

        mock_tf = MagicMock()
        mock_tf.keras = mock_keras
        original_device = MagicMock(name="original_device")
        mock_tf.device = original_device

        mock_tifffile = MagicMock()
        original_imread = MagicMock(name="original_imread")
        mock_tifffile.imread = original_imread

        mock_run = MagicMock(side_effect=self._mock_side_effect)
        out_prob = Path(self.temp_dir.name) / "prob_restored.tif"

        sys_patch = self._get_sys_patch(
            mock_run,
            keras_mock=mock_keras,
            tf_mock=mock_tf,
            tifffile_mock=mock_tifffile,
        )
        with patch.dict(sys.modules, sys_patch):
            classify_optically_shallow_deep(
                image=self.input_dir,
                output=out_prob,
            )
            self.assertIs(mock_keras.get, original_get)
            self.assertIs(mock_tf.device, original_device)
            self.assertIs(mock_tifffile.imread, original_imread)

    def test_osd_lock_used(self):
        from geoai.osd import _osd_lock

        self.assertTrue(hasattr(_osd_lock, "acquire"))
        self.assertTrue(hasattr(_osd_lock, "release"))

    def test_concurrent_invocations(self):
        import threading

        mock_keras = MagicMock()
        original_get = MagicMock(name="original_get")
        mock_keras.get = original_get

        mock_tf = MagicMock()
        mock_tf.keras = mock_keras
        original_device = MagicMock(name="original_device")
        mock_tf.device = original_device

        mock_tifffile = MagicMock()
        original_imread = MagicMock(name="original_imread")
        mock_tifffile.imread = original_imread

        mock_run = MagicMock(side_effect=self._mock_side_effect)

        out_prob_1 = Path(self.temp_dir.name) / "prob_concurrent_1.tif"
        out_prob_2 = Path(self.temp_dir.name) / "prob_concurrent_2.tif"

        sys_patch = self._get_sys_patch(
            mock_run,
            keras_mock=mock_keras,
            tf_mock=mock_tf,
            tifffile_mock=mock_tifffile,
        )
        errors = []

        def run_classify(out_p):
            try:
                classify_optically_shallow_deep(
                    image=self.input_dir,
                    output=out_p,
                )
            except Exception as e:
                errors.append(e)

        with patch.dict(sys.modules, sys_patch):
            t1 = threading.Thread(target=run_classify, args=(out_prob_1,))
            t2 = threading.Thread(target=run_classify, args=(out_prob_2,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")
        self.assertTrue(out_prob_1.exists())
        self.assertTrue(out_prob_2.exists())
        self.assertIs(mock_keras.get, original_get)
        self.assertIs(mock_tf.device, original_device)
        self.assertIs(mock_tifffile.imread, original_imread)


if __name__ == "__main__":
    unittest.main()
