"""Tests for instance segmentation inference output and vectorization."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from geoai.object_detect import detections_to_geodataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_detections():
    """Create sample detections matching the compact-mask format."""
    mask1 = np.zeros((64, 64), dtype=bool)
    mask1[10:30, 10:30] = True

    mask2 = np.zeros((64, 64), dtype=bool)
    mask2[20:50, 25:55] = True

    return [
        {
            "mask": mask1,
            "mask_offset": (0, 0, 64, 64),
            "score": 0.95,
            "box": [10.0, 10.0, 30.0, 30.0],
            "label": 1,
            "instance_id": 1,
        },
        {
            "mask": mask2,
            "mask_offset": (100, 100, 64, 64),
            "score": 0.80,
            "box": [125.0, 120.0, 155.0, 150.0],
            "label": 2,
            "instance_id": 2,
        },
    ]


@pytest.fixture
def full_mask_detections():
    """Create detections with full-image masks (no mask_offset)."""
    mask1 = np.zeros((256, 256), dtype=bool)
    mask1[10:30, 10:30] = True

    mask2 = np.zeros((256, 256), dtype=bool)
    mask2[120:150, 125:155] = True

    return [
        {
            "mask": mask1,
            "score": 0.95,
            "box": [10.0, 10.0, 30.0, 30.0],
            "label": 1,
            "instance_id": 1,
        },
        {
            "mask": mask2,
            "score": 0.80,
            "box": [125.0, 120.0, 155.0, 150.0],
            "label": 2,
            "instance_id": 2,
        },
    ]


@pytest.fixture
def mock_geotiff(tmp_path):
    """Create a small GeoTIFF for transform/CRS extraction."""
    import rasterio
    from rasterio.transform import from_bounds

    path = str(tmp_path / "test.tif")
    transform = from_bounds(0, 0, 256, 256, 256, 256)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs="EPSG:32617",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 256, 256), dtype=np.uint8))
    return path


# ---------------------------------------------------------------------------
# detections_to_geodataframe: return structure
# ---------------------------------------------------------------------------


class TestDetectionsToGeoDataFrame:
    """Test detections_to_geodataframe output columns and attributes."""

    def test_empty_detections(self, mock_geotiff):
        """Empty detection list returns empty GeoDataFrame with all columns."""
        gdf = detections_to_geodataframe([], mock_geotiff)
        expected_cols = {
            "geometry",
            "class_id",
            "class_name",
            "score",
            "instance_id",
            "area_pixels",
        }
        assert expected_cols == set(gdf.columns)
        assert len(gdf) == 0

    def test_columns_present(self, sample_detections, mock_geotiff):
        """GeoDataFrame has all required columns."""
        gdf = detections_to_geodataframe(sample_detections, mock_geotiff)
        expected_cols = {
            "geometry",
            "class_id",
            "class_name",
            "score",
            "instance_id",
            "area_pixels",
        }
        assert expected_cols == set(gdf.columns)
        assert len(gdf) == 2

    def test_instance_id_preserved(self, sample_detections, mock_geotiff):
        """instance_id values from detections are preserved in the output."""
        gdf = detections_to_geodataframe(sample_detections, mock_geotiff)
        assert list(gdf["instance_id"]) == [1, 2]

    def test_class_id_preserved(self, sample_detections, mock_geotiff):
        """class_id values from detections are preserved in the output."""
        gdf = detections_to_geodataframe(sample_detections, mock_geotiff)
        assert list(gdf["class_id"]) == [1, 2]

    def test_score_preserved(self, sample_detections, mock_geotiff):
        """Score values from detections are preserved in the output."""
        gdf = detections_to_geodataframe(sample_detections, mock_geotiff)
        assert list(gdf["score"]) == [0.95, 0.80]

    def test_label_defaults_to_one(self, mock_geotiff):
        """Detections without a label key default to class_id=1."""
        det = [
            {
                "mask": np.ones((10, 10), dtype=bool),
                "score": 0.9,
                "box": [0.0, 0.0, 10.0, 10.0],
            }
        ]
        gdf = detections_to_geodataframe(det, mock_geotiff)
        assert gdf.iloc[0]["class_id"] == 1

    def test_instance_id_defaults_sequential(self, mock_geotiff):
        """Detections without instance_id get sequential IDs starting at 1."""
        dets = [
            {
                "mask": np.ones((10, 10), dtype=bool),
                "score": 0.9,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "mask": np.ones((10, 10), dtype=bool),
                "score": 0.8,
                "box": [20.0, 20.0, 30.0, 30.0],
            },
        ]
        gdf = detections_to_geodataframe(dets, mock_geotiff)
        assert list(gdf["instance_id"]) == [1, 2]

    def test_class_names_mapping(self, sample_detections, mock_geotiff):
        """class_names list is used to populate class_name column."""
        names = ["background", "building", "tree"]
        gdf = detections_to_geodataframe(
            sample_detections, mock_geotiff, class_names=names
        )
        assert list(gdf["class_name"]) == ["building", "tree"]


# ---------------------------------------------------------------------------
# detections_to_geodataframe: mask geometry
# ---------------------------------------------------------------------------


class TestMaskGeometry:
    """Test use_mask_geometry option for polygon vectorization."""

    def test_bbox_geometry_default(self, sample_detections, mock_geotiff):
        """Default geometry is a bounding box rectangle (4 corners)."""
        gdf = detections_to_geodataframe(
            sample_detections, mock_geotiff, use_mask_geometry=False
        )
        for geom in gdf.geometry:
            assert geom.is_valid
            assert geom.geom_type == "Polygon"

    def test_mask_geometry_with_compact_mask(self, sample_detections, mock_geotiff):
        """use_mask_geometry=True with compact masks produces valid polygons."""
        gdf = detections_to_geodataframe(
            sample_detections, mock_geotiff, use_mask_geometry=True
        )
        assert len(gdf) == 2
        for geom in gdf.geometry:
            assert geom.is_valid
            assert geom.geom_type in ("Polygon", "MultiPolygon")

    def test_mask_geometry_with_full_mask(self, full_mask_detections, mock_geotiff):
        """use_mask_geometry=True with full-image masks uses bbox crop."""
        gdf = detections_to_geodataframe(
            full_mask_detections, mock_geotiff, use_mask_geometry=True
        )
        assert len(gdf) == 2
        for geom in gdf.geometry:
            assert geom.is_valid

    def test_simplify_preserves_topology(self, sample_detections, mock_geotiff):
        """Simplification preserves valid topology."""
        gdf = detections_to_geodataframe(
            sample_detections,
            mock_geotiff,
            use_mask_geometry=True,
            simplify_tolerance=2.0,
        )
        for geom in gdf.geometry:
            assert geom.is_valid

    def test_multicomponent_mask_unions(self, mock_geotiff):
        """A mask with disconnected components produces a valid geometry."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:15, 5:15] = True
        mask[40:55, 40:55] = True  # disconnected component

        det = [
            {
                "mask": mask,
                "score": 0.9,
                "box": [0.0, 0.0, 64.0, 64.0],
                "label": 1,
                "instance_id": 1,
            }
        ]
        gdf = detections_to_geodataframe(det, mock_geotiff, use_mask_geometry=True)
        geom = gdf.iloc[0].geometry
        assert geom.is_valid
        # Two disconnected components should produce MultiPolygon
        assert geom.geom_type == "MultiPolygon"


# ---------------------------------------------------------------------------
# instance_segmentation_inference_on_geotiff: return structure
# ---------------------------------------------------------------------------


class TestInferenceReturnStructure:
    """Test the return dict and detection format from the inference function."""

    @pytest.fixture
    def mock_inference_result(self, tmp_path, mock_geotiff):
        """Run inference with a mock model and return the result."""
        import torch

        # Create a mock model that returns fixed detections
        def mock_forward(images):
            results = []
            for _ in images:
                mask = torch.zeros((1, 1, 512, 512))
                mask[:, :, 100:200, 100:200] = 1.0
                results.append(
                    {
                        "masks": mask,
                        "scores": torch.tensor([0.9]),
                        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
                        "labels": torch.tensor([1]),
                    }
                )
            return results

        model = MagicMock()
        model.eval.return_value = None
        model.to.return_value = model
        model.side_effect = mock_forward
        model.__call__ = mock_forward

        output_path = str(tmp_path / "instances.tif")

        from geoai.train import instance_segmentation_inference_on_geotiff

        result = instance_segmentation_inference_on_geotiff(
            model=model,
            geotiff_path=mock_geotiff,
            output_path=output_path,
            window_size=256,
            overlap=64,
            confidence_threshold=0.5,
            device=torch.device("cpu"),
        )
        return result

    def test_returns_three_tuple(self, mock_inference_result):
        """Function returns a 3-tuple."""
        assert len(mock_inference_result) == 3

    def test_output_paths_dict_keys(self, mock_inference_result):
        """First element is a dict with instance/class_label/score keys."""
        output_paths, _, _ = mock_inference_result
        assert isinstance(output_paths, dict)
        assert set(output_paths.keys()) == {"instance", "class_label", "score"}

    def test_output_files_exist(self, mock_inference_result):
        """All output raster files exist on disk."""
        output_paths, _, _ = mock_inference_result
        for key, path in output_paths.items():
            assert os.path.exists(path), f"{key} raster not found at {path}"

    def test_output_file_suffixes(self, mock_inference_result):
        """Class and score rasters have correct suffixes."""
        output_paths, _, _ = mock_inference_result
        assert output_paths["class_label"].endswith("_class.tif")
        assert output_paths["score"].endswith("_score.tif")

    def test_detections_have_compact_masks(self, mock_inference_result):
        """Detections keep compact masks with mask_offset (not full-image)."""
        _, _, detections = mock_inference_result
        if len(detections) > 0:
            det = detections[0]
            assert "mask" in det
            assert "mask_offset" in det
            assert "score" in det
            assert "box" in det
            assert "label" in det
            assert "instance_id" in det

    def test_inference_time_positive(self, mock_inference_result):
        """Inference time is a positive float."""
        _, inference_time, _ = mock_inference_result
        assert isinstance(inference_time, float)
        assert inference_time >= 0


# ---------------------------------------------------------------------------
# Multiclass ObjectDetectionDataset
# ---------------------------------------------------------------------------


class TestObjectDetectionDatasetMulticlass:
    """Ensure ObjectDetectionDataset reads multi-class labels from masks."""

    def _write_pair(self, tmp_path, image_arr, label_arr, name):
        """Write an image and its label mask as single-band GeoTIFFs."""
        rasterio = pytest.importorskip("rasterio")
        img_path = tmp_path / f"{name}_img.tif"
        lbl_path = tmp_path / f"{name}_lbl.tif"
        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=image_arr.shape[1],
            width=image_arr.shape[2],
            count=image_arr.shape[0],
            dtype=image_arr.dtype,
        ) as dst:
            dst.write(image_arr)
        with rasterio.open(
            lbl_path,
            "w",
            driver="GTiff",
            height=label_arr.shape[0],
            width=label_arr.shape[1],
            count=1,
            dtype=label_arr.dtype,
        ) as dst:
            dst.write(label_arr, 1)
        return str(img_path), str(lbl_path)

    def test_multiclass_reads_class_from_pixel_values(self, tmp_path):
        """Each connected component is tagged with its mask pixel value."""
        pytest.importorskip("torch")
        from geoai.train import ObjectDetectionDataset

        image = np.ones((3, 64, 64), dtype=np.uint8) * 128
        label = np.zeros((64, 64), dtype=np.uint8)
        # Two disjoint objects with different class IDs
        label[5:20, 5:20] = 3  # class 3
        label[30:50, 30:55] = 5  # class 5

        img, lbl = self._write_pair(tmp_path, image, label, "mc")
        ds = ObjectDetectionDataset([img], [lbl], multiclass=True)
        _, target = ds[0]

        labels = target["labels"].tolist()
        assert sorted(labels) == [3, 5]
        assert target["boxes"].shape[0] == 2
        assert target["masks"].shape[0] == 2

    def test_multiclass_per_class_connected_components(self, tmp_path):
        """Touching regions of different classes stay as separate instances."""
        pytest.importorskip("torch")
        from geoai.train import ObjectDetectionDataset

        image = np.ones((3, 64, 64), dtype=np.uint8) * 128
        label = np.zeros((64, 64), dtype=np.uint8)
        # Two rectangles sharing an edge: a binary CC would merge them,
        # but per-class CC must keep them as two separate instances.
        label[10:40, 10:30] = 2  # class 2
        label[10:40, 30:50] = 4  # class 4

        img, lbl = self._write_pair(tmp_path, image, label, "touch")
        ds = ObjectDetectionDataset([img], [lbl], multiclass=True)
        _, target = ds[0]

        labels = sorted(target["labels"].tolist())
        assert labels == [2, 4]
        assert target["masks"].shape[0] == 2

    def test_multiclass_and_instance_labels_rejected(self):
        """Combining multiclass=True with instance_labels=True is a user error."""
        pytest.importorskip("torch")
        from geoai.train import ObjectDetectionDataset

        with pytest.raises(ValueError, match="instance_labels"):
            ObjectDetectionDataset(
                ["x.tif"], ["y.tif"], multiclass=True, instance_labels=True
            )

    def test_binary_default_still_assigns_one(self, tmp_path):
        """Backward compatible: without multiclass, labels are all 1."""
        pytest.importorskip("torch")
        from geoai.train import ObjectDetectionDataset

        image = np.ones((3, 64, 64), dtype=np.uint8) * 128
        label = np.zeros((64, 64), dtype=np.uint8)
        label[5:20, 5:20] = 3
        label[30:50, 30:55] = 5

        img, lbl = self._write_pair(tmp_path, image, label, "bin")
        ds = ObjectDetectionDataset([img], [lbl], multiclass=False)
        _, target = ds[0]

        assert target["labels"].tolist() == [1, 1]
