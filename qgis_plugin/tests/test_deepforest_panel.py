from geoai.dialogs.deepforest_panel import (
    _empty_detection_guidance,
    _is_large_image,
    _large_image_recommendation,
    _raster_layer_dimensions,
)


def test_empty_detection_guidance_points_single_image_to_large_tile():
    guidance = _empty_detection_guidance("Single Image")

    assert "Large Tile" in guidance
    assert "high-resolution" in guidance
    assert "score threshold" in guidance


def test_empty_detection_guidance_for_large_tile_mentions_tile_settings():
    guidance = _empty_detection_guidance("Large Tile")

    assert "patch overlap" in guidance
    assert "model" in guidance


def test_is_large_image_uses_side_and_total_pixel_thresholds():
    assert _is_large_image(4096, 1000)
    assert _is_large_image(4000, 4000)
    assert not _is_large_image(1024, 1024)


def test_large_image_recommendation_points_single_image_to_large_tile():
    message = _large_image_recommendation((5000, 3000), "Single Image")

    assert "Large Tile" in message
    assert "high-resolution" in message


def test_large_image_recommendation_confirms_large_tile_mode():
    message = _large_image_recommendation((5000, 3000), "Large Tile")

    assert "Large Tile mode is active" in message


def test_large_image_recommendation_ignores_small_images():
    assert _large_image_recommendation((1024, 1024), "Single Image") is None


class _FakeProvider:
    def xSize(self):
        return 5000

    def ySize(self):
        return 3000


class _FakeLayer:
    def dataProvider(self):
        return _FakeProvider()


def test_raster_layer_dimensions_uses_provider_size():
    assert _raster_layer_dimensions(_FakeLayer()) == (5000, 3000)
