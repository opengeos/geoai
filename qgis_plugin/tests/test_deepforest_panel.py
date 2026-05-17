from geoai.dialogs.deepforest_panel import _empty_detection_guidance


def test_empty_detection_guidance_points_single_image_to_large_tile():
    guidance = _empty_detection_guidance("Single Image")

    assert "Large Tile" in guidance
    assert "high-resolution" in guidance
    assert "score threshold" in guidance


def test_empty_detection_guidance_for_large_tile_mentions_tile_settings():
    guidance = _empty_detection_guidance("Large Tile")

    assert "patch overlap" in guidance
    assert "model" in guidance
