#!/usr/bin/env python

"""Tests for the Google/AlphaEarth embedding download metadata.

These are offline: the year guard runs before any network access, so the
available-year set can be checked without touching Source Cooperative.
"""

import pytest

from geoai.embeddings import (
    _AEF_AVAILABLE_YEARS,
    _AEF_LATEST_YEAR,
    EMBEDDING_DATASETS,
    download_google_satellite_embedding,
)


def test_available_years_cover_full_published_range():
    """The index publishes 2017-2025; 2017 and 2025 were both missing."""
    assert _AEF_AVAILABLE_YEARS == list(range(2017, 2026))


def test_available_years_include_2017():
    """Regression test for issue #849."""
    assert 2017 in _AEF_AVAILABLE_YEARS


def test_latest_year_tracks_available_years():
    assert _AEF_LATEST_YEAR == max(_AEF_AVAILABLE_YEARS)


def test_available_years_match_catalog_temporal_extent():
    """The registry and the download guard must not drift apart."""
    extent = EMBEDDING_DATASETS["google_satellite"]["temporal_extent"]
    start, end = (int(part) for part in extent.split("-"))
    assert [start, end] == [_AEF_AVAILABLE_YEARS[0], _AEF_AVAILABLE_YEARS[-1]]


@pytest.mark.parametrize("year", [2016, 2026])
def test_unavailable_year_is_rejected(year):
    """Years outside the published range still fail fast, before any download."""
    with pytest.raises(ValueError, match="not available"):
        download_google_satellite_embedding(bbox=(0.0, 0.0, 0.01, 0.01), years=year)
