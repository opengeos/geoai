"""Pytest configuration and fixtures for GeoAI tests."""

import pytest


@pytest.fixture
def sample_data_path(tmp_path):
    """Create a temporary directory for sample data."""
    data_path = tmp_path / "data"
    data_path.mkdir()
    return data_path


@pytest.fixture
def output_path(tmp_path):
    """Create a temporary directory for output."""
    output = tmp_path / "output"
    output.mkdir()
    return output
