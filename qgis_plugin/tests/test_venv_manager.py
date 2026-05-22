from geoai.core import venv_manager


def _write_dist_metadata(site_packages, dist_name, version):
    """Write minimal dist-info metadata for quick-check tests.

    Args:
        site_packages: Temporary site-packages path.
        dist_name: Distribution name to write.
        version: Distribution version to write.
    """
    dist_info = site_packages / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Name: {dist_name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_quick_check_rejects_stale_geoai_distribution(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    for package_dir in ("torch", "torchvision", "geoai"):
        (site_packages / package_dir).mkdir()
    _write_dist_metadata(site_packages, "geoai-py", "0.9.0")

    monkeypatch.setattr(
        venv_manager,
        "get_venv_site_packages",
        lambda _venv_dir=None: str(site_packages),
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("geoai-py", ">=0.39.0")],
    )

    ready, message = venv_manager._quick_check_packages(str(tmp_path / "venv"))

    assert ready is False
    assert "geoai-py 0.9.0 does not satisfy >=0.39.0" in message


def test_quick_check_accepts_current_geoai_distribution(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    for package_dir in ("torch", "torchvision", "geoai"):
        (site_packages / package_dir).mkdir()
    _write_dist_metadata(site_packages, "geoai-py", "0.39.0")

    monkeypatch.setattr(
        venv_manager,
        "get_venv_site_packages",
        lambda _venv_dir=None: str(site_packages),
    )
    monkeypatch.setattr(
        venv_manager,
        "_get_required_packages",
        lambda: [("geoai-py", ">=0.39.0")],
    )

    ready, message = venv_manager._quick_check_packages(str(tmp_path / "venv"))

    assert ready is True
    assert message == "All packages found"
