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


def test_version_satisfies_rejects_prerelease_for_minimum_version():
    assert venv_manager._version_satisfies("0.39.0rc1", ">=0.39.0") is False
    assert venv_manager._version_satisfies("0.39.0", ">=0.39.0") is True


def test_version_satisfies_fails_closed_for_unsupported_specifier():
    assert venv_manager._version_satisfies("0.39.0", "~=0.39.0") is False


def test_pip_ssl_flags_trust_pytorch_wheel_host():
    flags = venv_manager._get_pip_ssl_flags()

    assert "--trusted-host" in flags
    assert "download.pytorch.org" in flags


def test_uv_ssl_flags_use_native_tls_and_allow_pytorch_wheel_host():
    flags = venv_manager._get_uv_ssl_flags()

    assert "--native-tls" in flags
    assert "--allow-insecure-host" in flags
    assert "download.pytorch.org" in flags
