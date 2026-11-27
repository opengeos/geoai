"""Regression tests for CI, packaging, preprocessing script, and documentation fixes."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_preprocess_dales_module(monkeypatch: pytest.MonkeyPatch, fake_las: object):
    """Load scripts/preprocess_dales_ptv3.py with a lightweight mock laspy module."""
    fake_laspy = ModuleType("laspy")
    fake_laspy.read = lambda _path: fake_las  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laspy", fake_laspy)

    script_path = REPO_ROOT / "scripts" / "preprocess_dales_ptv3.py"
    spec = importlib.util.spec_from_file_location(
        "preprocess_dales_ptv3_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preprocess_dales_preserves_utm_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_tile returns float64 and split_into_blocks preserves sub-meter UTM precision."""
    n_points = 120
    offsets = np.arange(n_points, dtype=np.float64) * 0.01
    x = 500_000.0 + offsets
    y = 5_000_000.0 + offsets
    z = 100.0 + offsets

    fake_las = SimpleNamespace(
        x=x,
        y=y,
        z=z,
        return_number=np.ones(n_points, dtype=np.uint8),
        classification=np.ones(n_points, dtype=np.uint8),
    )
    module = _load_preprocess_dales_module(monkeypatch, fake_las)

    coord, strength, segment = module.read_tile(Path("dummy.las"))
    assert coord.dtype == np.float64

    blocks = module.split_into_blocks(
        coord, strength, segment, block_size=50.0, min_points=100
    )
    assert len(blocks) == 1

    centred = blocks[0]["coord"]
    assert centred.dtype == np.float32

    expected_xyz = np.column_stack([x, y, z])
    expected_centred = (expected_xyz - expected_xyz.mean(axis=0)).astype(np.float32)
    np.testing.assert_allclose(centred, expected_centred, atol=1e-5)


def test_pyproject_has_no_universal_wheel_setting() -> None:
    """pyproject.toml is valid TOML and does not enable deprecated universal wheels."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.12"
    distutils_cfg = data.get("tool", {}).get("distutils", {})
    bdist_wheel_cfg = distutils_cfg.get("bdist_wheel", {})
    assert bdist_wheel_cfg.get("universal") is not True
    assert "bdist_wheel" not in distutils_cfg


def test_pre_commit_codespell_skip_list() -> None:
    """codespell --skip separates *.yml and *.js and uses *.cff."""
    config_path = REPO_ROOT / ".pre-commit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    skip_arg = None
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "codespell":
                for arg in hook.get("args", []):
                    if arg.startswith("--skip="):
                        skip_arg = arg
    assert skip_arg is not None

    patterns = skip_arg.removeprefix("--skip=").split(",")
    assert "*.yml" in patterns
    assert "*.js" in patterns
    assert "*.cff" in patterns
    assert "*.yml*.js" not in patterns
    assert "*cff" not in patterns


@pytest.mark.parametrize(
    "workflow_name",
    ["ubuntu.yml", "docs.yml", "docs-build.yml"],
)
def test_ci_workflows_install_local_extra(workflow_name: str) -> None:
    """CI workflows are valid YAML and install local '.[extra]' instead of PyPI geoai-py[extra]."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / workflow_name
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    run_scripts = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    ]
    combined = "\n".join(run_scripts)
    assert 'uv pip install "geoai-py[extra]"' not in combined
    assert "uv pip install geoai-py[extra]" not in combined
    assert 'uv pip install ".[extra]"' in combined


@pytest.mark.parametrize("rel_path", ["README.md", "docs/index.md"])
def test_no_broken_youtube_playlist_links(rel_path: str) -> None:
    """youtu.be playlist links use '?list=' instead of '&list='."""
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert not re.search(r"https?://youtu\.be/[^\s)]+&list=", text)
    assert re.search(r"https?://youtu\.be/[^\s)]+\?list=", text)


def test_docs_python_version_and_extras() -> None:
    """Installation and contributing docs specify Python 3.12+ and valid extras."""
    install_text = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    assert "Python 3.12 or above" in install_text
    assert "Python 3.10" not in install_text
    assert "geoai-py[all]" not in install_text
    assert 'pip install "geoai-py[extra]"' in install_text

    contributing_text = (REPO_ROOT / "docs" / "contributing.md").read_text(
        encoding="utf-8"
    )
    assert "Python 3.12 or above" in contributing_text
    assert "Python 3.10" not in contributing_text


def test_qgis_plugin_docs_paths_and_links() -> None:
    """QGIS plugin docs reference qgis_plugin/geoai, GitHub LICENSE, and canonical OmniWaterMask."""
    text = (REPO_ROOT / "docs" / "qgis_plugin.md").read_text(encoding="utf-8")
    assert "Copy the `qgis_plugin/geoai` folder" in text
    assert "[LICENSE](../LICENSE)" not in text
    assert "[LICENSE](https://github.com/opengeos/geoai/blob/main/LICENSE)" in text
    assert "https://github.com/NikolasEnt/OmniWaterMask" not in text
    assert "https://github.com/DPIRD-DMA/OmniWaterMask" in text


def test_no_relative_ipynb_links_in_top_level_docs() -> None:
    """Top-level docs/*.md link to generated examples/*.md pages backed by real notebooks."""
    docs_dir = REPO_ROOT / "docs"
    relative_ipynb_pattern = re.compile(r"\]\((examples/[^)\s]+\.ipynb)\)")
    relative_md_pattern = re.compile(r"\]\((examples/([^)\s]+)\.md)\)")

    offenders: list[str] = []
    missing_notebooks: list[str] = []

    for md_path in sorted(docs_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for match in relative_ipynb_pattern.finditer(text):
            offenders.append(f"{md_path.name}: {match.group(1)}")
        for match in relative_md_pattern.finditer(text):
            nb_path = docs_dir / "examples" / f"{match.group(2)}.ipynb"
            if not nb_path.exists():
                missing_notebooks.append(f"{md_path.name}: {match.group(1)}")

    assert not offenders, f"Relative .ipynb links remain: {offenders}"
    assert not missing_notebooks, f"Missing source notebooks: {missing_notebooks}"
