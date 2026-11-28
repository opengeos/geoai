"""Tests for scripts/convert_notebooks.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_convert_notebooks():
    script_path = REPO_ROOT / "scripts" / "convert_notebooks.py"
    spec = importlib.util.spec_from_file_location("convert_notebooks", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_badge_inserted_after_h1_heading(tmp_path: Path) -> None:
    """Badge is inserted immediately after the first H1 heading."""
    mod = _load_convert_notebooks()
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Demo Title\n\nIntro paragraph."),
            nbformat.v4.new_code_cell("print('hello')"),
        ]
    )
    nb_path = tmp_path / "demo.ipynb"
    nbformat.write(nb, nb_path)

    md = mod.notebook_to_markdown(nb_path)
    badge = mod.DOWNLOAD_TEMPLATE.format(name="demo.ipynb")
    assert md.startswith(f"# Demo Title\n\n{badge}\n\nIntro paragraph.")
    assert md.count(badge) == 1


def test_badge_prepended_when_notebook_has_no_h1_heading(tmp_path: Path) -> None:
    """Notebooks without an H1 heading still receive the download badge at the top."""
    mod = _load_convert_notebooks()
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("## Subtitle Only\n\nSome details."),
            nbformat.v4.new_code_cell("# comment inside code\nx = 1"),
        ]
    )
    nb_path = tmp_path / "no_h1.ipynb"
    nbformat.write(nb, nb_path)

    md = mod.notebook_to_markdown(nb_path)
    badge = mod.DOWNLOAD_TEMPLATE.format(name="no_h1.ipynb")
    assert md.startswith(f"{badge}\n\n## Subtitle Only")
    assert md.count(badge) == 1
    assert f"```python\n# comment inside code\nx = 1\n```" in md


def test_code_only_notebook_gets_prepended_badge(tmp_path: Path) -> None:
    """Code-only notebooks receive the download badge above the first code fence."""
    mod = _load_convert_notebooks()
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("# not a markdown heading\nimport geoai")]
    )
    nb_path = tmp_path / "code_only.ipynb"
    nbformat.write(nb, nb_path)

    md = mod.notebook_to_markdown(nb_path)
    badge = mod.DOWNLOAD_TEMPLATE.format(name="code_only.ipynb")
    assert md.startswith(f"{badge}\n\n```python\n# not a markdown heading")
    assert md.count(badge) == 1
