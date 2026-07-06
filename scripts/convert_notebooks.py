#!/usr/bin/env python
"""Convert Jupyter notebooks under docs/ to Markdown pages for Zensical.

Zensical does not (yet) support the mkdocs-jupyter plugin, so this script
converts every ``*.ipynb`` file under the docs directory into a sibling
Markdown file that Zensical can render. The generated Markdown files are
ignored by git (see .gitignore) and are regenerated on every docs build.

The notebooks in this repository are stored without outputs, so the
conversion is lossless: markdown cells are passed through verbatim and code
cells become fenced Python code blocks. The original notebook is copied to
the built site unchanged, so a download link is added below the first
heading of each page.

Usage:
    python scripts/convert_notebooks.py [--docs-dir docs]
"""

import argparse
import re
from pathlib import Path

import nbformat

DOWNLOAD_TEMPLATE = (
    "[![Jupyter Notebook](https://img.shields.io/badge/Open-Jupyter%20Notebook-orange?"
    "logo=jupyter)]({name})"
)


def notebook_to_markdown(path: Path) -> str:
    """Convert a Jupyter notebook to a Markdown document.

    Markdown cells are passed through verbatim, code cells are wrapped in
    fenced Python code blocks, and raw cells are skipped. A notebook
    download badge is inserted after the first Markdown heading.

    Args:
        path: Path to the ``.ipynb`` file to convert.

    Returns:
        The Markdown representation of the notebook.
    """
    nb = nbformat.read(path, as_version=4)
    blocks = []
    for cell in nb.cells:
        source = cell.source.rstrip()
        if not source:
            continue
        if cell.cell_type == "markdown":
            blocks.append(source)
        elif cell.cell_type == "code":
            blocks.append(f"```python\n{source}\n```")
    markdown = "\n\n".join(blocks) + "\n"
    return _insert_download_badge(markdown, path.name)


def _insert_download_badge(markdown: str, notebook_name: str) -> str:
    """Insert a notebook download badge after the first Markdown heading.

    Args:
        markdown: The Markdown document to modify.
        notebook_name: File name of the source notebook, used as a relative
            link target since the notebook is copied to the built site.

    Returns:
        The Markdown document with the badge inserted, or unchanged if no
        heading was found.
    """
    badge = DOWNLOAD_TEMPLATE.format(name=notebook_name)
    match = re.search(r"^# .+$", markdown, flags=re.MULTILINE)
    if match is None:
        return markdown
    position = match.end()
    return f"{markdown[:position]}\n\n{badge}{markdown[position:]}"


def convert_all(docs_dir: Path) -> int:
    """Convert all notebooks under a docs directory to Markdown files.

    Args:
        docs_dir: Root directory to search for ``.ipynb`` files.

    Returns:
        The number of notebooks converted.
    """
    count = 0
    for notebook in sorted(docs_dir.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in notebook.parts:
            continue
        target = notebook.with_suffix(".md")
        target.write_text(notebook_to_markdown(notebook), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    """Parse command-line arguments and run the conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs",
        help="Docs directory to search for notebooks (default: <repo>/docs).",
    )
    args = parser.parse_args()
    count = convert_all(args.docs_dir)
    print(f"Converted {count} notebooks to Markdown under {args.docs_dir}")


if __name__ == "__main__":
    main()
