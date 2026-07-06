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

# Subdirectories of the docs directory that contain notebooks. Generated
# Markdown files are gitignored for exactly these subtrees, so keep this
# list in sync with .gitignore when adding a new notebook directory.
NOTEBOOK_DIRS = ("examples", "workshops")

DOWNLOAD_TEMPLATE = (
    "[![Jupyter Notebook](https://img.shields.io/badge/Open-Jupyter%20Notebook-orange?"
    "logo=jupyter)]({name})"
)


def notebook_to_markdown(path: Path) -> str:
    """Convert a Jupyter notebook to a Markdown document.

    Markdown cells are passed through verbatim, code cells are wrapped in
    fenced Python code blocks, and raw cells are skipped. A notebook
    download badge is inserted after the first Markdown heading. Only
    markdown cells are searched for the heading, so ``#`` comment lines in
    code cells can never attract the badge into a code fence.

    Args:
        path: Path to the ``.ipynb`` file to convert.

    Returns:
        The Markdown representation of the notebook.
    """
    nb = nbformat.read(path, as_version=4)
    badge = DOWNLOAD_TEMPLATE.format(name=path.name)
    blocks = []
    badge_inserted = False
    for cell in nb.cells:
        source = cell.source.rstrip()
        if not source:
            continue
        if cell.cell_type == "markdown":
            if not badge_inserted:
                source, badge_inserted = _insert_download_badge(source, badge)
            blocks.append(source)
        elif cell.cell_type == "code":
            blocks.append(f"```python\n{source}\n```")
    return "\n\n".join(blocks) + "\n"


def _insert_download_badge(markdown: str, badge: str) -> tuple:
    """Insert a notebook download badge after the first Markdown heading.

    Args:
        markdown: Source of a single markdown cell.
        badge: Badge markup linking to the source notebook, which is copied
            to the built site next to the page.

    Returns:
        A tuple of the (possibly modified) markdown and a bool indicating
        whether the badge was inserted.
    """
    match = re.search(r"^# .+$", markdown, flags=re.MULTILINE)
    if match is None:
        return markdown, False
    position = match.end()
    return f"{markdown[:position]}\n\n{badge}{markdown[position:]}", True


def convert_all(docs_dir: Path) -> int:
    """Convert all notebooks in the docs notebook directories to Markdown.

    Only the subtrees listed in ``NOTEBOOK_DIRS`` are scanned, matching the
    gitignore rules for the generated files. Files whose rendered content is
    unchanged are left untouched so their mtimes stay stable and Zensical's
    incremental builds stay fast.

    Args:
        docs_dir: Docs directory containing the notebook subdirectories.

    Returns:
        The number of notebooks converted.

    Raises:
        RuntimeError: If a notebook cannot be parsed or converted.
    """
    count = 0
    for subdir in NOTEBOOK_DIRS:
        for notebook in sorted((docs_dir / subdir).rglob("*.ipynb")):
            if ".ipynb_checkpoints" in notebook.parts:
                continue
            try:
                markdown = notebook_to_markdown(notebook)
            except Exception as exc:
                raise RuntimeError(f"Failed to convert {notebook}: {exc}") from exc
            target = notebook.with_suffix(".md")
            if not target.exists() or target.read_text(encoding="utf-8") != markdown:
                target.write_text(markdown, encoding="utf-8")
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
