# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/opengeos/geoai/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with `enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

GeoAI could always use more documentation,
whether as part of the official GeoAI docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/opengeos/geoai/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up GeoAI for local development.

1.  Fork the GeoAI repo on GitHub.

2.  Clone your fork locally, using either SSH:

    ```shell
    git clone git@github.com:your_name_here/geoai.git
    ```

    Or HTTPS:

    ```shell
    git clone https://github.com/your_name_here/geoai.git
    ```

3.  Create a Python 3.12+ virtual environment and install the package in editable mode.

    Using [uv](https://docs.astral.sh/uv) (recommended):

    ```shell
    cd geoai/
    uv venv --python 3.12
    source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
    uv pip install -e ".[extra]"
    uv pip install pytest ruff pre-commit
    ```

    Or using standard `venv` and `pip`:

    ```shell
    cd geoai/
    python -m venv venv
    source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install pytest ruff pre-commit
    ```

4.  Create a branch for local development:

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  Run tests and quality checks before committing (see [Testing and Code Quality](#testing-and-code-quality) below).

6.  Commit your changes and push your branch to GitHub:

    ```shell
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

## Testing and Code Quality

### Running Tests

Run the test suite with `pytest`:

```shell
pytest tests
```

Or target a specific test file while iterating:

```shell
pytest tests/test_cli.py -v
```

### Linting with Ruff

CI runs Ruff critical syntax and name checks on `geoai/` and `scripts/`:

```shell
ruff check --select E9,F63,F7 geoai/ scripts/
```

### Pre-commit Hooks

This repository configures `.pre-commit-config.yaml` with `black-jupyter`, `codespell`, `nbstripout`, `bandit` (for `qgis_plugin/geoai`), `check-toml`, `check-yaml`, `requirements-txt-fixer`, and `trailing-whitespace`. Install and run the hooks locally:

```shell
pre-commit install
pre-commit run --all-files
```

Note that `requirements.txt` must stay alphabetically sorted (enforced by `requirements-txt-fixer`).

## Previewing Documentation Locally

The documentation site is built with [Zensical](https://zensical.org) (`zensical.toml`), and notebooks under `docs/examples/` and `docs/workshops/` are converted to Markdown automatically at build time via `scripts/convert_notebooks.py`.

1.  Install documentation dependencies:

    ```shell
    pip install -r requirements_docs.txt
    ```

2.  Convert Jupyter notebooks to Markdown (generated `.md` files in `docs/examples/` and `docs/workshops/` are gitignored and should not be committed):

    ```shell
    python scripts/convert_notebooks.py
    ```

3.  Serve the docs locally with live reload or run a full static build:

    ```shell
    zensical serve
    # or run a full production build into ./site
    zensical build
    ```

    If you edit any `.ipynb` notebook while `zensical serve` is running, re-run `python scripts/convert_notebooks.py` so the updated Markdown is regenerated (Zensical does not automatically invoke the notebook converter on rebuild).

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in `README.md` and `docs/index.md`.
3.  The pull request should work for Python 3.12 or above. Check <https://github.com/opengeos/geoai/pulls> and make sure that the tests pass for all
    supported Python versions.

## Seek Help

If you have any questions, please ask in our [GitHub Discussions](https://github.com/opengeos/geoai/discussions).
