"""Regression tests for Issue #1013 Section 5: CI, packaging, scripts, and docs."""

from __future__ import annotations

import ast
import base64
import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_nav_files(node, out: set[str]) -> None:
    if isinstance(node, str):
        if not node.startswith("http"):
            out.add(node)
    elif isinstance(node, list):
        for item in node:
            _collect_nav_files(item, out)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_nav_files(value, out)


# ===========================================================================
# GitHub Actions workflows and Dependabot
# ===========================================================================


def test_all_workflows_and_dependabot_parse_and_declare_permissions():
    workflow_files = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflow_files, "Expected GitHub Actions workflows"

    for wf_path in workflow_files:
        data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{wf_path.name} did not parse to a mapping"
        has_top_perm = "permissions" in data
        jobs = data.get("jobs", {})
        all_jobs_have_perm = bool(jobs) and all(
            isinstance(cfg, dict) and "permissions" in cfg for cfg in jobs.values()
        )
        assert (
            has_top_perm or all_jobs_have_perm
        ), f"{wf_path.name} must declare explicit permissions at workflow or job level"

    dependabot = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    pip_dirs = {
        entry["directory"]
        for entry in dependabot["updates"]
        if entry.get("package-ecosystem") == "pip"
    }
    assert {"/", "/geoai-mcp-server", "/agent-harness"}.issubset(pip_dirs)


def test_publish_workflow_checkout_ref_and_qgis_gate():
    publish = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(
            encoding="utf-8"
        )
    )
    assert publish["permissions"] == {"contents": "read"}
    job = publish["jobs"]["publish"]
    assert job["permissions"] == {"contents": "write"}

    checkout_steps = [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkout_steps) == 1
    checkout_with = checkout_steps[0].get("with", {})
    assert "inputs.tag" in str(checkout_with.get("ref", ""))
    assert checkout_with.get("fetch-depth") == 0

    upload_steps = [
        step
        for step in job["steps"]
        if step.get("name") == "Upload to plugins.qgis.org"
    ]
    assert len(upload_steps) == 1
    assert "if" in upload_steps[0], "plugins.qgis.org upload step must be gated"
    assert "should_upload" in str(upload_steps[0]["if"])


def test_pypi_workflow_triggers_on_published_and_pins_python():
    pypi_text = (REPO_ROOT / ".github" / "workflows" / "pypi.yml").read_text(
        encoding="utf-8"
    )
    pypi = yaml.safe_load(pypi_text)
    on_cfg = pypi.get("on", pypi.get(True))
    assert on_cfg["release"]["types"] == ["published"]
    assert pypi["permissions"] == {"contents": "read"}

    setup_steps = [
        step
        for step in pypi["jobs"]["deploy"]["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    assert len(setup_steps) == 1
    assert setup_steps[0]["with"]["python-version"] == "3.12"
    assert "gh-action-pypi-publish" in pypi_text


def test_draft_pdf_workflow_path_filter_and_permissions():
    draft = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "draft-pdf.yml").read_text(
            encoding="utf-8"
        )
    )
    on_cfg = draft.get("on", draft.get(True))
    assert draft["permissions"] == {"contents": "read"}
    assert "paper/**" in on_cfg["push"]["paths"]
    assert "paper/**" in on_cfg["pull_request"]["paths"]
    assert "workflow_dispatch" in on_cfg


def test_macos_workflow_python_matrix_and_smoke_tests():
    macos = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "macos.yml").read_text(encoding="utf-8")
    )
    assert macos["permissions"] == {"contents": "read"}
    job = macos["jobs"]["test-macOS"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.12", "3.13", "3.14"]
    combined_run = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "geoai --help" in combined_run
    assert "tests/test_lazy_import.py" in combined_run
    assert "tests/test_cli.py" in combined_run


def test_lint_workflow_exists_and_checks_configs_docs_and_ruff():
    lint_path = REPO_ROOT / ".github" / "workflows" / "lint.yml"
    assert lint_path.is_file()
    lint_text = lint_path.read_text(encoding="utf-8")
    lint = yaml.safe_load(lint_text)
    assert lint["permissions"] == {"contents": "read"}
    assert "tomllib" in lint_text
    assert "yaml.safe_load" in lint_text
    assert "ruff check --select E9,F63,F7" in lint_text


# ===========================================================================
# Packaging, dependencies, and Docker
# ===========================================================================


def test_pyproject_and_requirements_consistency():
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)

    build_requires = pyproject["build-system"]["requires"]
    assert all("setuptools_scm" not in req for req in build_requires)

    extra_deps = pyproject["project"]["optional-dependencies"]["extra"]
    extra_names = {
        re.split(r"[<>=!~;\[]", dep)[0].strip().lower() for dep in extra_deps
    }
    assert "overturemaps" not in extra_names
    assert "smoothify" not in extra_names
    assert "localtileserver" not in extra_names

    req_lines = (
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )
    active_reqs = [
        line.strip()
        for line in req_lines
        if line.strip() and not line.strip().startswith("#")
    ]
    from packaging.version import Version

    def _assert_min_floor(reqs, pkg_name, min_ver):
        matches = [
            line
            for line in reqs
            if re.match(rf"^{re.escape(pkg_name)}>=(.+)$", line, re.IGNORECASE)
        ]
        assert matches, f"Expected lower-bounded requirement for {pkg_name}"
        ver_str = matches[0].split(">=", 1)[1].split(",")[0].strip()
        assert Version(ver_str) >= Version(
            min_ver
        ), f"{pkg_name} floor {ver_str} is below minimum {min_ver}"

    _assert_min_floor(active_reqs, "aiohttp", "3.13.4")
    _assert_min_floor(active_reqs, "requests", "2.33.0")
    _assert_min_floor(active_reqs, "cryptography", "46.0.6")
    assert active_reqs == sorted(
        active_reqs, key=str.lower
    ), "requirements.txt must remain alphabetically sorted for pre-commit"


def test_requirements_docs_and_manifest_and_docker():
    from packaging.version import Version

    docs_reqs = {
        line.strip()
        for line in (REPO_ROOT / "requirements_docs.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in (
        "zensical",
        "mkdocstrings",
        "mkdocstrings-python",
        "pymdown-extensions",
        "nbformat",
    ):
        assert any(
            re.split(r"[<>=!~;\[]", r)[0].strip().lower() == required for r in docs_reqs
        ), f"Expected {required} in requirements_docs.txt"

    pygments_matches = [r for r in docs_reqs if r.lower().startswith("pygments>=")]
    assert pygments_matches, "Expected lower-bounded pygments in requirements_docs.txt"
    assert Version(
        pygments_matches[0].split(">=", 1)[1].split(",")[0].strip()
    ) >= Version("2.21.0")

    for legacy in (
        "sphinx",
        "mystmd",
        "tox",
        "flake8",
        "isort",
        "bump-my-version",
        "twine",
        "wheel",
        "coverage",
        "livereload",
        "watchdog",
        "pre-commit",
    ):
        assert legacy not in docs_reqs

    manifest_lines = {
        line.strip()
        for line in (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for pruned in (
        "prune qgis_plugin",
        "prune geoai-mcp-server",
        "prune agent-harness",
        "prune scripts",
        "prune tests",
    ):
        assert pruned in manifest_lines

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "quay.io/jupyter/base-notebook:latest" not in dockerfile
    assert re.search(
        r"FROM\s+quay\.io/jupyter/base-notebook:python-3\.\d+", dockerfile
    ), "Dockerfile must pin a Python 3.x base-notebook tag"
    assert "--no-install-recommends" in dockerfile
    assert "mamba clean --all -f -y" in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_GEOAI" not in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile

    dockerignore = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "!docs/notebooks" not in dockerignore
    assert "!docs/maplibre" not in dockerignore
    for entry in (
        ".env.*",
        "qgis_plugin/",
        "agent-harness/",
        "geoai-mcp-server/",
        "paper/",
    ):
        assert entry in dockerignore


# ===========================================================================
# Documentation and Zensical navigation
# ===========================================================================


def test_zensical_nav_and_docs_integrity():
    with open(REPO_ROOT / "zensical.toml", "rb") as fh:
        zen = tomllib.load(fh)

    nav_files: set[str] = set()
    _collect_nav_files(zen["project"]["nav"], nav_files)

    expected_api_pages = {
        "pipeline.md",
        "clip_classify.md",
        "dinov3_finetune.md",
        "agents.md",
        "orthogonalization.md",
        "universat.md",
        "vllm_geo.md",
    }
    for page in expected_api_pages:
        assert (REPO_ROOT / "docs" / page).is_file(), f"Missing docs/{page}"
        assert page in nav_files, f"docs/{page} missing from zensical.toml nav"

    ignored = {"index.md", "changelog.md"}
    for md in sorted((REPO_ROOT / "docs").glob("*.md")):
        if md.name not in ignored:
            assert md.name in nav_files, f"docs/{md.name} not listed in zensical.toml"

    for example in (
        "examples/optically_shallow_deep.md",
        "examples/esrgan.md",
        "examples/train_point_cloud_ptv3.md",
    ):
        assert example in nav_files

    for nav_path in nav_files:
        if nav_path.startswith("examples/") and nav_path.endswith(".md"):
            ipynb = (REPO_ROOT / "docs" / nav_path).with_suffix(".ipynb")
            assert ipynb.exists(), f"Missing notebook for nav entry {nav_path}"

    for md in sorted((REPO_ROOT / "docs").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        assert not re.search(
            r"examples/[^\s\)\"']+\.ipynb", text
        ), f"Relative notebook link found in {md.name}"

    for doc_path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "index.md"):
        text = doc_path.read_text(encoding="utf-8")
        assert not re.search(r"youtu\.be/[^\s\)\"']*&list=", text)

    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "Remote Sensing Foundation Models" in docs_index
    assert "k-fold segmentation training" in docs_index

    citations = (REPO_ROOT / "docs" / "citations.md").read_text(encoding="utf-8")
    assert "GeoAI with Python" in citations
    assert "https://book.opengeoai.org" in citations

    changelog = (REPO_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    assert "https://github.com/opengeos/geoai/releases" in changelog


# ===========================================================================
# Scripts
# ===========================================================================


def test_upload_qgis_plugin_uses_header_auth_and_timeout(tmp_path):
    script_path = REPO_ROOT / "scripts" / "upload_to_qgis_plugin_repo.py"
    source = script_path.read_text(encoding="utf-8")
    assert "{user}:{password}@" not in source

    mod = _load_module("upload_to_qgis_plugin_repo_test", script_path)
    transport = mod._AuthenticatedSafeTransport("alice", "s3cr3t", timeout=60)
    assert transport._timeout == 60
    expected_token = base64.b64encode(b"alice:s3cr3t").decode("ascii")
    assert transport._auth_header == f"Basic {expected_token}"

    conn = mock.MagicMock()
    transport.send_headers(conn, [("X-Test", "1")])
    conn.putheader.assert_any_call("Authorization", f"Basic {expected_token}")

    zip_file = tmp_path / "plugin.zip"
    zip_file.write_bytes(b"PK\x03\x04fake")

    fake_proxy = mock.MagicMock()
    fake_proxy.plugin.upload.return_value = (42, 7)

    with mock.patch.object(mod, "ServerProxy", return_value=fake_proxy) as proxy_cls:
        result = mod.upload(str(zip_file), "alice", "s3cr3t")

    assert result == (42, 7)
    endpoint_arg = proxy_cls.call_args.args[0]
    assert endpoint_arg == "https://plugins.qgis.org/plugins/RPC2/"
    assert "alice" not in endpoint_arg
    assert "s3cr3t" not in endpoint_arg
    assert isinstance(
        proxy_cls.call_args.kwargs["transport"], mod._AuthenticatedSafeTransport
    )


def test_train_ptv3_dales_fixes():
    script_path = REPO_ROOT / "scripts" / "train_ptv3_dales.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Verify --hf_pretrained does not overwrite args.resume
    assert "args.resume = hf_hub_download" not in source

    # Verify eval_only partitions evaluation across ranks without duplicate padding
    eval_block = source.split("# -- eval-only mode")[1].split("# -- training loop")[0]
    assert "Subset(ds, list(range(rank, len(ds), world_size)))" in eval_block

    # Verify _load_checkpoint tries weights_only=True first and falls back on error
    func_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_load_checkpoint" in func_names

    mod = _load_module("train_ptv3_dales_test", script_path)
    calls = []

    def fake_torch_load(path, map_location="cpu", weights_only=True):
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("legacy pickle object")
        return {"model_state_dict": {}}

    with mock.patch.object(mod.torch, "load", side_effect=fake_torch_load):
        ckpt = mod._load_checkpoint("fake.pth", map_location="cpu")

    assert ckpt == {"model_state_dict": {}}
    assert calls == [True, False]


def test_train_s2_water_uses_data_dir_and_tempdir(tmp_path):
    script_path = REPO_ROOT / "scripts" / "train_s2_water.py"
    source = script_path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "tempfile.gettempdir()" not in source
    assert "tempfile.TemporaryDirectory()" in source

    mod = _load_module("train_s2_water_test", script_path)

    target_dir = tmp_path / "custom_data"
    extracted = target_dir / "dset-s2"
    extracted.mkdir(parents=True)

    with mock.patch.object(
        mod.geoai, "download_file", return_value=str(extracted)
    ) as dl_mock:
        out_dir = mod.download_dataset(str(target_dir))

    assert out_dir == str(extracted)
    dl_mock.assert_called_once()
    output_path = dl_mock.call_args.kwargs.get("output_path")
    assert output_path is not None
    assert Path(output_path).parent == target_dir

    fake_api = mock.MagicMock()
    uploaded_paths = []

    def capture_upload(**kwargs):
        p = Path(kwargs["path_or_fileobj"])
        assert p.is_file()
        assert p.name == "README.md"
        assert p.parent != Path(mod.tempfile.gettempdir())
        uploaded_paths.append(p)

    fake_api.upload_file.side_effect = capture_upload

    fake_hf_module = mock.MagicMock()
    fake_hf_module.HfApi.return_value = fake_api

    with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
        url = mod.push_dataset_to_hub(str(extracted), repo_id="org/test-ds")

    assert url == "https://huggingface.co/datasets/org/test-ds"
    assert len(uploaded_paths) == 1
    assert not uploaded_paths[0].exists(), "Temporary directory should be cleaned up"
