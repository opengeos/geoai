"""Tests for Segment Anything outputs: scores, context fields, tiles and HF login."""

import json
import os
import sys
import types

import pytest

from geoai.core import samgeo_outputs as so
from geoai.core.samgeo_subprocess import SamGeoSubprocessClient
from geoai.workers import geoai_task_worker, samgeo_worker

TOKEN = "hf_SuperSecretToken1234567890"


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------


def test_default_output_path_next_to_image(tmp_path):
    image = tmp_path / "scene.tif"
    image.write_bytes(b"x")

    path = so.default_output_path(str(image), ".gpkg")

    assert path == str(tmp_path / "scene_sam_masks.gpkg")


def test_default_output_path_does_not_overwrite(tmp_path):
    image = tmp_path / "scene.tif"
    image.write_bytes(b"x")
    (tmp_path / "scene_sam_masks.geojson").write_text("{}")
    (tmp_path / "scene_sam_masks_1.geojson").write_text("{}")

    path = so.default_output_path(str(image), ".geojson")

    assert path == str(tmp_path / "scene_sam_masks_2.geojson")


def test_default_output_path_none_when_dir_not_writable(tmp_path):
    image = tmp_path / "scene.tif"
    image.write_bytes(b"x")

    path = so.default_output_path(
        str(image), ".tif", is_writable_dir=lambda _directory: False
    )

    assert path is None


@pytest.mark.parametrize(
    "source",
    [None, "", "/vsicurl/https://example.com/a.tif", "relative/a.tif"],
)
def test_default_output_path_none_for_non_local(source):
    assert so.default_output_path(source, ".tif") is None


def test_default_output_path_from_geopackage_source(tmp_path):
    gpkg = tmp_path / "ortho.gpkg"
    gpkg.write_bytes(b"x")

    for source in (f"GPKG:{gpkg}:tiles", f"{gpkg}|layername=tiles"):
        path = so.default_output_path(source, ".shp")
        assert path == str(tmp_path / "ortho_sam_masks.shp")


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "GPKG:/data/archive.gpkg.backup/scene.gpkg:tiles",
            "/data/archive.gpkg.backup/scene.gpkg",
        ),
        ("GPKG:/data/scene.gpkg:tiles", "/data/scene.gpkg"),
        ("GPKG:/data/scene.gpkg", "/data/scene.gpkg"),
        ("GPKG:C:/data/Scene.GPKG:tiles", "C:/data/Scene.GPKG"),
        ("/data/scene.gpkg|layername=tiles", "/data/scene.gpkg"),
        ("/data/image.tif", "/data/image.tif"),
    ],
)
def test_source_file_path_variants(source, expected):
    assert so.source_file_path(source) == expected


def test_unique_path_with_fake_exists():
    taken = {"/d/a.tif", "/d/a_1.tif"}

    assert so.unique_path("/d/a.tif", exists=taken.__contains__) == "/d/a_2.tif"
    assert so.unique_path("/d/b.tif", exists=taken.__contains__) == "/d/b.tif"


def test_suggested_output_name():
    assert so.suggested_output_name("/x/y/img.tiff", ".gpkg") == "img_sam_masks.gpkg"
    assert so.suggested_output_name(None, ".tif") == "sam_masks.tif"


@pytest.mark.parametrize(
    "label, ext",
    [
        ("Raster (GeoTIFF)", ".tif"),
        ("Vector (GeoPackage)", ".gpkg"),
        ("Vector (Shapefile)", ".shp"),
        ("Vector (GeoJSON)", ".geojson"),
    ],
)
def test_output_extension(label, ext):
    assert so.output_extension(label) == ext


def test_temp_output_path_shapefile_in_own_folder():
    path = so.temp_output_path(".shp")
    try:
        assert path.endswith("masks.shp")
        assert os.path.isdir(os.path.dirname(path))
        assert not os.path.exists(path)
    finally:
        os.rmdir(os.path.dirname(path))


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------


def test_field_name_truncated_for_shapefile():
    assert so.field_name("confidence_score", True) == "confidence"
    assert len(so.field_name("confidence_score", True)) == 10
    assert so.field_name("confidence_score", False) == "confidence_score"
    for name in ("score", "source", "prompt"):
        assert so.field_name(name, True) == name


@pytest.mark.parametrize(
    "fields, expected",
    [
        (["fid", "class"], "class"),  # raster_to_vector / smooth_vector
        (["value", "area_m2"], "value"),  # orthogonalize
        (["CLASS"], "CLASS"),
        (["fid", "name"], None),
    ],
)
def test_find_mask_id_field(fields, expected):
    assert so.find_mask_id_field(fields) == expected


def test_normalize_mask_scores_drops_non_finite():
    scores = so.normalize_mask_scores(
        {"1": float("nan"), "2": float("inf"), "3": float("-inf"), "4": 0.25}
    )

    assert scores == {4: 0.25}


def test_clip_shapefile_text_by_utf8_bytes():
    text = "é" * 200  # 400 UTF-8 bytes
    clipped = so._clip(text, True)

    assert len(clipped.encode("utf-8")) <= so.SHAPEFILE_STRING_MAX
    assert clipped == "é" * 127  # 254 bytes, no partial character
    assert so._clip("a" + "é" * 200, True) == "a" + "é" * 126  # drops half a char
    assert so._clip(text, False) == text
    assert so._clip("short", True) == "short"


def test_count_raster_labels_binary_is_none():
    assert so.count_raster_labels("missing.tif", unique=False) is None


def test_count_raster_labels_unreadable_is_none(tmp_path):
    assert so.count_raster_labels(str(tmp_path / "missing.tif"), True) is None


def test_normalize_mask_scores_handles_json_keys():
    scores = so.normalize_mask_scores({"1": 0.9, "2": "0.5", "x": 1, "3": None})

    assert scores == {1: 0.9, 2: 0.5}
    assert so.normalize_mask_scores(None) == {}


@pytest.mark.parametrize(
    "value, expected", [(3, 3), (3.0, 3), ("2", 2), (None, None), ("abc", None)]
)
def test_mask_id_from_value(value, expected):
    assert so.mask_id_from_value(value) == expected


def test_format_prompt_label():
    assert so.format_prompt_label("text", " tree ") == "text: tree"
    assert so.format_prompt_label("point") == "point"
    assert so.format_prompt_label("batch points", "") == "batch points"


# ---------------------------------------------------------------------------
# Hugging Face error help and redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error.\n\nCannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json.",
        "Access to model facebook/sam3 is restricted. You must have access to it and be authenticated to access it. Please log in.",
        "You are trying to access a gated repo.\nMake sure to have access to it at https://huggingface.co/facebook/sam3.",
        "Repository Not Found for url: https://huggingface.co/api/models/facebook/sam3",
    ],
)
def test_hf_access_help_for_gated_errors(message):
    help_text = so.hf_access_help(message)

    assert help_text is not None
    assert so.HF_SAM3_URL in help_text
    assert "Hugging Face field" in help_text


@pytest.mark.parametrize(
    "message",
    [
        "Torch not compiled with CUDA enabled",
        "403 Client Error: Forbidden for url: http://proxy.corp.local/check",
        "401 Client Error: Unauthorized for url: https://download.pytorch.org/x",
        "Repository not found: https://github.com/example/repo",
        "Unauthorized",
    ],
)
def test_hf_access_help_ignores_other_errors(message):
    assert so.hf_access_help(message) is None


def test_redact():
    assert so.redact(f"bad token {TOKEN}!", TOKEN) == "bad token ***!"
    assert so.redact("nothing here", "") == "nothing here"
    assert so.redact(None, TOKEN) == ""


# ---------------------------------------------------------------------------
# SamGeo worker: scores and tiles
# ---------------------------------------------------------------------------


class _NumpyLikeScalar:
    """Stand-in for a numpy scalar that exposes ``tolist()``.

    Keeps these tests free of a numpy dependency (the plugin CI job installs
    only PyQt6 and pytest) while exercising the same ``tolist()`` path the
    worker's JSON conversion takes for real numpy scores.
    """

    def __init__(self, value):
        """Store the wrapped value.

        Args:
            value: The Python float to return from ``tolist()``.
        """
        self._value = value

    def tolist(self):
        """Return the plain Python value, like numpy's ``tolist()``."""
        return self._value

    def __float__(self):
        """Convert to a Python float, like ``float(np.float32(...))``."""
        return float(self._value)


class FakeSam:
    """Minimal stand-in for SamGeo3."""

    def __init__(self, scores_attr=True):
        self.masks = [1, 2, 3]
        self.saved = None
        self.tiled = None
        self.objects = None
        if scores_attr:
            self.mask_scores = None

    def save_masks(self, output, unique=True):
        self.saved = (output, unique)
        if hasattr(self, "mask_scores"):
            self.mask_scores = (
                {1: _NumpyLikeScalar(0.875), 2: 0.5, 3: _NumpyLikeScalar(0.25)}
                if unique
                else None
            )

    def generate_masks_tiled(self, **kwargs):
        self.tiled = kwargs
        self.masks = None
        self.objects = [[0, 1, 1], [2, 0, 5]]
        return kwargs["output"]


@pytest.fixture
def fake_sam(monkeypatch):
    sam = FakeSam()
    monkeypatch.setitem(samgeo_worker._STATE, "sam", sam)
    return sam


def test_save_masks_returns_json_safe_scores(fake_sam):
    result = samgeo_worker._dispatch(
        {"action": "save_masks", "output": "out.tif", "unique": True}
    )
    safe = samgeo_worker._json_safe(result)

    assert fake_sam.saved == ("out.tif", True)
    assert safe["mask_count"] == 3
    assert safe["mask_scores"] == {"1": 0.875, "2": 0.5, "3": 0.25}
    assert json.loads(json.dumps(safe)) == safe


def test_save_masks_no_scores_when_not_unique(fake_sam):
    result = samgeo_worker._handle_save_masks({"output": "out.tif", "unique": False})

    assert result["mask_scores"] is None


def test_save_masks_without_mask_scores_attribute(monkeypatch):
    """Older samgeo has no mask_scores attribute."""
    monkeypatch.setitem(samgeo_worker._STATE, "sam", FakeSam(scores_attr=False))

    result = samgeo_worker._handle_save_masks({"output": "out.tif", "unique": True})

    assert result["mask_scores"] is None


def test_generate_masks_tiled_handler(fake_sam, monkeypatch):
    # Objects are counted from the written raster, one block at a time.
    blocks = [[[0, 1, 1], [2, 0, 5]], [[5, 7], [0, 0]]]
    seen = []

    def _blocks(path):
        seen.append(path)
        return iter(blocks)

    monkeypatch.setattr(samgeo_worker, "_iter_raster_blocks", _blocks)
    result = samgeo_worker._dispatch(
        {
            "action": "generate_masks_tiled",
            "source": "big.tif",
            "prompt": "tree",
            "output": "tiles.tif",
            "tile_size": 512,
            "overlap": 64,
            "min_size": 10,
            "max_size": None,
            "unique": True,
            "bands": [3, 2, 1],
        }
    )

    assert fake_sam.tiled["tile_size"] == 512
    assert fake_sam.tiled["overlap"] == 64
    assert fake_sam.tiled["bands"] == [3, 2, 1]
    assert result == {"output": "tiles.tif", "object_count": 4, "mask_count": 0}
    assert seen == ["tiles.tif"]


def test_count_raster_labels_worker(monkeypatch):
    monkeypatch.setattr(
        samgeo_worker, "_iter_raster_blocks", lambda path: iter([[[0, 3], [3, 9]]])
    )

    assert samgeo_worker._count_raster_labels("x.tif", unique=True) == 2
    assert samgeo_worker._count_raster_labels("x.tif", unique=False) is None


def test_count_raster_labels_worker_unreadable(monkeypatch):
    def _fail(path):
        raise OSError("cannot open")

    monkeypatch.setattr(samgeo_worker, "_iter_raster_blocks", _fail)

    assert samgeo_worker._count_raster_labels("x.tif", unique=True) is None


def test_save_masks_drops_non_finite_scores(monkeypatch):
    sam = FakeSam()
    monkeypatch.setitem(samgeo_worker._STATE, "sam", sam)

    def save_masks(output, unique=True):
        sam.mask_scores = {1: float("nan"), 2: 0.5, 3: float("inf")}

    sam.save_masks = save_masks

    result = samgeo_worker._handle_save_masks({"output": "o.tif", "unique": True})

    assert result["mask_scores"] == {"2": 0.5}
    json.loads(json.dumps(samgeo_worker._json_safe(result), allow_nan=False))


def test_supported_methods_include_tiled():
    assert "generate_masks_tiled" in samgeo_worker._supported_methods_for(FakeSam())


def test_client_save_masks_stores_scores(monkeypatch):
    client = SamGeoSubprocessClient("SamGeo3", "meta", None, 0.5, True)
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, timeout=None: {"mask_count": 2, "mask_scores": {"1": 0.7}},
    )

    client.save_masks("out.tif", unique=True)

    assert client.mask_scores == {"1": 0.7}
    assert len(client.masks) == 2


def test_client_generate_masks_tiled_payload(monkeypatch):
    client = SamGeoSubprocessClient("SamGeo3", "meta", None, 0.5, True)
    sent = {}

    def _request(payload, timeout=None):
        sent.update(payload)
        return {"output": payload["output"], "object_count": 4, "mask_count": 0}

    monkeypatch.setattr(client, "_request", _request)

    result = client.generate_masks_tiled(
        "big.tif", "tree", "out.tif", tile_size=2048, overlap=256
    )

    assert sent["action"] == "generate_masks_tiled"
    assert sent["tile_size"] == 2048 and sent["overlap"] == 256
    assert result["object_count"] == 4
    assert client.masks is None


# ---------------------------------------------------------------------------
# Hugging Face tasks
# ---------------------------------------------------------------------------


def _fake_hub(
    monkeypatch, *, valid=True, stored_token=None, login_error=None, token_path=None
):
    hub = types.ModuleType("huggingface_hub")
    calls = {}

    def login(token=None, add_to_git_credential=False):
        calls["login"] = (token, add_to_git_credential)
        print("Login successful")  # must not reach the protocol stdout
        if login_error:
            raise ValueError(login_error)
        if token_path is not None:
            with open(token_path, "w") as handle:
                handle.write(token)

    def whoami(token=None):
        calls["whoami"] = token
        if not valid:
            raise RuntimeError(f"401 Client Error: Invalid token {token}")
        return {"name": "jdoe", "auth": {"accessToken": {"role": "read"}}}

    hub.login = login
    hub.whoami = whoami
    hub.get_token = lambda: stored_token
    if token_path is not None:
        hub.constants = types.SimpleNamespace(HF_TOKEN_PATH=str(token_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_hf_tasks_registered():
    assert geoai_task_worker._TASKS["hf_login"] is geoai_task_worker._task_hf_login
    assert geoai_task_worker._TASKS["hf_whoami"] is geoai_task_worker._task_hf_whoami


def test_hf_login_success(monkeypatch, capsys):
    calls = _fake_hub(monkeypatch)

    result = geoai_task_worker._task_hf_login({"token": f"  {TOKEN}\n"})

    assert result == {"logged_in": True, "user": "jdoe"}
    assert calls["login"] == (TOKEN, False)
    assert TOKEN not in json.dumps(result)
    # Library prints are diverted away from the JSON protocol stdout.
    assert "Login successful" not in capsys.readouterr().out


def test_hf_login_error_is_redacted(monkeypatch):
    _fake_hub(monkeypatch, login_error=f"Invalid user token {TOKEN}.")

    with pytest.raises(RuntimeError) as excinfo:
        geoai_task_worker._task_hf_login({"token": TOKEN})

    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


def test_hf_login_whoami_failure_still_logged_in(monkeypatch):
    """Once login() saved the token, a failed name lookup is not a login error."""
    _fake_hub(monkeypatch, valid=False)

    result = geoai_task_worker._task_hf_login({"token": TOKEN})

    assert result == {"logged_in": True, "user": None}


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_hf_login_token_file_is_private(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("old")
    os.chmod(token_file, 0o644)
    _fake_hub(monkeypatch, token_path=token_file)

    geoai_task_worker._task_hf_login({"token": TOKEN})

    assert (token_file.stat().st_mode & 0o777) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_hf_login_new_token_file_created_private(monkeypatch, tmp_path):
    token_file = tmp_path / "new_token"
    _fake_hub(monkeypatch, token_path=token_file)
    monkeypatch.setattr(geoai_task_worker, "_restrict_token_files", lambda hub: None)

    geoai_task_worker._task_hf_login({"token": TOKEN})

    # Created under umask 077 even without the chmod step.
    assert (token_file.stat().st_mode & 0o077) == 0


def test_hf_login_requires_token(monkeypatch):
    _fake_hub(monkeypatch)

    with pytest.raises(ValueError):
        geoai_task_worker._task_hf_login({"token": "  "})


def test_hf_login_main_never_echoes_token(monkeypatch, capsys):
    """End to end through main(): the protocol output must not contain the token."""
    _fake_hub(monkeypatch, login_error=f"bad {TOKEN}")
    emitted = []
    monkeypatch.setattr(geoai_task_worker, "_emit", emitted.append)
    monkeypatch.setattr(
        sys,
        "stdin",
        types.SimpleNamespace(
            readline=lambda: json.dumps(
                {"action": "hf_login", "params": {"token": TOKEN}}
            )
        ),
    )

    assert geoai_task_worker.main() == 1

    assert emitted and emitted[-1]["type"] == "error"
    assert TOKEN not in json.dumps(emitted)
    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


def test_hf_whoami_not_logged_in(monkeypatch):
    _fake_hub(monkeypatch, stored_token=None)

    result = geoai_task_worker._task_hf_whoami({})

    assert result["logged_in"] is False
    assert result["token_found"] is False


def test_hf_whoami_logged_in(monkeypatch):
    _fake_hub(monkeypatch, stored_token=TOKEN)

    result = geoai_task_worker._task_hf_whoami({})

    assert result["logged_in"] is True
    assert result["user"] == "jdoe"
    assert TOKEN not in json.dumps(result)


def test_hf_whoami_invalid_token_redacted(monkeypatch):
    _fake_hub(monkeypatch, stored_token=TOKEN, valid=False)

    result = geoai_task_worker._task_hf_whoami({})

    assert result["logged_in"] is False
    assert result["token_found"] is True
    assert TOKEN not in json.dumps(result)
