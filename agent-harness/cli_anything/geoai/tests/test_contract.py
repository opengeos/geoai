"""Contract and regression tests for agent-harness (cli-anything-geoai).

1. Validates every geoai import and function/class signature called by
   cli_anything.geoai.core.* against the real geoai modules via
   ``inspect.signature`` (with AST fallback only when optional heavy ML deps
   are absent in lightweight CI).
2. Regression tests for one-shot --project persistence, REPL --json mode,
   undo/redo step descriptions and MAX_UNDO, atomic project writes, corrupt
   non-dict project JSON, and NaN/Inf JSON sanitization.
"""

import ast
import importlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_bounds

from cli_anything.geoai import geoai_cli
from cli_anything.geoai.core import (
    change as change_mod,
    classify as classify_mod,
    data as data_mod,
    detect as detect_mod,
    project as project_mod,
    raster as raster_mod,
    segment as segment_mod,
    session as session_mod,
    vector as vector_mod,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Real geoai signature verification helpers
# ---------------------------------------------------------------------------


def _find_ast_node(body, parts):
    head, *rest = parts
    for stmt in body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == head:
            if not rest:
                for sub in stmt.body:
                    if (
                        isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and sub.name == "__init__"
                    ):
                        return sub
                return ast.FunctionDef(
                    name="__init__",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="self")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[],
                    decorator_list=[],
                )
            return _find_ast_node(stmt.body, rest)
        if (
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name == head
            and not rest
        ):
            return stmt
    return None


def _ast_func_to_signature(func_node, is_method: bool = False) -> inspect.Signature:
    args = func_node.args
    params = []
    all_pos = list(args.posonlyargs) + list(args.args)
    if (
        (is_method or func_node.name == "__init__")
        and all_pos
        and all_pos[0].arg == "self"
    ):
        all_pos = all_pos[1:]
        posonly_count = max(0, len(args.posonlyargs) - 1)
    else:
        posonly_count = len(args.posonlyargs)

    first_default_idx = len(all_pos) - len(args.defaults)
    for idx, arg in enumerate(all_pos):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if idx < posonly_count
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = None if idx >= first_default_idx else inspect.Parameter.empty
        params.append(inspect.Parameter(arg.arg, kind, default=default))

    if args.vararg:
        params.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )

    for arg, kw_def in zip(args.kwonlyargs, args.kw_defaults):
        default = None if kw_def is not None else inspect.Parameter.empty
        params.append(
            inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=default)
        )

    if args.kwarg:
        params.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))

    return inspect.Signature(params)


def _signature_from_ast(module_name: str, qualname: str) -> inspect.Signature:
    rel = Path(*module_name.split("."))
    candidates = [
        REPO_ROOT / rel.with_suffix(".py"),
        REPO_ROOT / rel / "__init__.py",
    ]
    pkg_dir = REPO_ROOT / rel
    if pkg_dir.is_dir():
        candidates.extend(sorted(pkg_dir.glob("*.py")))

    target_parts = qualname.split(".")
    for path in candidates:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = _find_ast_node(tree.body, target_parts)
        if node is not None:
            return _ast_func_to_signature(node, is_method=(len(target_parts) > 1))

    raise AssertionError(
        f"Symbol {module_name}.{qualname} not found in real geoai source under {REPO_ROOT / 'geoai'}"
    )


def get_real_signature(module_name: str, qualname: str) -> inspect.Signature:
    """Get inspect.Signature from real imported geoai object, with AST fallback."""
    try:
        mod = importlib.import_module(module_name)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        sig = inspect.signature(obj)
        if len(qualname.split(".")) > 1 and "self" in sig.parameters:
            return sig.replace(
                parameters=[p for n, p in sig.parameters.items() if n != "self"]
            )
        return sig
    except ImportError:
        return _signature_from_ast(module_name, qualname)


def make_strict_callable(module_name: str, qualname: str, return_value=None):
    """Return a fake callable that enforces binding against the real geoai signature."""
    sig = get_real_signature(module_name, qualname)
    calls = []

    def _fn(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        calls.append(bound.arguments)
        return return_value

    _fn.calls = calls
    _fn.signature = sig
    return _fn


# ---------------------------------------------------------------------------
# Contract tests for all 5 core modules
# ---------------------------------------------------------------------------


class TestHarnessGeoAIContracts:
    """Verify every core wrapper binds strictly to real geoai signatures."""

    def test_segment_wrappers_match_real_signatures(self, tmp_path):
        raster = tmp_path / "img.tif"
        raster.write_bytes(b"tif")
        model = tmp_path / "model.pth"
        model.write_bytes(b"pth")
        img_dir = tmp_path / "imgs"
        lbl_dir = tmp_path / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        sam_init = make_strict_callable("geoai.sam", "SamGeo")
        sam_gen = make_strict_callable("geoai.sam", "SamGeo.generate")
        gsam_init = make_strict_callable("geoai.segment", "GroundedSAM")
        gsam_seg = make_strict_callable("geoai.segment", "GroundedSAM.segment_image")
        sem_fn = make_strict_callable("geoai.train", "semantic_segmentation")
        train_seg_fn = make_strict_callable("geoai.train", "train_segmentation_model")

        class _FakeSamGeo:
            def __init__(self, *args, **kwargs):
                sam_init(*args, **kwargs)

            def generate(self, *args, **kwargs):
                return sam_gen(*args, **kwargs)

        class _FakeGroundedSAM:
            def __init__(self, *args, **kwargs):
                gsam_init(*args, **kwargs)

            def segment_image(self, *args, **kwargs):
                return gsam_seg(*args, **kwargs)

        fake_sam_mod = types.ModuleType("geoai.sam")
        fake_sam_mod.SamGeo = _FakeSamGeo
        fake_seg_mod = types.ModuleType("geoai.segment")
        fake_seg_mod.GroundedSAM = _FakeGroundedSAM
        fake_train_mod = types.ModuleType("geoai.train")
        fake_train_mod.semantic_segmentation = sem_fn
        fake_train_mod.train_segmentation_model = train_seg_fn

        with patch.dict(
            sys.modules,
            {
                "geoai.sam": fake_sam_mod,
                "geoai.segment": fake_seg_mod,
                "geoai.train": fake_train_mod,
            },
        ):
            segment_mod.run_sam(str(raster), str(tmp_path / "sam.tif"))
            segment_mod.run_grounded_sam(
                str(raster), str(tmp_path / "gsam.tif"), prompt="building"
            )
            segment_mod.run_semantic_segmentation(
                str(raster),
                str(model),
                str(tmp_path / "sem.tif"),
                num_classes=3,
                chip_size=256,
                overlap=0.25,
            )
            segment_mod.train_segmentation(
                str(img_dir),
                str(lbl_dir),
                str(tmp_path / "out_train"),
                architecture="unet",
                backbone="resnet50",
                num_classes=3,
                in_channels=4,
                epochs=5,
                loss="ce",
            )

        assert sam_init.calls[0]["model"] == "facebook/sam-vit-huge"
        assert sam_gen.calls[0]["source"] == os.path.abspath(str(raster))
        assert sem_fn.calls[0]["input_path"] == os.path.abspath(str(raster))
        assert sem_fn.calls[0]["window_size"] == 256
        assert sem_fn.calls[0]["overlap"] == 64
        assert train_seg_fn.calls[0]["images_dir"] == os.path.abspath(str(img_dir))
        assert train_seg_fn.calls[0]["encoder_name"] == "resnet50"
        assert train_seg_fn.calls[0]["num_channels"] == 4
        assert train_seg_fn.calls[0]["num_epochs"] == 5

    def test_detect_wrappers_match_real_signatures(self, tmp_path):
        raster = tmp_path / "img.tif"
        raster.write_bytes(b"tif")
        model = tmp_path / "model.pth"
        model.write_bytes(b"pth")
        img_dir = tmp_path / "imgs"
        lbl_dir = tmp_path / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        det_fn = make_strict_callable("geoai.object_detect", "multiclass_detection")
        train_det_fn = make_strict_callable("geoai.train", "train_MaskRCNN_model")

        fake_od = types.ModuleType("geoai.object_detect")
        fake_od.multiclass_detection = det_fn
        fake_train = types.ModuleType("geoai.train")
        fake_train.train_MaskRCNN_model = train_det_fn

        with patch.dict(
            sys.modules,
            {"geoai.object_detect": fake_od, "geoai.train": fake_train},
        ):
            detect_mod.run_detection(
                str(raster),
                str(model),
                num_classes=4,
                output_raster=str(tmp_path / "det.tif"),
                chip_size=512,
                overlap=0.5,
            )
            detect_mod.train_detector(
                str(img_dir),
                str(lbl_dir),
                str(tmp_path / "det_train"),
                num_classes=4,
                epochs=12,
            )

        assert det_fn.calls[0]["input_path"] == os.path.abspath(str(raster))
        assert det_fn.calls[0]["output_path"] == os.path.abspath(
            str(tmp_path / "det.tif")
        )
        assert det_fn.calls[0]["overlap"] == 256
        assert train_det_fn.calls[0]["num_epochs"] == 12

    def test_classify_wrappers_match_real_signatures(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        img = tmp_path / "sample.tif"
        img.write_bytes(b"tif")
        ckpt = tmp_path / "cls.ckpt"
        ckpt.write_bytes(b"ckpt")

        train_cls_fn = make_strict_callable(
            "geoai.recognize",
            "train_image_classifier",
            return_value={"checkpoint_path": str(ckpt), "class_names": ["a", "b"]},
        )
        pred_fn = make_strict_callable(
            "geoai.timm_train",
            "predict_with_timm",
            return_value=(np.array([1]), np.array([[0.2, 0.8]])),
        )

        class _FakeTimmClassifier:
            @classmethod
            def load_from_checkpoint(cls, checkpoint_path, **kwargs):
                return MagicMock()

        fake_rec = types.ModuleType("geoai.recognize")
        fake_rec.train_image_classifier = train_cls_fn
        fake_timm = types.ModuleType("geoai.timm_train")
        fake_timm.TimmClassifier = _FakeTimmClassifier
        fake_timm.predict_with_timm = pred_fn

        with patch.dict(
            sys.modules,
            {"geoai.recognize": fake_rec, "geoai.timm_train": fake_timm},
        ):
            tr = classify_mod.train_classifier(
                str(train_dir), output_dir=str(tmp_path / "cls_out"), num_epochs=5
            )
            pr = classify_mod.predict_classification(
                str(img), str(ckpt), num_classes=2, class_names=["forest", "water"]
            )

        assert train_cls_fn.calls[0]["data_dir"] == os.path.abspath(str(train_dir))
        assert tr["checkpoint_path"] == str(ckpt)
        assert pred_fn.calls[0]["image_paths"] == [os.path.abspath(str(img))]
        assert pr["predicted_class"] == "water"

    def test_change_wrappers_match_real_signatures(self, tmp_path):
        img1 = tmp_path / "t1.tif"
        img2 = tmp_path / "t2.tif"
        img1.write_bytes(b"t1")
        img2.write_bytes(b"t2")

        cs_fn = make_strict_callable("geoai.change_detection", "changestar_detect")
        cd_init = make_strict_callable("geoai.change_detection", "ChangeDetection")
        cd_det = make_strict_callable(
            "geoai.change_detection", "ChangeDetection.detect_changes"
        )

        class _FakeChangeDetection:
            def __init__(self, *args, **kwargs):
                cd_init(*args, **kwargs)

            def detect_changes(self, *args, **kwargs):
                return cd_det(*args, **kwargs)

        fake_cd = types.ModuleType("geoai.change_detection")
        fake_cd.changestar_detect = cs_fn
        fake_cd.ChangeDetection = _FakeChangeDetection

        with patch.dict(sys.modules, {"geoai.change_detection": fake_cd}):
            change_mod.detect_changes(
                str(img1),
                str(img2),
                str(tmp_path / "cs.tif"),
                method="changestar",
                confidence_threshold=128,
            )
            change_mod.detect_changes(
                str(img1),
                str(img2),
                str(tmp_path / "ac.tif"),
                method="anychange",
            )

        assert cs_fn.calls[0]["image1_path"] == os.path.abspath(str(img1))
        assert cs_fn.calls[0]["output_change"] == os.path.abspath(
            str(tmp_path / "cs.tif")
        )
        assert 0.49 <= cs_fn.calls[0]["threshold"] <= 0.51
        assert cd_det.calls[0]["output_path"] == os.path.abspath(
            str(tmp_path / "ac.tif")
        )

    def test_data_wrappers_match_real_signatures(self, tmp_path):
        fake_item = SimpleNamespace(
            id="scene1",
            datetime="2024-06-01",
            properties={"eo:cloud_cover": 2.0},
            assets={"visual": "href"},
        )
        stac_fn = make_strict_callable(
            "geoai.download", "pc_stac_search", return_value=[fake_item]
        )
        naip_fn = make_strict_callable(
            "geoai.download", "download_naip", return_value=["a.tif"]
        )
        overture_fn = make_strict_callable(
            "geoai.download", "get_overture_data", return_value=None
        )

        fake_dl = types.ModuleType("geoai.download")
        fake_dl.pc_stac_search = stac_fn
        fake_dl.download_naip = naip_fn
        fake_dl.get_overture_data = overture_fn

        with patch.dict(sys.modules, {"geoai.download": fake_dl}):
            stac_res = data_mod.search_stac(
                (-84.0, 35.9, -83.9, 36.0),
                collection="sentinel-2-l2a",
                start_date="2024-01-01",
                end_date="2024-06-01",
                max_items=5,
            )
            naip_res = data_mod.download_naip(
                (-84.0, 35.9, -83.9, 36.0),
                output=str(tmp_path / "naip_dir"),
                year=2022,
                max_items=3,
            )
            ov_res = data_mod.download_overture(
                (-84.0, 35.9, -83.9, 36.0),
                output=str(tmp_path / "places"),
                data_type="place",
                output_format="parquet",
            )

        assert stac_res["total_results"] == 1
        assert stac_fn.calls[0]["collection"] == "sentinel-2-l2a"
        assert stac_fn.calls[0]["time_range"] == "2024-01-01/2024-06-01"
        assert naip_fn.calls[0]["output_dir"] == os.path.abspath(
            str(tmp_path / "naip_dir")
        )
        assert naip_res["file_count"] == 1
        assert overture_fn.calls[0]["overture_type"] == "place"
        assert overture_fn.calls[0]["output"].endswith(".parquet")
        assert ov_res["output_format"] == "parquet"


# ---------------------------------------------------------------------------
# Regression tests for CLI, Session, Project, and NaN/Inf JSON safety
# ---------------------------------------------------------------------------


class TestHarnessRegressions:
    """Regression tests for CLI/session/project/JSON fixes."""

    def test_oneshot_project_add_file_persists_to_disk(self, tmp_path):
        """One-shot --project project add-file must persist the mutation to disk."""
        raster_path = tmp_path / "scene.tif"
        transform = from_bounds(0, 0, 1, 1, 8, 8)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.ones((1, 8, 8), dtype=np.uint8))

        proj_path = tmp_path / "work.json"
        runner = CliRunner()
        r1 = runner.invoke(
            geoai_cli.cli,
            ["--json", "project", "new", "-n", "persist_test", "-o", str(proj_path)],
        )
        assert r1.exit_code == 0, r1.output

        r2 = runner.invoke(
            geoai_cli.cli,
            [
                "--json",
                "--project",
                str(proj_path),
                "project",
                "add-file",
                str(raster_path),
            ],
        )
        assert r2.exit_code == 0, r2.output

        reloaded = project_mod.open_project(str(proj_path))
        assert len(reloaded["files"]) == 1
        assert reloaded["files"][0]["path"] == os.path.abspath(str(raster_path))

    def test_project_load_error_returns_clean_json(self, tmp_path):
        """--json --project on missing or corrupt JSON returns valid JSON error."""
        runner = CliRunner()
        bad_proj = tmp_path / "corrupt.json"
        bad_proj.write_text("[1, 2, 3]", encoding="utf-8")

        res_corrupt = runner.invoke(
            geoai_cli.cli,
            ["--json", "--project", str(bad_proj), "project", "info"],
        )
        assert res_corrupt.exit_code != 0
        payload = json.loads(res_corrupt.output)
        assert "error" in payload
        assert payload["type"] == "ValueError"

        res_missing = runner.invoke(
            geoai_cli.cli,
            [
                "--json",
                "--project",
                str(tmp_path / "missing.json"),
                "project",
                "info",
            ],
        )
        assert res_missing.exit_code != 0
        payload2 = json.loads(res_missing.output)
        assert payload2["type"] == "FileNotFoundError"

    def test_open_project_rejects_non_dict_json(self, tmp_path):
        """open_project raises ValueError when JSON root is not an object."""
        for content in ("[1, 2, 3]", '"just a string"', "null", "42"):
            p = tmp_path / "not_dict.json"
            p.write_text(content, encoding="utf-8")
            with pytest.raises(ValueError, match="JSON object"):
                project_mod.open_project(str(p))

    def test_repl_preserves_json_output_across_dispatches(self):
        """When REPL is running with _json_output=True, subcommands preserve JSON mode."""
        runner = CliRunner()
        prev_repl = geoai_cli._repl_mode
        prev_json = geoai_cli._json_output
        try:
            geoai_cli._repl_mode = True
            geoai_cli._json_output = True
            res = runner.invoke(geoai_cli.cli, ["data", "sources"])
            assert res.exit_code == 0, res.output
            parsed = json.loads(res.output)
            assert isinstance(parsed, list)
            assert geoai_cli._json_output is True
        finally:
            geoai_cli._repl_mode = prev_repl
            geoai_cli._json_output = prev_json

    def test_session_undo_redo_preserves_descriptions_and_max_undo(self):
        """undo() and redo() preserve original descriptions and enforce MAX_UNDO."""
        sess = session_mod.Session()
        sess.MAX_UNDO = 3
        sess.set_project(project_mod.create_project(name="v0"))

        sess.snapshot("rename to v1")
        sess.project["name"] = "v1"
        sess.snapshot("rename to v2")
        sess.project["name"] = "v2"

        assert sess.undo() == "rename to v2"
        assert sess.project["name"] == "v1"
        assert sess.redo() == "rename to v2"
        assert sess.project["name"] == "v2"
        # Description preserved back on undo stack after redo
        assert sess._undo_stack[-1]["description"] == "rename to v2"

        # MAX_UNDO enforced on redo() as well
        sess._undo_stack = [
            {"project": {"name": f"u{i}"}, "description": f"u{i}", "timestamp": ""}
            for i in range(3)
        ]
        sess._redo_stack = [
            {"project": {"name": "r0"}, "description": "r0", "timestamp": ""}
        ]
        sess.redo()
        assert len(sess._undo_stack) == 3

    def test_save_failure_does_not_mutate_modified_timestamp(self, tmp_path):
        """Failed write in save_project/save_session leaves project['modified'] intact."""
        proj = project_mod.create_project(name="atomic_test")
        orig_modified = proj["modified"]
        sess = session_mod.Session()
        sess.set_project(proj, str(tmp_path / "p.json"))

        with patch(
            "cli_anything.geoai.core.project.atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError):
                project_mod.save_project(proj, str(tmp_path / "p.json"))
        assert proj["modified"] == orig_modified

        with patch(
            "cli_anything.geoai.core.session.atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError):
                sess.save_session()
        assert sess.project["modified"] == orig_modified

    def test_nan_and_inf_sanitized_in_raster_vector_and_cli_json(self, tmp_path):
        """NaN/Inf values are converted to None so strict JSON serialization succeeds."""
        raster_path = tmp_path / "nan_raster.tif"
        data = np.full((1, 8, 8), np.nan, dtype=np.float32)
        data[0, 0, 0] = np.inf
        transform = from_bounds(0, 0, 1, 1, 8, 8)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=float("nan"),
        ) as dst:
            dst.write(data)

        stats = raster_mod.get_raster_stats(str(raster_path), band=1)
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["nodata_value"] is None
        # Must be strictly valid RFC 8259 JSON (no NaN/Infinity literals)
        dumped = json.dumps(stats, allow_nan=False)
        assert "NaN" not in dumped and "Infinity" not in dumped

        # Empty vector file produces NaN bounds in geopandas -> must become None
        import geopandas as gpd

        empty_vec = tmp_path / "empty.geojson"
        gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326").to_file(
            empty_vec, driver="GeoJSON"
        )
        vinfo = vector_mod._get_vector_info_geopandas(str(empty_vec))
        dumped_vec = json.dumps(vinfo, allow_nan=False)
        assert "NaN" not in dumped_vec

        # CLI --json raster stats must also emit strict JSON
        runner = CliRunner()
        res = runner.invoke(
            geoai_cli.cli, ["--json", "raster", "stats", str(raster_path)]
        )
        assert res.exit_code == 0, res.output
        parsed = json.loads(
            res.output,
            parse_constant=lambda c: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant: {c}")
            ),
        )
        assert parsed["min"] is None

    def test_change_threshold_int_vs_float_and_min_area_filter(self, tmp_path):
        """Integer thresholds use 0-255 scale (1 -> 1/255), float 1.0 stays 1.0, and min_area filters output."""
        img1 = tmp_path / "t1.tif"
        img2 = tmp_path / "t2.tif"
        img1.write_bytes(b"t1")
        img2.write_bytes(b"t2")

        transform = from_bounds(0, 0, 10, 10, 10, 10)

        def _write_change_mask(path):
            arr = np.zeros((1, 10, 10), dtype=np.uint8)
            arr[0, 0:2, 0:2] = 1  # 4 pixels (below min_area=10)
            arr[0, 5:9, 5:9] = 1  # 16 pixels (above min_area=10)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(arr)

        thresholds_seen = []

        def _fake_changestar(image1_path, image2_path, output_change, threshold):
            thresholds_seen.append(threshold)
            _write_change_mask(output_change)

        class _FakeAnyChange:
            def detect_changes(self, image1_path, image2_path, output_path):
                _write_change_mask(output_path)

        fake_cd = types.ModuleType("geoai.change_detection")
        fake_cd.changestar_detect = _fake_changestar
        fake_cd.ChangeDetection = _FakeAnyChange

        with patch.dict(sys.modules, {"geoai.change_detection": fake_cd}):
            r_int1 = change_mod.detect_changes(
                str(img1),
                str(img2),
                str(tmp_path / "cs_int1.tif"),
                method="changestar",
                confidence_threshold=1,
                min_area=10,
            )
            change_mod.detect_changes(
                str(img1),
                str(img2),
                str(tmp_path / "cs_flt1.tif"),
                method="changestar",
                confidence_threshold=1.0,
                min_area=0,
            )
            r_any = change_mod.detect_changes(
                str(img1),
                str(img2),
                str(tmp_path / "ac_min.tif"),
                method="anychange",
                min_area=10,
            )

        assert pytest.approx(thresholds_seen[0], rel=1e-6) == 1.0 / 255.0
        assert thresholds_seen[1] == 1.0
        assert r_int1["changed_pixels"] == 16
        assert r_any["changed_pixels"] == 16

    def test_detect_preserves_band_order_and_instance_separation(self, tmp_path):
        """run_detection subsets requested band_indexes in order and polygonizes band 2 instances."""
        raster_path = tmp_path / "four_band.tif"
        transform = from_bounds(0, 0, 8, 8, 8, 8)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=4,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for idx in range(4):
                dst.write(np.full((8, 8), (idx + 1) * 10, dtype=np.uint8), idx + 1)

        model = tmp_path / "det.pth"
        model.write_bytes(b"pth")

        observed_band_means = []

        def _fake_multiclass_detection(
            input_path,
            output_path,
            model_path,
            num_classes,
            confidence_threshold,
            window_size,
            overlap,
            num_channels=3,
        ):
            with rasterio.open(input_path) as src:
                assert src.count == num_channels
                observed_band_means.append(
                    [int(src.read(i).mean()) for i in range(1, src.count + 1)]
                )
            # Write 2-band output: band 1 = class (both class 2), band 2 = instance IDs (1 and 2, touching!)
            class_band = np.zeros((8, 8), dtype=np.uint8)
            inst_band = np.zeros((8, 8), dtype=np.uint8)
            class_band[1:4, 1:3] = 2
            inst_band[1:4, 1:3] = 1
            class_band[1:4, 3:5] = 2
            inst_band[1:4, 3:5] = 2
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=2,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(class_band, 1)
                dst.write(inst_band, 2)

        fake_od = types.ModuleType("geoai.object_detect")
        fake_od.multiclass_detection = _fake_multiclass_detection

        out_vec = tmp_path / "dets.geojson"
        with patch.dict(sys.modules, {"geoai.object_detect": fake_od}):
            res = detect_mod.run_detection(
                str(raster_path),
                str(model),
                num_classes=3,
                output_vector=str(out_vec),
                band_indexes=[3, 1, 0],
            )

        assert observed_band_means == [[40, 20, 10]]
        assert res["detection_count"] == 2
        import geopandas as gpd

        gdf = gpd.read_file(out_vec)
        assert len(gdf) == 2
        assert sorted(gdf["instance_id"].tolist()) == [1, 2]
        assert gdf["class"].tolist() == [2, 2]

    def test_segment_rejects_non_automatic_sam_and_preserves_semantic_config(
        self, tmp_path
    ):
        """run_sam rejects automatic=False; run_semantic_segmentation forwards backbone/in_channels and band order."""
        raster_path = tmp_path / "four_band.tif"
        transform = from_bounds(0, 0, 8, 8, 8, 8)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=4,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for idx in range(4):
                dst.write(np.full((8, 8), (idx + 1) * 10, dtype=np.uint8), idx + 1)

        model = tmp_path / "seg.pth"
        model.write_bytes(b"pth")

        with pytest.raises(ValueError, match="automatic=False"):
            segment_mod.run_sam(
                str(raster_path), str(tmp_path / "sam.tif"), automatic=False
            )

        with pytest.raises(ValueError, match="band_indexes"):
            segment_mod.run_semantic_segmentation(
                str(raster_path),
                str(model),
                str(tmp_path / "sem.tif"),
                in_channels=4,
                band_indexes=[0, 1, 2],
            )

        recorded = {}

        def _fake_sem(
            input_path,
            output_path,
            model_path,
            architecture,
            encoder_name,
            num_channels,
            num_classes,
            window_size,
            overlap,
        ):
            with rasterio.open(input_path) as src:
                recorded["band_means"] = [
                    int(src.read(i).mean()) for i in range(1, src.count + 1)
                ]
            recorded["architecture"] = architecture
            recorded["encoder_name"] = encoder_name
            recorded["num_channels"] = num_channels

        fake_train = types.ModuleType("geoai.train")
        fake_train.semantic_segmentation = _fake_sem

        with patch.dict(sys.modules, {"geoai.train": fake_train}):
            segment_mod.run_semantic_segmentation(
                str(raster_path),
                str(model),
                str(tmp_path / "sem.tif"),
                in_channels=3,
                band_indexes=[3, 2, 1],
            )

        assert recorded["band_means"] == [40, 30, 20]
        assert recorded["architecture"] == "unet"
        assert recorded["encoder_name"] == "resnet50"
        assert recorded["num_channels"] == 3

    def test_repl_project_load_error_stops_subcommand_dispatch(self, tmp_path):
        """In REPL mode, --project missing.json must NOT run the subcommand on the prior project."""
        proj_a = project_mod.create_project(name="project_a")
        proj_a_path = tmp_path / "proj_a.json"
        project_mod.save_project(proj_a, str(proj_a_path))

        scene = tmp_path / "scene.tif"
        scene.write_bytes(b"tif")

        sess = geoai_cli.get_session()
        sess.set_project(proj_a, str(proj_a_path))
        prev_repl = geoai_cli._repl_mode
        try:
            geoai_cli._repl_mode = True
            exit_code = geoai_cli.cli.main(
                [
                    "--project",
                    str(tmp_path / "missing.json"),
                    "project",
                    "add-file",
                    str(scene),
                ],
                standalone_mode=False,
            )
            assert exit_code == 1
            assert len(sess.project["files"]) == 0
        finally:
            geoai_cli._repl_mode = prev_repl

    def test_sanitize_json_value_terminates_on_longdouble(self):
        """sanitize_json_value and serialize_info terminate on np.longdouble scalars."""
        from cli_anything.geoai.utils.serialization import (
            sanitize_json_value,
            serialize_info,
        )

        ld = np.longdouble("1.25")
        val = sanitize_json_value(ld)
        assert isinstance(val, float)
        assert val == pytest.approx(1.25)

        info = serialize_info({"ld": ld, "nan_ld": np.longdouble("nan")})
        assert info["ld"] == pytest.approx(1.25)
        assert info["nan_ld"] is None
        json.dumps(info, allow_nan=False)
