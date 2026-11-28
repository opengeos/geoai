"""Contract tests verifying every geoai API call in geoai_mcp_server/server.py.

Every mocked geoai symbol invoked by server.py is validated against the real
``inspect.signature`` of the target function/class in the repo's ``geoai``
package (falling back to AST signature parsing of the real ``geoai`` source
file only when optional heavy ML dependencies such as torch/transformers are
not installed in lightweight CI).
"""

import ast
import asyncio
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _signature_from_ast(module_name: str, qualname: str) -> inspect.Signature:
    """Extract an inspect.Signature from the real geoai source file via AST."""
    parts = module_name.split(".")
    rel = Path(*parts)
    candidates = [
        REPO_ROOT / rel.with_suffix(".py"),
        REPO_ROOT / rel / "__init__.py",
    ]
    # Also scan submodules under geoai/utils/ when target is re-exported on geoai.utils
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


def _find_ast_node(body, parts):
    head, *rest = parts
    for stmt in body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == head:
            if not rest:
                # Look for __init__ inside class
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


def _ast_func_to_signature(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False
) -> inspect.Signature:
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

    num_defaults = len(args.defaults)
    first_default_idx = len(all_pos) - num_defaults

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


def get_real_signature(module_name: str, qualname: str) -> inspect.Signature:
    """Get inspect.Signature from real imported geoai object, with AST fallback."""
    try:
        mod = importlib.import_module(module_name)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        sig = inspect.signature(obj)
        # Strip 'self' when inspecting an unbound method on a class
        parts = qualname.split(".")
        if len(parts) > 1 and "self" in sig.parameters:
            new_params = [p for name, p in sig.parameters.items() if name != "self"]
            return sig.replace(parameters=new_params)
        return sig
    except ImportError:
        return _signature_from_ast(module_name, qualname)


def make_strict_callable(module_name: str, qualname: str, return_value=None):
    """Build a wrapper that binds every call strictly against the real geoai signature."""
    sig = get_real_signature(module_name, qualname)
    calls = []

    def _call(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        calls.append(bound.arguments)
        if callable(return_value):
            return return_value(*args, **kwargs)
        return return_value

    _call.calls = calls
    _call.signature = sig
    return _call


class TestMCPGeoAIContracts:
    """Verify every geoai call made by geoai_mcp_server.server binds to the real API."""

    def test_sam_and_auto_segment_contract(self, sample_geotiff, temp_output_dir):
        from geoai_mcp_server.server import (
            auto_segment_image,
            segment_objects_with_prompts,
        )

        sam_init = make_strict_callable("geoai.sam", "SamGeo")
        sam_generate = make_strict_callable("geoai.sam", "SamGeo.generate")
        clean_mask = make_strict_callable(
            "geoai.utils", "clean_instance_mask", return_value="cleaned.tif"
        )
        r2v = make_strict_callable(
            "geoai.utils", "raster_to_vector", return_value=[1, 2]
        )

        class _FakeSamGeo:
            def __init__(self, *args, **kwargs):
                sam_init(*args, **kwargs)
                self.masks = [{"area": 100}]

            def generate(self, *args, **kwargs):
                return sam_generate(*args, **kwargs)

        modules = {
            "sam": SimpleNamespace(SamGeo=_FakeSamGeo),
            "utils": SimpleNamespace(
                clean_instance_mask=clean_mask, raster_to_vector=r2v
            ),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_sam = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="sam",
                    output_format="geotiff",
                )
            )
            res_sam2 = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="sam2",
                    output_format="geojson",
                )
            )
            res_auto = asyncio.run(
                auto_segment_image(
                    image_path=sample_geotiff.name,
                    output_format="geojson",
                    min_object_size=50,
                    clean_results=True,
                )
            )

        assert res_sam["success"] is True
        assert res_sam2["success"] is True
        assert res_auto["success"] is True
        assert len(sam_init.calls) == 3
        assert len(sam_generate.calls) == 3
        assert len(clean_mask.calls) == 1
        assert len(r2v.calls) == 2

    def test_grounded_sam_clipseg_and_detect_contract(
        self, sample_geotiff, temp_output_dir
    ):
        from geoai_mcp_server.server import (
            detect_and_classify_features,
            segment_objects_with_prompts,
        )

        gsam_init = make_strict_callable("geoai.segment", "GroundedSAM")
        gsam_seg = make_strict_callable(
            "geoai.segment",
            "GroundedSAM.segment_image",
            return_value={"segmentation": str(temp_output_dir / "a.tif")},
        )
        clip_init = make_strict_callable("geoai.segment", "CLIPSegmentation")
        clip_seg = make_strict_callable(
            "geoai.segment",
            "CLIPSegmentation.segment_image",
            return_value=str(temp_output_dir / "c.tif"),
        )
        r2v = make_strict_callable("geoai.utils", "raster_to_vector", return_value=[1])

        class _FakeGroundedSAM:
            def __init__(self, *args, **kwargs):
                gsam_init(*args, **kwargs)

            def segment_image(self, *args, **kwargs):
                return gsam_seg(*args, **kwargs)

        class _FakeCLIPSeg:
            def __init__(self, *args, **kwargs):
                clip_init(*args, **kwargs)

            def segment_image(self, *args, **kwargs):
                return clip_seg(*args, **kwargs)

        modules = {
            "segment": SimpleNamespace(
                GroundedSAM=_FakeGroundedSAM, CLIPSegmentation=_FakeCLIPSeg
            ),
            "utils": SimpleNamespace(raster_to_vector=r2v),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r1 = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="grounded_sam",
                )
            )
            r2 = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="clipseg",
                    output_format="geojson",
                )
            )
            r3 = asyncio.run(
                detect_and_classify_features(
                    image_path=sample_geotiff.name,
                    feature_types=["buildings", "vehicles"],
                )
            )

        assert r1["success"] is True
        assert r2["success"] is True
        assert r3["success"] is True
        assert len(gsam_seg.calls) == 2
        assert len(clip_seg.calls) == 1

    def test_classify_change_canopy_contract(self, sample_geotiff, temp_output_dir):
        from geoai_mcp_server.server import (
            classify_land_cover,
            detect_temporal_changes,
            estimate_canopy_height,
        )

        model_file = sample_geotiff.parent / "model.ckpt"
        model_file.write_bytes(b"ckpt")

        classify_fn = make_strict_callable(
            "geoai.classify", "classify_image", return_value="out.tif"
        )
        changestar_fn = make_strict_callable(
            "geoai.change_detection",
            "changestar_detect",
            return_value={"change_map": np.array([[0, 1], [1, 0]], dtype=np.uint8)},
        )
        canopy_fn = make_strict_callable(
            "geoai.canopy",
            "canopy_height_estimation",
            return_value=np.array([[0.0, 12.5], [8.5, 0.0]], dtype=np.float32),
        )

        modules = {
            "classify": SimpleNamespace(classify_image=classify_fn),
            "change_detection": SimpleNamespace(changestar_detect=changestar_fn),
            "canopy": SimpleNamespace(canopy_height_estimation=canopy_fn),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_cls = asyncio.run(
                classify_land_cover(
                    image_path=sample_geotiff.name, model_path="model.ckpt"
                )
            )
            r_chg = asyncio.run(
                detect_temporal_changes(
                    image1_path=sample_geotiff.name,
                    image2_path=sample_geotiff.name,
                    change_threshold=0.4,
                )
            )
            r_chg_vec = asyncio.run(
                detect_temporal_changes(
                    image1_path=sample_geotiff.name,
                    image2_path=sample_geotiff.name,
                    change_threshold=0.4,
                    output_format="geojson",
                )
            )
            r_can = asyncio.run(estimate_canopy_height(image_path=sample_geotiff.name))

        assert r_cls["success"] is True
        assert r_chg["success"] is True
        assert r_chg["change_percentage"] == 50.0
        assert r_chg_vec["success"] is True
        assert r_chg_vec["change_percentage"] == 50.0
        assert r_can["success"] is True
        assert r_can["max_height"] == 12.5
        assert len(classify_fn.calls) == 1
        assert len(changestar_fn.calls) == 2
        assert len(canopy_fn.calls) == 1

    def test_download_and_training_and_stac_contract(
        self, sample_geotiff, temp_output_dir
    ):
        from geoai_mcp_server.server import (
            download_satellite_imagery,
            prepare_training_data,
            search_stac_imagery,
        )

        naip_fn = make_strict_callable(
            "geoai.download", "download_naip", return_value=[]
        )
        fake_item = SimpleNamespace(
            id="s2_1",
            datetime="2024-06-01",
            bbox=[-122.5, 37.7, -122.4, 37.8],
            properties={"eo:cloud_cover": 5.0},
            assets={"B04": "href"},
        )
        stac_search_fn = make_strict_callable(
            "geoai.download", "pc_stac_search", return_value=[fake_item]
        )

        def _fake_stac_download(items, output_dir, **kwargs):
            p = Path(output_dir) / "s2_1_B04.tif"
            p.write_bytes(b"tif")
            return {"s2_1": {"B04": str(p)}}

        stac_dl_fn = make_strict_callable(
            "geoai.download", "pc_stac_download", return_value=_fake_stac_download
        )
        tiles_batch_fn = make_strict_callable(
            "geoai.utils",
            "export_geotiff_tiles_batch",
            return_value={"total_tiles": 10},
        )
        tiles_single_fn = make_strict_callable(
            "geoai.utils", "export_geotiff_tiles", return_value=None
        )

        img_dir = sample_geotiff.parent / "imgs"
        lbl_dir = sample_geotiff.parent / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        modules = {
            "download": SimpleNamespace(
                download_naip=naip_fn,
                pc_stac_search=stac_search_fn,
                pc_stac_download=stac_dl_fn,
            ),
            "utils": SimpleNamespace(
                export_geotiff_tiles_batch=tiles_batch_fn,
                export_geotiff_tiles=tiles_single_fn,
            ),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_naip = asyncio.run(
                download_satellite_imagery(
                    -122.5, 37.7, -122.4, 37.8, data_source="naip"
                )
            )
            r_s2 = asyncio.run(
                download_satellite_imagery(
                    -122.5, 37.7, -122.4, 37.8, data_source="sentinel2"
                )
            )
            r_stac = asyncio.run(search_stac_imagery(-122.5, 37.7, -122.4, 37.8))
            r_batch = asyncio.run(
                prepare_training_data(images_dir="imgs", labels_dir="lbls")
            )
            r_single = asyncio.run(
                prepare_training_data(
                    images_dir=sample_geotiff.name, labels_dir=sample_geotiff.name
                )
            )

        assert r_naip["success"] is True
        assert r_s2["success"] is True
        assert r_stac["success"] is True
        assert r_batch["success"] is True
        assert r_batch["num_tiles"] == 10
        assert r_single["success"] is True
        assert len(naip_fn.calls) == 1
        assert len(stac_search_fn.calls) == 2
        assert len(stac_dl_fn.calls) == 1
        assert len(tiles_batch_fn.calls) == 1
        assert len(tiles_single_fn.calls) == 1

    def test_foundation_vlm_and_regularize_contract(
        self, sample_geotiff, sample_geojson, temp_output_dir
    ):
        from geoai_mcp_server.server import (
            analyze_with_vision_language_model,
            clean_segmentation_results,
            extract_features_with_foundation_model,
        )

        sim_map_fn = make_strict_callable(
            "geoai.dinov3",
            "create_similarity_map",
            return_value={"similarity_map": np.zeros((16, 16))},
        )
        dino_init = make_strict_callable("geoai.dinov3", "DINOv3GeoProcessor")
        dino_load = make_strict_callable(
            "geoai.dinov3",
            "DINOv3GeoProcessor.load_image",
            return_value=(np.zeros((3, 16, 16), dtype=np.uint8), {}),
        )
        dino_pre = make_strict_callable(
            "geoai.dinov3",
            "DINOv3GeoProcessor.preprocess_image_for_dinov3",
            return_value=MagicMock(),
        )
        dino_feat = make_strict_callable(
            "geoai.dinov3",
            "DINOv3GeoProcessor.extract_features",
            return_value=(np.zeros((4, 8), dtype=np.float32), 2, 2),
        )

        import torch

        prithvi_init = make_strict_callable("geoai.prithvi", "PrithviProcessor")
        prithvi_load = make_strict_callable(
            "geoai.prithvi",
            "PrithviProcessor.load_images",
            return_value=(
                np.zeros((1, 6, 1, 16, 16), dtype=np.float32),
                [{}],
                [],
                [],
            ),
        )
        prithvi_enc = make_strict_callable(
            "geoai.prithvi",
            "PrithviViT.forward",
            return_value=(
                torch.zeros((1, 5, 32), dtype=torch.float32),
                torch.zeros((1, 4), dtype=torch.float32),
                torch.arange(4, dtype=torch.long).unsqueeze(0),
            ),
        )

        md_caption = make_strict_callable(
            "geoai.moondream", "moondream_caption", return_value="An urban scene"
        )
        md_query = make_strict_callable(
            "geoai.moondream", "moondream_query", return_value="Two buildings"
        )
        md_detect = make_strict_callable(
            "geoai.moondream",
            "moondream_detect",
            return_value={"objects": [{"x_min": 0.1, "y_min": 0.1}]},
        )
        md_point = make_strict_callable(
            "geoai.moondream",
            "moondream_point",
            return_value={"points": [{"x": 0.5, "y": 0.5}]},
        )
        reg_fn = make_strict_callable("geoai.utils", "regularize", return_value=None)

        class _FakeDino:
            def __init__(self, *args, **kwargs):
                dino_init(*args, **kwargs)

            def load_image(self, *args, **kwargs):
                return dino_load(*args, **kwargs)

            def preprocess_image_for_dinov3(self, *args, **kwargs):
                return dino_pre(*args, **kwargs)

            def extract_features(self, *args, **kwargs):
                return dino_feat(*args, **kwargs)

        class _FakePrithviEncoder:
            def __call__(self, *args, **kwargs):
                return prithvi_enc(*args, **kwargs)

        class _FakePrithvi:
            def __init__(self, *args, **kwargs):
                prithvi_init(*args, **kwargs)
                self.img_size = 16
                self.device = torch.device("cpu")
                self.model = SimpleNamespace(encoder=_FakePrithviEncoder())

            def load_images(self, *args, **kwargs):
                return prithvi_load(*args, **kwargs)

        modules = {
            "dinov3": SimpleNamespace(
                create_similarity_map=sim_map_fn, DINOv3GeoProcessor=_FakeDino
            ),
            "prithvi": SimpleNamespace(PrithviProcessor=_FakePrithvi),
            "moondream": SimpleNamespace(
                moondream_caption=md_caption,
                moondream_query=md_query,
                moondream_detect=md_detect,
                moondream_point=md_point,
            ),
            "utils": SimpleNamespace(regularize=reg_fn),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_d_emb = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name, model="dinov3"
                )
            )
            r_d_sim = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="dinov3",
                    output_type="similarity_map",
                    reference_x=10,
                    reference_y=20,
                )
            )
            r_pr = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name, model="prithvi"
                )
            )
            r_cap = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name, task="caption"
                )
            )
            r_qry = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="query",
                    query="How many buildings?",
                )
            )
            r_det = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="detect",
                    detect_target="building",
                )
            )
            r_pt = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="point",
                    detect_target="building",
                )
            )
            r_reg = asyncio.run(
                clean_segmentation_results(
                    input_path=sample_geojson.name,
                    operation="regularize",
                    regularize_buildings=True,
                )
            )

        assert r_d_emb["success"] is True
        assert r_d_sim["success"] is True
        assert r_pr["success"] is True
        assert r_cap["success"] is True and r_cap["caption"] == "An urban scene"
        assert r_qry["success"] is True and r_qry["answer"] == "Two buildings"
        assert r_det["success"] is True and len(r_det["detections"]) == 1
        assert r_pt["success"] is True and len(r_pt["detections"]) == 1
        assert r_reg["success"] is True
        assert len(reg_fn.calls) == 1
