"""Tests for the lazy import infrastructure in geoai/__init__.py.

These tests verify that ``import geoai`` is fast, that all public symbols
are discoverable, and that backward-compatible access patterns still work.
"""

import importlib
import sys
import unittest


class TestLazyImportInfrastructure(unittest.TestCase):
    """Tests for the PEP 562 lazy import mechanism."""

    def test_import_is_fast(self):
        """import geoai must complete without loading heavy dependencies."""
        # Force a fresh import
        saved = sys.modules.pop("geoai", None)
        try:
            import time

            start = time.monotonic()
            importlib.import_module("geoai")
            elapsed = time.monotonic() - start
            # Should be well under 1 second — no torch/geopandas/leafmap loaded
            self.assertLess(
                elapsed,
                2.0,
                f"import geoai took {elapsed:.3f}s — expected < 2.0s",
            )
        finally:
            if saved is not None:
                sys.modules["geoai"] = saved

    def test_version_accessible(self):
        """__version__ must be available without triggering lazy imports."""
        import geoai

        self.assertIsNotNone(geoai.__version__)
        self.assertRegex(geoai.__version__, r"^\d+\.\d+\.\d+")

    def test_all_lazy_symbols_in_dir(self):
        """Every symbol in _LAZY_SYMBOL_MAP must appear in dir(geoai)."""
        import geoai

        geoai_dir = dir(geoai)
        for name in geoai._LAZY_SYMBOL_MAP:
            self.assertIn(
                name,
                geoai_dir,
                f"Lazy symbol {name!r} missing from dir(geoai)",
            )

    def test_all_lazy_submodules_in_dir(self):
        """Every entry in _LAZY_SUBMODULES must appear in dir(geoai)."""
        import geoai

        geoai_dir = dir(geoai)
        for name in geoai._LAZY_SUBMODULES:
            self.assertIn(
                name,
                geoai_dir,
                f"Lazy submodule {name!r} missing from dir(geoai)",
            )

    def test_all_contains_all_lazy_symbols(self):
        """__all__ must include every lazy symbol for `from geoai import *`."""
        import geoai

        all_set = set(geoai.__all__)
        for name in geoai._LAZY_SYMBOL_MAP:
            self.assertIn(
                name,
                all_set,
                f"Lazy symbol {name!r} missing from __all__",
            )

    def test_nonexistent_symbol_raises_attribute_error(self):
        """Accessing a nonexistent attribute must raise AttributeError."""
        import geoai

        with self.assertRaises(AttributeError):
            _ = geoai.this_symbol_does_not_exist_at_all

    def test_pipeline_symbols_in_lazy_map(self):
        """Pipeline symbols must be accessible via lazy loading."""
        import geoai

        pipeline_symbols = [
            "Pipeline",
            "PipelineStep",
            "FunctionStep",
            "GlobStep",
            "PipelineResult",
            "load_pipeline",
            "register_step",
        ]
        for name in pipeline_symbols:
            self.assertIn(name, geoai._LAZY_SYMBOL_MAP)

    def test_key_symbols_present(self):
        """Critical public API symbols must be listed."""
        import geoai

        expected = [
            # Core
            "LeafMap",
            "Map",
            # Models
            "DINOv3GeoProcessor",
            "MoondreamGeo",
            "CanopyHeightEstimation",
            "PrithviProcessor",
            # Training
            "semantic_segmentation",
            "train_segmentation_model",
            "train_timm_classifier",
            # Utils
            "orthogonalize",
            "get_raster_info",
            "download_file",
            # Data
            "download_naip",
            "segment_water",
        ]
        geoai_dir = dir(geoai)
        for name in expected:
            self.assertIn(name, geoai_dir, f"Expected symbol {name!r} missing")

    def test_alias_mapping(self):
        """plot_classification_history should map to recognize.plot_training_history."""
        import geoai

        module_rel, original_name = geoai._LAZY_SYMBOL_MAP[
            "plot_classification_history"
        ]
        self.assertEqual(module_rel, "recognize")
        self.assertEqual(original_name, "plot_training_history")

    def test_no_torch_in_sys_modules_after_import(self):
        """Verify that importing geoai does not pull torch into sys.modules."""
        # Remove all geoai modules
        saved_geoai = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.startswith("geoai")
        }
        # Remove all torch modules (torch, torch.nn, torch.cuda, etc.)
        saved_torch = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "torch" or k.startswith("torch.")
        }

        try:
            importlib.import_module("geoai")
            torch_loaded = [
                k
                for k in sys.modules
                if k == "torch" or k.startswith("torch.")
            ]
            self.assertFalse(
                torch_loaded,
                f"torch was loaded during import geoai — lazy loading is broken. "
                f"Found: {torch_loaded}",
            )
        finally:
            # Restore original state
            for k, v in saved_geoai.items():
                sys.modules[k] = v
            for k, v in saved_torch.items():
                sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
