#!/usr/bin/env python

"""Tests for `geoai.foundation_models` module."""

import unittest

import pandas as pd


class TestFoundationModelRegistry(unittest.TestCase):
    """Tests for the FOUNDATION_MODELS registry structure and integrity."""

    def _get_registry(self):
        from geoai.foundation_models import FOUNDATION_MODELS

        return FOUNDATION_MODELS

    def test_registry_is_dict(self):
        """FOUNDATION_MODELS must be a plain dict."""
        registry = self._get_registry()
        self.assertIsInstance(registry, dict)

    def test_registry_not_empty(self):
        """Registry must contain at least 10 entries."""
        registry = self._get_registry()
        self.assertGreaterEqual(len(registry), 10)

    def test_all_entries_have_required_keys(self):
        """Every registry entry must contain all 14 required keys."""
        required_keys = {
            "name",
            "abbreviation",
            "category",
            "modality",
            "tasks",
            "backbone",
            "publication",
            "year",
            "paper_url",
            "code_url",
            "huggingface_id",
            "license",
            "terratorch_supported",
            "terratorch_key",
            "description",
        }
        registry = self._get_registry()
        for key, entry in registry.items():
            missing = required_keys - set(entry.keys())
            self.assertEqual(
                missing,
                set(),
                msg=f"Entry '{key}' is missing keys: {missing}",
            )

    def test_no_none_in_required_string_fields(self):
        """Non-nullable string fields must never be None or empty."""
        non_nullable = (
            "name",
            "abbreviation",
            "category",
            "modality",
            "backbone",
            "publication",
            "description",
        )
        registry = self._get_registry()
        for key, entry in registry.items():
            for field in non_nullable:
                self.assertIsNotNone(
                    entry[field],
                    msg=f"Entry '{key}' has None for required field '{field}'",
                )
                self.assertGreater(
                    len(entry[field]),
                    0,
                    msg=f"Entry '{key}' has empty string for required field '{field}'",
                )

    def test_category_values_are_valid(self):
        """Every entry's category must be from the allowed vocabulary."""
        from geoai.foundation_models import _VALID_CATEGORIES

        registry = self._get_registry()
        for key, entry in registry.items():
            self.assertIn(
                entry["category"],
                _VALID_CATEGORIES,
                msg=f"Entry '{key}' has invalid category '{entry['category']}'",
            )

    def test_modality_values_are_valid(self):
        """Every entry's modality must be from the allowed vocabulary."""
        from geoai.foundation_models import _VALID_MODALITIES

        registry = self._get_registry()
        for key, entry in registry.items():
            self.assertIn(
                entry["modality"],
                _VALID_MODALITIES,
                msg=f"Entry '{key}' has invalid modality '{entry['modality']}'",
            )

    def test_tasks_is_nonempty_list_of_strings(self):
        """tasks must be a non-empty list of lowercase strings."""
        registry = self._get_registry()
        for key, entry in registry.items():
            tasks = entry["tasks"]
            self.assertIsInstance(
                tasks, list, msg=f"Entry '{key}': tasks is not a list"
            )
            self.assertGreater(
                len(tasks), 0, msg=f"Entry '{key}': tasks list is empty"
            )
            for t in tasks:
                self.assertIsInstance(
                    t, str, msg=f"Entry '{key}': task element is not a string"
                )

    def test_year_is_int_in_reasonable_range(self):
        """year must be an integer between 2018 and 2030 inclusive."""
        registry = self._get_registry()
        for key, entry in registry.items():
            self.assertIsInstance(
                entry["year"], int, msg=f"Entry '{key}': year is not an int"
            )
            self.assertGreaterEqual(
                entry["year"], 2018, msg=f"Entry '{key}': year < 2018"
            )
            self.assertLessEqual(
                entry["year"], 2030, msg=f"Entry '{key}': year > 2030"
            )

    def test_terratorch_supported_is_strict_bool(self):
        """terratorch_supported must be a Python bool (not just truthy)."""
        registry = self._get_registry()
        for key, entry in registry.items():
            self.assertIsInstance(
                entry["terratorch_supported"],
                bool,
                msg=f"Entry '{key}': terratorch_supported is not a bool",
            )

    def test_description_is_long_enough(self):
        """description must be at least 20 characters."""
        registry = self._get_registry()
        for key, entry in registry.items():
            self.assertGreater(
                len(entry["description"]),
                20,
                msg=f"Entry '{key}': description is too short",
            )

    def test_terratorch_key_set_when_supported(self):
        """Every terratorch_supported=True entry must have a non-None terratorch_key."""
        registry = self._get_registry()
        for key, entry in registry.items():
            if entry["terratorch_supported"]:
                self.assertIsNotNone(
                    entry["terratorch_key"],
                    msg=f"Entry '{key}' is terratorch_supported but terratorch_key is None",
                )

    def test_huggingface_id_format_when_set(self):
        """When huggingface_id is not None it must contain a '/' separator."""
        registry = self._get_registry()
        for key, entry in registry.items():
            hf_id = entry["huggingface_id"]
            if hf_id is not None:
                self.assertIn(
                    "/",
                    hf_id,
                    msg=(
                        f"Entry '{key}': huggingface_id '{hf_id}' "
                        "does not look like 'org/repo'"
                    ),
                )


class TestListFoundationModels(unittest.TestCase):
    """Tests for the list_foundation_models function."""

    def _list(self, **kwargs):
        from geoai.foundation_models import list_foundation_models

        kwargs.setdefault("verbose", False)
        return list_foundation_models(**kwargs)

    def test_returns_dataframe_by_default(self):
        """Default call must return a non-empty DataFrame."""
        result = self._list()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_dataframe_has_expected_columns(self):
        """DataFrame must include key summary columns."""
        expected = {
            "name",
            "abbreviation",
            "category",
            "modality",
            "tasks",
            "year",
            "publication",
            "terratorch_supported",
            "terratorch_key",
            "huggingface_id",
        }
        result = self._list()
        self.assertTrue(expected.issubset(set(result.columns)))

    def test_returns_dict_when_as_dataframe_false(self):
        """as_dataframe=False must return a dict."""
        result = self._list(as_dataframe=False)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_filter_by_category_vision(self):
        """Filter category='vision' must only return vision models."""
        result = self._list(category="vision")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["category"] == "vision"))
        self.assertGreater(len(result), 0)

    def test_filter_by_category_vision_language(self):
        """Filter category='vision-language' must only return VLM models."""
        result = self._list(category="vision-language")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["category"] == "vision-language"))

    def test_filter_invalid_category_raises_value_error(self):
        """An unknown category must raise ValueError with a helpful message."""
        with self.assertRaises(ValueError) as ctx:
            self._list(category="not-a-real-category")
        self.assertIn("not-a-real-category", str(ctx.exception))

    def test_filter_by_modality(self):
        """Filter by modality must return only matching rows."""
        result = self._list(modality="multispectral")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["modality"] == "multispectral"))
        self.assertGreater(len(result), 0)

    def test_filter_invalid_modality_raises_value_error(self):
        """An unknown modality must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self._list(modality="infrared-xray")
        self.assertIn("infrared-xray", str(ctx.exception))

    def test_filter_by_task_segmentation(self):
        """Filter task='segmentation' must return only segmentation models."""
        result = self._list(task="segmentation")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all("segmentation" in row for row in result["tasks"]),
            msg="Some returned rows do not support segmentation",
        )

    def test_filter_terratorch_only(self):
        """terratorch_only=True must return only terratorch-supported models."""
        result = self._list(terratorch_only=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["terratorch_supported"]))
        self.assertGreater(len(result), 0)

    def test_filter_huggingface_only(self):
        """huggingface_only=True must return only models with HF IDs."""
        result = self._list(huggingface_only=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["huggingface_id"] != ""))
        self.assertGreater(len(result), 0)

    def test_multiple_filters_combined(self):
        """Combining category + terratorch_only must apply both filters."""
        result = self._list(category="vision", terratorch_only=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(result["category"] == "vision"))
        self.assertTrue(all(result["terratorch_supported"]))

    def test_empty_result_returns_empty_dataframe_not_error(self):
        """A filter combination that matches nothing returns an empty DataFrame."""
        result = self._list(category="agents")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_verbose_false_does_not_raise(self):
        """verbose=False must complete without error or side effects."""
        result = self._list(verbose=False)
        self.assertIsInstance(result, pd.DataFrame)


class TestGetFoundationModelInfo(unittest.TestCase):
    """Tests for the get_foundation_model_info function."""

    def _get(self, name):
        from geoai.foundation_models import get_foundation_model_info

        return get_foundation_model_info(name)

    def test_known_model_returns_dict(self):
        """A known model key must return a dict."""
        result = self._get("prithvi-eo-2.0-300m")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["year"], 2024)

    def test_returns_copy_not_original_reference(self):
        """Mutating the returned dict must not affect the registry."""
        from geoai.foundation_models import FOUNDATION_MODELS

        original_description = FOUNDATION_MODELS["clay-v1"]["description"]
        info = self._get("clay-v1")
        info["description"] = "MUTATED"
        self.assertEqual(
            FOUNDATION_MODELS["clay-v1"]["description"],
            original_description,
        )

    def test_unknown_model_raises_value_error(self):
        """An unknown model key must raise ValueError."""
        with self.assertRaises(ValueError):
            self._get("no-such-model-xyz")

    def test_error_message_contains_queried_name(self):
        """The ValueError message must echo the queried name."""
        with self.assertRaises(ValueError) as ctx:
            self._get("ghost-model")
        self.assertIn("ghost-model", str(ctx.exception))

    def test_error_message_lists_available_models(self):
        """The ValueError message must list available model keys."""
        with self.assertRaises(ValueError) as ctx:
            self._get("ghost-model")
        self.assertIn("prithvi-eo-2.0-300m", str(ctx.exception))


class TestCheckTerratorch(unittest.TestCase):
    """Tests for the check_terratorch_available helper."""

    def test_returns_strict_bool(self):
        """Return value must be a Python bool regardless of install state."""
        from geoai.foundation_models import check_terratorch_available

        result = check_terratorch_available()
        self.assertIsInstance(result, bool)


class TestLazyImport(unittest.TestCase):
    """Tests for lazy import wiring in geoai.__init__."""

    def test_symbols_accessible_from_top_level_geoai(self):
        """All public symbols must be importable from the top-level geoai package."""
        import geoai

        for attr in (
            "FOUNDATION_MODELS",
            "list_foundation_models",
            "get_foundation_model_info",
            "check_terratorch_available",
            "load_foundation_model",
        ):
            self.assertTrue(
                hasattr(geoai, attr),
                msg=f"geoai.{attr} is not accessible",
            )

    def test_submodule_importable_directly(self):
        """``from geoai import foundation_models`` must work."""
        from geoai import foundation_models

        self.assertTrue(hasattr(foundation_models, "FOUNDATION_MODELS"))
        self.assertTrue(hasattr(foundation_models, "list_foundation_models"))

    def test_list_via_top_level_geoai(self):
        """geoai.list_foundation_models() must run end-to-end via lazy import."""
        import geoai

        df = geoai.list_foundation_models(verbose=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
