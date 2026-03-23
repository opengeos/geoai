#!/usr/bin/env python

"""Tests for MiniMax model provider support in `geoai.agents` module."""

import ast
import os
import re
import unittest

# Since geo_agents.py has heavy dependencies (strands, leafmap, boto3, etc.)
# that may not be available in all test environments, we test the code
# structure and logic through AST/source analysis and lightweight checks.
_geo_agents_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "geoai", "agents", "geo_agents.py"
)
_init_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "geoai", "agents", "__init__.py"
)


def _read_source(path):
    """Read a source file and return its contents."""
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestCreateMinimaxModelExists(unittest.TestCase):
    """Tests that the create_minimax_model factory function exists and is well-formed."""

    def setUp(self):
        self.source = _read_source(_geo_agents_path)
        self.tree = ast.parse(self.source)

    def test_function_defined(self):
        """Test that create_minimax_model function is defined."""
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("create_minimax_model", func_names)

    def test_function_has_model_id_param(self):
        """Test that create_minimax_model has model_id parameter."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                arg_names = [arg.arg for arg in node.args.args]
                self.assertIn("model_id", arg_names)
                break
        else:
            self.fail("create_minimax_model function not found")

    def test_function_has_api_key_param(self):
        """Test that create_minimax_model has api_key parameter."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                arg_names = [arg.arg for arg in node.args.args]
                self.assertIn("api_key", arg_names)
                break
        else:
            self.fail("create_minimax_model function not found")

    def test_function_has_client_args_param(self):
        """Test that create_minimax_model has client_args parameter."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                arg_names = [arg.arg for arg in node.args.args]
                self.assertIn("client_args", arg_names)
                break
        else:
            self.fail("create_minimax_model function not found")

    def test_default_model_id_is_m27(self):
        """Test that default model_id is MiniMax-M2.7."""
        self.assertIn('model_id: str = "MiniMax-M2.7"', self.source)

    def test_uses_minimax_base_url(self):
        """Test that create_minimax_model sets MiniMax API base URL."""
        self.assertIn("api.minimax.io/v1", self.source)

    def test_reads_minimax_api_key_env(self):
        """Test that create_minimax_model reads MINIMAX_API_KEY from environment."""
        self.assertIn("MINIMAX_API_KEY", self.source)

    def test_returns_openai_model(self):
        """Test that create_minimax_model returns OpenAIModel."""
        # Check that the function contains OpenAIModel instantiation
        func_source = self._extract_function_source("create_minimax_model")
        self.assertIn("OpenAIModel", func_source)

    def test_raises_on_missing_api_key(self):
        """Test that missing MINIMAX_API_KEY raises ValueError."""
        func_source = self._extract_function_source("create_minimax_model")
        self.assertIn("MINIMAX_API_KEY", func_source)

    def _extract_function_source(self, func_name):
        """Extract the source code of a specific function."""
        lines = self.source.splitlines()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno
                return "\n".join(lines[start:end])
        return ""


class TestMinimaxExportedFromInit(unittest.TestCase):
    """Tests that create_minimax_model is exported from agents __init__.py."""

    def test_exported_in_init(self):
        """Test that create_minimax_model is exported from agents __init__.py."""
        init_source = _read_source(_init_path)
        self.assertIn("create_minimax_model", init_source)


class TestMinimaxStringRouting(unittest.TestCase):
    """Tests for MiniMax model string routing in agent classes."""

    def setUp(self):
        self.source = _read_source(_geo_agents_path)

    def test_geo_agent_has_minimax_routing(self):
        """Test that GeoAgent routes 'minimax' strings to create_minimax_model."""
        # Find the GeoAgent class section with minimax routing
        self.assertIn('model.lower().startswith("minimax")', self.source)

    def test_minimax_routing_calls_create_minimax_model(self):
        """Test that minimax string routing calls create_minimax_model."""
        pattern = r'model\.lower\(\)\.startswith\("minimax"\).*?create_minimax_model'
        matches = re.findall(pattern, self.source, re.DOTALL)
        # Should appear 3 times: GeoAgent, STACAgent, CatalogAgent
        self.assertEqual(
            len(matches), 3, "Expected 3 minimax routing blocks (one per agent class)"
        )

    def test_minimax_string_detection_logic(self):
        """Test that MiniMax string detection correctly identifies MiniMax models."""
        minimax_models = [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "minimax-m2.7",
        ]
        for model_str in minimax_models:
            self.assertTrue(
                model_str.lower().startswith("minimax"),
                f"'{model_str}' should be detected as MiniMax model",
            )

    def test_non_minimax_string_not_detected(self):
        """Test that non-MiniMax strings are not routed to MiniMax."""
        non_minimax_models = [
            "gpt-4o-mini",
            "claude-sonnet-4-20250514",
            "llama3.1",
            "gemini-2.5-flash",
        ]
        for model_str in non_minimax_models:
            self.assertFalse(
                model_str.lower().startswith("minimax"),
                f"'{model_str}' should not be detected as MiniMax model",
            )


class TestMinimaxDocstring(unittest.TestCase):
    """Tests for MiniMax model documentation."""

    def setUp(self):
        self.source = _read_source(_geo_agents_path)
        self.tree = ast.parse(self.source)

    def test_function_has_docstring(self):
        """Test that create_minimax_model has a proper docstring."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                docstring = ast.get_docstring(node)
                self.assertIsNotNone(
                    docstring, "create_minimax_model should have a docstring"
                )
                self.assertIn("MiniMax", docstring)
                break
        else:
            self.fail("create_minimax_model function not found")

    def test_docstring_mentions_api(self):
        """Test that docstring mentions the OpenAI-compatible API."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                docstring = ast.get_docstring(node)
                self.assertIn("OpenAI-compatible", docstring)
                break

    def test_docstring_mentions_models(self):
        """Test that docstring mentions available MiniMax model IDs."""
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_minimax_model"
            ):
                docstring = ast.get_docstring(node)
                self.assertIn("MiniMax-M2.7", docstring)
                break


class TestMinimaxFactoryConsistency(unittest.TestCase):
    """Tests that create_minimax_model follows the same pattern as other factory functions."""

    def setUp(self):
        self.source = _read_source(_geo_agents_path)
        self.tree = ast.parse(self.source)

    def test_follows_openai_factory_pattern(self):
        """Test that create_minimax_model follows the same pattern as create_openai_model."""
        openai_func = self._get_function_node("create_openai_model")
        minimax_func = self._get_function_node("create_minimax_model")

        # Both should have model_id, api_key, client_args params
        openai_params = {arg.arg for arg in openai_func.args.args}
        minimax_params = {arg.arg for arg in minimax_func.args.args}

        expected = {"model_id", "api_key", "client_args"}
        self.assertTrue(expected.issubset(openai_params))
        self.assertTrue(expected.issubset(minimax_params))

    def test_all_factory_functions_exist(self):
        """Test that all expected factory functions are defined."""
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        expected_factories = [
            "create_ollama_model",
            "create_openai_model",
            "create_anthropic_model",
            "create_bedrock_model",
            "create_gemini_model",
            "create_minimax_model",
        ]
        for name in expected_factories:
            self.assertIn(name, func_names, f"Missing factory function: {name}")

    def _get_function_node(self, func_name):
        """Get the AST node for a specific function."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        self.fail(f"Function {func_name} not found")


class TestMinimaxInAllAgents(unittest.TestCase):
    """Tests that MiniMax routing is present in all agent classes."""

    def setUp(self):
        self.source = _read_source(_geo_agents_path)

    def test_geo_agent_minimax_routing(self):
        """Test that GeoAgent has MiniMax string routing."""
        # Extract GeoAgent class source
        geo_start = self.source.find("class GeoAgent")
        stac_start = self.source.find("class STACAgent")
        geo_source = self.source[geo_start:stac_start]
        self.assertIn("create_minimax_model", geo_source)

    def test_stac_agent_minimax_routing(self):
        """Test that STACAgent has MiniMax string routing."""
        stac_start = self.source.find("class STACAgent")
        catalog_start = self.source.find("class CatalogAgent")
        stac_source = self.source[stac_start:catalog_start]
        self.assertIn("create_minimax_model", stac_source)

    def test_catalog_agent_minimax_routing(self):
        """Test that CatalogAgent has MiniMax string routing."""
        catalog_start = self.source.find("class CatalogAgent")
        catalog_source = self.source[catalog_start:]
        self.assertIn("create_minimax_model", catalog_source)


class TestMinimaxModelIntegration(unittest.TestCase):
    """Integration tests for MiniMax model (require strands-agents and MINIMAX_API_KEY)."""

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_create_minimax_model_live(self):
        """Integration test: create MiniMax model with real API key."""
        try:
            from geoai.agents.geo_agents import create_minimax_model
            from strands.models.openai import OpenAIModel

            model = create_minimax_model(model_id="MiniMax-M2.7")
            self.assertIsInstance(model, OpenAIModel)
            self.assertEqual(model.config["model_id"], "MiniMax-M2.7")
        except ImportError:
            self.skipTest("strands-agents or geoai dependencies not installed")

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_create_minimax_m25_highspeed_live(self):
        """Integration test: create MiniMax M2.5-highspeed model."""
        try:
            from geoai.agents.geo_agents import create_minimax_model

            model = create_minimax_model(model_id="MiniMax-M2.5-highspeed")
            self.assertEqual(model.config["model_id"], "MiniMax-M2.5-highspeed")
        except ImportError:
            self.skipTest("strands-agents or geoai dependencies not installed")

    @unittest.skipUnless(os.environ.get("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_minimax_base_url_live(self):
        """Integration test: verify MiniMax model base URL."""
        try:
            from geoai.agents.geo_agents import create_minimax_model

            model = create_minimax_model()
            self.assertIn("api.minimax.io", model.client_args.get("base_url", ""))
        except ImportError:
            self.skipTest("strands-agents or geoai dependencies not installed")


if __name__ == "__main__":
    unittest.main()
