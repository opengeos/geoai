#!/usr/bin/env python

"""Tests for the default model IDs of the `geoai.agents` model factories."""

import ast
import os
import unittest

# geo_agents.py has heavy dependencies (strands, leafmap, boto3, etc.), so the
# defaults are read from the source instead of importing the module.
_geo_agents_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "geoai", "agents", "geo_agents.py"
)


def _default_model_id(func_name):
    """Return the default value of the model_id parameter of a function."""
    with open(_geo_agents_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args.args
            defaults = dict(
                zip(
                    [a.arg for a in args[-len(node.args.defaults) :]],
                    node.args.defaults,
                )
            )
            return ast.literal_eval(defaults["model_id"])
    raise AssertionError(f"{func_name} not found")


class TestDefaultModelIds(unittest.TestCase):
    """The default model IDs must be models the providers still serve."""

    def test_anthropic_default(self):
        """Claude Sonnet 4 was retired on the Anthropic API on 2026-06-15."""
        self.assertEqual(
            _default_model_id("create_anthropic_model"), "claude-sonnet-5-5"
        )

    def test_bedrock_default(self):
        """Claude Sonnet 4 reached end of life on Amazon Bedrock on 2026-10-14."""
        self.assertEqual(
            _default_model_id("create_bedrock_model"),
            "global.anthropic.claude-sonnet-4-6",
        )

    def test_gemini_default(self):
        """New Gemini API keys get a 404 for gemini-2.5-flash."""
        self.assertEqual(_default_model_id("create_gemini_model"), "gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main()
