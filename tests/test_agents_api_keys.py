"""Credential handling in model factories without optional agent dependencies."""

import ast
from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


class TestModelApiKeys(unittest.TestCase):
    """Execute the real factory bodies with mocked model constructors."""

    def setUp(self):
        """Load factory bodies with mocked model constructors."""
        source = Path(__file__).resolve().parents[1] / "geoai/agents/geo_agents.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"create_openai_model", "create_anthropic_model"}
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                )
            ]
            + functions,
            type_ignores=[],
        )
        self.constructors = {"OpenAIModel": Mock(), "AnthropicModel": Mock()}
        self.namespace = {"os": os, **self.constructors}
        exec(
            compile(ast.fix_missing_locations(module), str(source), "exec"),
            self.namespace,
        )

    def check_keys(self, options, expected=None, environment=None):
        """Check credential resolution for both model providers."""
        for provider, constructor in (
            ("openai", "OpenAIModel"),
            ("anthropic", "AnthropicModel"),
        ):
            with self.subTest(provider=provider):
                env_name = f"{provider.upper()}_API_KEY"
                env = {env_name: environment} if environment is not None else {}
                factory = self.namespace[f"create_{provider}_model"]
                with patch.dict(os.environ, env, clear=True):
                    if expected is None:
                        with self.assertRaisesRegex(ValueError, env_name):
                            factory(**deepcopy(options))
                        self.constructors[constructor].assert_not_called()
                    else:
                        result = factory(**deepcopy(options))
                        model = self.constructors[constructor]
                        self.assertIs(result, model.return_value)
                        self.assertEqual(
                            model.call_args.kwargs["client_args"]["api_key"], expected
                        )

    def test_client_key_without_environment(self):
        """Client key without environment."""
        self.check_keys({"client_args": {"api_key": "client-key"}}, "client-key")

    def test_client_key_takes_precedence(self):
        """Client key takes precedence."""
        self.check_keys(
            {"client_args": {"api_key": "client-key"}, "api_key": "explicit-key"},
            "client-key",
            environment="env-key",
        )

    def test_explicit_key_without_environment(self):
        """Explicit key without environment."""
        self.check_keys({"api_key": "explicit-key"}, "explicit-key")

    def test_environment_fallback(self):
        """Environment fallback."""
        self.check_keys({"client_args": {}}, "env-key", environment="env-key")

    def test_missing_key(self):
        """Missing key."""
        self.check_keys({"client_args": {}})

    def test_null_client_key_with_environment(self):
        """Null client key with environment."""
        self.check_keys(
            {"client_args": {"api_key": None}}, "env-key", environment="env-key"
        )

    def test_null_client_key_without_environment(self):
        """Null client key without environment."""
        self.check_keys({"client_args": {"api_key": None}})

    def test_empty_client_key_without_environment(self):
        """Empty client key without environment."""
        self.check_keys({"client_args": {"api_key": ""}})

    def test_empty_client_key_with_environment(self):
        """Use the environment key when the client key is empty."""
        self.check_keys(
            {"client_args": {"api_key": ""}}, "env-key", environment="env-key"
        )

    def test_anthropic_token_preserves_null_key(self):
        """Keep token authentication with explicit, environment, or absent keys."""
        for explicit, environment in (
            (None, None),
            ("explicit-key", None),
            (None, "env-key"),
        ):
            with self.subTest(explicit=explicit, environment=environment):
                env = {"ANTHROPIC_API_KEY": environment} if environment else {}
                options = {"api_key": None, "auth_token": "token"}
                with patch.dict(os.environ, env, clear=True):
                    result = self.namespace["create_anthropic_model"](
                        api_key=explicit, client_args=options
                    )
                model = self.constructors["AnthropicModel"]
                self.assertIs(result, model.return_value)
                self.assertEqual(
                    model.call_args.kwargs["client_args"],
                    {"api_key": None, "auth_token": "token"},
                )


if __name__ == "__main__":
    unittest.main()
